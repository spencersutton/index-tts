import sys
import time
from collections.abc import Collection, Iterable, Sequence
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.types import Number
from transformers import GPT2Model

from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings

from .attention import ForwardContext
from .kv_manager import KVCacheManager, Seq


class Sampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.compile(dynamic=True, mode="reduce-overhead")
    def forward(self, logits: Tensor, temperatures: Tensor) -> Tensor:
        temperatures = temperatures.to(logits.device).clamp(min=1e-8)
        greedy_mask = temperatures < 1e-5
        temp_for_scaling = torch.where(greedy_mask, 1.0, temperatures)
        scaled_logits = logits / temp_for_scaling.unsqueeze(-1)
        probs = torch.softmax(scaled_logits, dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        sampled_tokens = probs.div_(q).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)


GRAPH_BS: Final = [1, 2, 4, 8]
MAX_BS: Final = 8  # Support up to batch size 8


class AccelInferenceEngine:
    _tts_mode: bool = False
    _tts_prompt_len: int = 0

    def __init__(
        self,
        model: GPT2Model,
        lm_head: nn.Sequential,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 256,
        num_blocks: int = 128,
        use_cuda_graph: bool = True,
    ) -> None:
        """Args:
        model: The GPT transformer model (should have accel attention)
        lm_head: Language model head for generating logits
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        block_size: KV cache block size
        num_blocks: Total number of KV cache blocks
        use_cuda_graph: Whether to use CUDA Graph for decode optimization.

        """
        self.model = model
        self.lm_head = lm_head
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        if hasattr(model, "config"):
            self.hidden_size = cast(int, model.config.hidden_size)
        else:
            self.hidden_size = head_dim * num_heads
        self.kv_manager = KVCacheManager(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=torch.float16,  # Force fp16 for FlashAttention
        )
        self.kv_manager.wire_kv_cache_to_model(model)
        self.sampler = Sampler()
        self.current_sequences = []
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_vars = None
        self.graph_pool = None
        self.graph_captured = False

    def _prepare_prefill(self, requests: Iterable[Seq]) -> tuple[Tensor, Tensor]:
        input_ids_int: list[int] = []
        positions_int: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping_int: list[int] = []

        for req in requests:
            seqlen = len(req)
            input_ids_int.extend(req[req.num_cached_tokens :])
            positions_int.extend(list(range(req.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - req.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if req.block_table:
                num_cached = req.num_cached_tokens
                num_total = len(req)

                for token_idx in range(num_cached, num_total):
                    block_idx = token_idx // self.block_size
                    block_offset = token_idx % self.block_size
                    block_id = req.block_table[block_idx]
                    slot_idx = block_id * self.block_size + block_offset
                    slot_mapping_int.append(slot_idx)

        input_ids = torch.tensor(input_ids_int, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions_int, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping_int, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            max_len = max(len(req.block_table) for req in requests)
            block_tables_list: list[list[int]] = []
            for req in requests:
                table = req.block_table + [-1] * (max_len - len(req.block_table))
                block_tables_list.append(table)
            block_tables = torch.tensor(block_tables_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        ForwardContext.set(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )

        return input_ids, positions

    def _prepare_decode(self, requests: Collection[Seq]) -> tuple[Tensor, Tensor]:
        if not requests:
            msg = "FATAL: No requests provided to _prepare_decode!"
            raise RuntimeError(msg)

        input_ids_int: list[Number] = []
        positions_int: list[int] = []
        slot_mapping_int: list[int] = []
        context_lens_int: list[int] = []

        for req in requests:
            input_ids_int.append(req.last_token)

            pos = len(req) - 1
            if hasattr(self, "_tts_mode") and self._tts_mode:
                pos -= self._tts_prompt_len - 1
            positions_int.append(pos)

            context_lens_int.append(len(req))
            slot_mapping_int.append(req.block_table[-1] * self.block_size + req.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids_int, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions_int, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping_int, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens_int, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        max_len = max(len(req.block_table) for req in requests)
        block_tables_list: list[list[int]] = []
        for req in requests:
            table = req.block_table + [-1] * (max_len - len(req.block_table))
            block_tables_list.append(table)
        block_tables = torch.tensor(block_tables_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        assert block_tables.dim() == 2, f"block_tables must be 2D, got shape {block_tables.shape}"
        assert block_tables.size(0) == len(requests), (
            f"block_tables batch size mismatch: {block_tables.size(0)} vs {len(requests)}"
        )

        ForwardContext.set(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        return input_ids, positions

    def _prepare_sample(self, requests: Collection[Seq], temperature: float) -> Tensor:
        temperatures = [temperature] * len(requests)
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    def _capture_cuda_graphs(
        self,
        tts_mel_embedding: nn.Embedding | None = None,
        tts_text_pos_embedding: LearnedPositionEmbeddings | None = None,
    ) -> None:
        print("Capturing CUDA graphs for decode optimization...")
        max_num_blocks = (2048 + self.block_size - 1) // self.block_size
        model_dtype = next(self.model.parameters()).dtype
        input_ids = torch.ones(MAX_BS, dtype=torch.int64, device="cuda")
        positions = torch.ones(MAX_BS, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(MAX_BS, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(MAX_BS, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(MAX_BS, max_num_blocks, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(MAX_BS, self.hidden_size, dtype=model_dtype, device="cuda")
        inputs_embeds_buffer = torch.zeros(MAX_BS, self.hidden_size, dtype=model_dtype, device="cuda")

        use_tts = tts_mel_embedding is not None and tts_text_pos_embedding is not None

        for bs in reversed(GRAPH_BS):
            graph = torch.cuda.CUDAGraph()

            slot_mapping[:bs] = torch.arange(bs, dtype=torch.int32, device="cuda")
            context_lens[:bs] = bs + 1
            block_tables[:bs, :] = 0

            ForwardContext.set(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            # warmup
            if use_tts:
                assert tts_mel_embedding is not None
                assert tts_text_pos_embedding is not None
                emb = tts_mel_embedding.forward(input_ids[:bs])
                pos_clamped = torch.clamp(positions[:bs], min=0)
                pos_emb = tts_text_pos_embedding.emb.forward(pos_clamped)
                inputs_embeds_buffer[:bs] = emb + pos_emb
                out = self.model.forward(
                    inputs_embeds=inputs_embeds_buffer[:bs].unsqueeze(1),
                    return_dict=True,
                ).last_hidden_state
            else:
                out = self.model.forward(input_ids=input_ids[:bs].unsqueeze(1), return_dict=True).last_hidden_state
            outputs[:bs] = out.squeeze(1) if out.dim() == 3 else out

            with torch.cuda.graph(graph, self.graph_pool):
                if use_tts:
                    assert tts_mel_embedding is not None
                    assert tts_text_pos_embedding is not None
                    emb = tts_mel_embedding.forward(input_ids[:bs])
                    pos_clamped = torch.clamp(positions[:bs], min=0)
                    pos_emb = tts_text_pos_embedding.emb.forward(pos_clamped)
                    inputs_embeds_buffer[:bs] = emb + pos_emb
                    out = self.model.forward(
                        inputs_embeds=inputs_embeds_buffer[:bs].unsqueeze(1),
                        return_dict=True,
                    ).last_hidden_state
                else:
                    out = self.model.forward(input_ids=input_ids[:bs].unsqueeze(1), return_dict=True).last_hidden_state
                outputs[:bs] = out.squeeze(1) if out.dim() == 3 else out

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            ForwardContext.reset()

        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
            "inputs_embeds": inputs_embeds_buffer,
        }
        print(f"CUDA graphs captured for batch sizes: {GRAPH_BS}")

    def _run_decode_with_graph(
        self,
        input_ids: Tensor,
        positions: Tensor,
        context: ForwardContext,
        tts_mel_embedding: nn.Embedding | None = None,
        tts_text_pos_embedding: LearnedPositionEmbeddings | None = None,
    ) -> Tensor:
        bs = input_ids.size(0)
        use_tts_embedding = hasattr(self, "_tts_mode") and self._tts_mode

        if not self.use_cuda_graph or not self.graphs:
            if use_tts_embedding:
                assert tts_mel_embedding is not None
                assert tts_text_pos_embedding is not None
                inputs_embeds = tts_mel_embedding.forward(input_ids)
                pos_clamped = torch.clamp(positions, min=0)
                pos_emb = tts_text_pos_embedding.emb.forward(pos_clamped)
                inputs_embeds += pos_emb
                out = self.model.forward(inputs_embeds=inputs_embeds.unsqueeze(1), return_dict=True).last_hidden_state
            else:
                out = self.model.forward(input_ids=input_ids.unsqueeze(1), return_dict=True).last_hidden_state
            return out.squeeze(1) if out.dim() == 3 else out

        graph_bs = next((x for x in GRAPH_BS if x >= bs), None)
        if graph_bs is None:
            if use_tts_embedding:
                assert tts_mel_embedding is not None
                assert tts_text_pos_embedding is not None
                inputs_embeds = tts_mel_embedding.forward(input_ids)
                pos_clamped = torch.clamp(positions, min=0)
                pos_emb = tts_text_pos_embedding.emb.forward(pos_clamped)
                inputs_embeds += pos_emb
                out = self.model.forward(inputs_embeds=inputs_embeds.unsqueeze(1), return_dict=True).last_hidden_state
            else:
                out = self.model.forward(input_ids=input_ids.unsqueeze(1), return_dict=True).last_hidden_state
            return out.squeeze(1) if out.dim() == 3 else out

        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars

        if graph_vars is None:
            msg = "Graph variables not initialized"
            raise RuntimeError(msg)

        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        assert context.slot_mapping is not None
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        assert context.context_lens is not None
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :].fill_(-1)
        assert context.block_tables is not None
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        graph.replay()

        return graph_vars["outputs"][:bs]

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        _top_k: int = 50,
        _top_p: float = 1.0,
        stop_tokens: Sequence[int] | None = None,
        attention_mask: Tensor | None = None,
        tts_embeddings: Tensor | None = None,  # TTS: [pad][cond][text] embeddings (87 tokens, NO start_mel)
        tts_mel_embedding: nn.Embedding | None = None,  # TTS: mel_embedding layer
        tts_text_pos_embedding: LearnedPositionEmbeddings | None = None,  # TTS: text_pos_embedding layer
    ) -> Tensor:
        """Generate tokens.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            stop_tokens: List of token IDs that stop generation

        Returns:
            Generated token IDs [batch_size, total_len]

        """
        t_start = time.perf_counter()
        batch_size = input_ids.size(0)
        device = input_ids.device

        self._tts_mode = tts_embeddings is not None
        self._tts_prompt_len = input_ids.size(1) if self._tts_mode else 0

        if self.use_cuda_graph and not self.graph_captured:
            print(
                f"[CAPTURE] use_cuda_graph={self.use_cuda_graph}, graph_captured={self.graph_captured}",
                file=sys.stderr,
                flush=True,
            )
            self._capture_cuda_graphs(
                tts_mel_embedding=tts_mel_embedding,
                tts_text_pos_embedding=tts_text_pos_embedding,
            )
            self.graph_captured = True
            print(
                f"[CAPTURE] Completed! graphs={list(self.graphs.keys())}",
                file=sys.stderr,
                flush=True,
            )

        actual_seq_len = tts_embeddings.size(1) + 1 if tts_embeddings is not None else input_ids.size(1)

        is_varlen_batch = (
            tts_embeddings is not None
            and attention_mask is not None
            and batch_size > 1
            and (attention_mask.sum(dim=1) != attention_mask.size(1)).any()
        )

        seq_lens: list[int]
        if is_varlen_batch and attention_mask is not None:
            seq_lens = [int(attention_mask[i].sum().item()) for i in range(batch_size)]
        else:
            seq_lens = [actual_seq_len] * batch_size

        sequences: list[Seq] = []
        for i in range(batch_size):
            seq_len = seq_lens[i]
            token_ids = [1] * seq_len
            if tts_embeddings is not None and seq_len > 0:
                if input_ids.size(1) > 0:
                    token_ids[-1] = int(input_ids[i, -1].item())
                else:
                    token_ids[-1] = 1
            else:
                token_ids = cast(list[int], input_ids[i].tolist())
            req = Seq(token_ids)
            self.kv_manager.allocate(req)
            sequences.append(req)

        self.current_sequences = sequences

        t_prefill_start = time.perf_counter()
        _prefill_ids, _prefill_pos = self._prepare_prefill(sequences)

        if tts_embeddings is not None and tts_mel_embedding is not None and tts_text_pos_embedding is not None:
            start_token_id = input_ids[0, -1] if input_ids.size(1) > 0 else 8192

            start_emb = tts_mel_embedding.forward(torch.tensor([[start_token_id]], device="cuda"))  # [1, 1, hidden_dim]

            start_pos = torch.tensor([[tts_embeddings.size(1)]], device="cuda", dtype=torch.long)
            pos_emb = tts_text_pos_embedding.emb.forward(start_pos)
            start_emb += pos_emb
            start_emb = start_emb.repeat(batch_size, 1, 1)

            if is_varlen_batch:
                valid_embeddings: list[torch.Tensor] = []
                for i in range(batch_size):
                    emb_len = seq_lens[i] - 1
                    padding_len = tts_embeddings.size(1) - emb_len
                    valid_emb = tts_embeddings[i, padding_len:].unsqueeze(0)  # [1, emb_len, hidden_dim]
                    valid_embeddings.append(torch.cat([valid_emb, start_emb[i : i + 1]], dim=1))
                full_embeddings = torch.cat(valid_embeddings, dim=1)  # [1, total_tokens, hidden_dim]
            else:
                full_embeddings = torch.cat([tts_embeddings, start_emb], dim=1)  # [batch_size, seq_len, hidden_dim]

            model_dtype = next(self.model.parameters()).dtype
            if full_embeddings.dtype != model_dtype:
                full_embeddings = full_embeddings.to(model_dtype)

            hidden_states = self.model.forward(inputs_embeds=full_embeddings, return_dict=True).last_hidden_state

        else:
            hidden_states = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        t_prefill_end = time.perf_counter()
        print(f"[Profile] Prefill time: {(t_prefill_end - t_prefill_start) * 1000:.2f}ms", file=sys.stderr, flush=True)

        if is_varlen_batch:
            context = ForwardContext.get()
            assert context.cu_seqlens_q is not None
            cu_seqlens = cast(list[int], context.cu_seqlens_q.cpu().tolist())
            last_hidden = torch.stack([hidden_states[0, cu_seqlens[i + 1] - 1] for i in range(batch_size)])
        else:
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        ForwardContext.reset()

        t_first_token_start = time.perf_counter()
        if last_hidden.dtype != next(self.lm_head.parameters()).dtype:
            last_hidden = last_hidden.to(next(self.lm_head.parameters()).dtype)
        logits = self.lm_head.forward(last_hidden)  # [batch_size, vocab_size]

        temperatures = self._prepare_sample(sequences, temperature)
        if temperature > 0:
            first_token = self.sampler.forward(logits, temperatures)
        else:
            first_token = torch.argmax(logits, dim=-1)

        generated_tokens: list[list[Number]] = [[] for _ in range(batch_size)]
        is_finished = [False] * batch_size

        for i, token_id in enumerate(first_token.tolist()):
            if stop_tokens and token_id in stop_tokens:
                is_finished[i] = True
            else:
                generated_tokens[i].append(token_id)
                sequences[i].append_token(token_id)
                self.kv_manager.append_to_seq(sequences[i])

        t_first_token_end = time.perf_counter()
        print(
            f"[Profile] First token time: {(t_first_token_end - t_first_token_start) * 1000:.2f}ms",
            file=sys.stderr,
            flush=True,
        )

        if all(is_finished):
            for req in sequences:
                self.kv_manager.remove_seq(req)
            self.current_sequences: list[Seq] = []

            output_ids: list[list[Number]] = []
            for i in range(batch_size):
                full_sequence = input_ids[i].tolist() + generated_tokens[i]
                output_ids.append(full_sequence)

            return torch.tensor(output_ids, dtype=torch.long, device=device)

        remaining_tokens = max_new_tokens - 1

        t_decode_start = time.perf_counter()
        for _ in range(remaining_tokens):
            decode_ids, decode_pos = self._prepare_decode(sequences)

            hidden_states = self._run_decode_with_graph(
                decode_ids,
                decode_pos,
                ForwardContext.get(),
                tts_mel_embedding=tts_mel_embedding,
                tts_text_pos_embedding=tts_text_pos_embedding,
            )

            # Get logits
            logits = self.lm_head.forward(hidden_states)  # [batch_size, vocab_size]

            ForwardContext.reset()

            temperatures = self._prepare_sample(sequences, temperature)
            next_token = self.sampler.forward(logits, temperatures) if temperature > 0 else torch.argmax(logits, dim=-1)
            next_token_list = cast(list[int], next_token.tolist())

            for i, token_id in enumerate(next_token_list):
                if is_finished[i]:
                    continue
                elif stop_tokens and token_id in stop_tokens:
                    is_finished[i] = True
                else:
                    sequences[i].append_token(token_id)
                    self.kv_manager.append_to_seq(sequences[i])
                    generated_tokens[i].append(token_id)

            if all(is_finished):
                break

        t_decode_end = time.perf_counter()
        if remaining_tokens > 0:
            print(
                f"[Profile] Decode loop time: {(t_decode_end - t_decode_start) * 1000:.2f}ms",
                file=sys.stderr,
                flush=True,
            )

        for req in sequences:
            self.kv_manager.remove_seq(req)
        self.current_sequences = []

        pad_token = stop_tokens[0] if stop_tokens else 0

        if is_varlen_batch:
            assert attention_mask is not None
            max_prompt_len = attention_mask.size(1)
            output_ids = []

            for i in range(batch_size):
                padding_len = max_prompt_len - seq_lens[i]
                initial_tokens = sequences[i].token_ids[: sequences[i].num_prompt_tokens]
                padded_prompt = [pad_token] * padding_len + list(initial_tokens)
                full_sequence = padded_prompt + generated_tokens[i]
                output_ids.append(full_sequence)
        else:
            output_ids = [
                list(sequences[i].token_ids[: sequences[i].num_prompt_tokens]) + generated_tokens[i]
                for i in range(batch_size)
            ]

        max_length = max(len(seq) for seq in output_ids)
        padded_output_ids = [seq + [pad_token] * (max_length - len(seq)) for seq in output_ids]

        output = torch.tensor(padded_output_ids, dtype=torch.long, device=device)

        assert output.size(0) == batch_size, f"Output batch size mismatch: {output.size(0)} != {batch_size}"

        t_end = time.perf_counter()
        print(f"[Profile] Total generation time: {(t_end - t_start) * 1000:.2f}ms", file=sys.stderr, flush=True)

        return output
