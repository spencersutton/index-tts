from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model, LogitsProcessorList
from transformers.generation.logits_process import TypicalLogitsWarper
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from indextts.gpt import GPT2InferenceModel, LearnedPositionEmbeddings, NullPositionEmbedding
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.gpt.utils import set_token_padding
from indextts.util import patch_call

logger = logging.getLogger(__name__)


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass(frozen=True)
class VoiceModelConfig:
    """Configuration constants for UnifiedVoice model."""

    # Token vocabulary sizes
    number_text_tokens: int = 12000
    number_mel_codes: int = 8194

    # Special tokens
    start_text_token: int = 0
    stop_text_token: int = 1
    start_mel_token: int = 8192
    stop_mel_token: int = 8193

    # Model architecture
    layers: int = 24
    heads: int = 20
    model_dim: int = 1280

    # Sequence lengths
    max_mel_tokens: int = 1815
    max_text_tokens: int = 600
    max_conditioning_inputs: int = 1

    # Conditioning
    cond_num: int = 32
    mel_length_compression: int = 1024

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.heads

    @property
    def seq_length(self) -> int:
        return self.max_mel_tokens + self.max_text_tokens + 2


# =============================================================================
# Unified Voice Model
# =============================================================================


class UnifiedVoice(nn.Module):
    """Unified voice synthesis model combining GPT-2 with conditioning encoders."""

    inference_model: GPT2InferenceModel | None
    use_accel: bool
    accel_engine: Any | None = None
    config: VoiceModelConfig
    conditioning_encoder: ConformerEncoder
    emo_conditioning_encoder: ConformerEncoder
    perceiver_encoder: PerceiverResampler
    emo_perceiver_encoder: PerceiverResampler
    text_embedding: nn.Embedding
    mel_embedding: nn.Embedding
    emo_layer: nn.Linear
    emovec_layer: nn.Linear
    speed_emb: nn.Embedding
    gpt: GPT2Model
    mel_pos_embedding: LearnedPositionEmbeddings
    text_pos_embedding: LearnedPositionEmbeddings
    final_norm: nn.LayerNorm
    text_head: nn.Linear
    mel_head: nn.Linear

    def __init__(self, use_accel: bool = False, config: VoiceModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or VoiceModelConfig()

        # -----------------------------------------------------------------
        # Conditioning encoders
        # -----------------------------------------------------------------
        self.cond_mask_pad = nn.ConstantPad1d((self.config.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        # Speaker conditioning encoder
        self.conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=512,
            linear_units=2048,
            attention_heads=8,
            num_blocks=6,
        )
        self.perceiver_encoder = PerceiverResampler(
            self.config.model_dim,
            dim_context=512,
            ff_mult=2,
            heads=8,
            num_latents=self.config.cond_num,
        )

        # Emotion conditioning encoder (smaller architecture)
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=512,
            linear_units=1024,
            attention_heads=4,
            num_blocks=4,
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=512,
            ff_mult=2,
            heads=4,
            num_latents=1,
        )

        # -----------------------------------------------------------------
        # Embeddings
        # -----------------------------------------------------------------
        self.text_embedding = nn.Embedding(self.config.number_text_tokens + 1, self.config.model_dim)
        self.mel_embedding = nn.Embedding(self.config.number_mel_codes, self.config.model_dim)

        # Emotion projection layers
        self.emo_layer = nn.Linear(self.config.model_dim, self.config.model_dim)
        self.emovec_layer = nn.Linear(1024, self.config.model_dim)

        # Speed/duration embedding (initialized to zero)
        self.speed_emb = nn.Embedding(2, self.config.model_dim)
        self.speed_emb.weight.data.zero_()

        # GPT-2 style initialization
        for emb in [self.text_embedding, self.mel_embedding]:
            emb.weight.data.normal_(mean=0.0, std=0.02)

        # -----------------------------------------------------------------
        # GPT-2 transformer + positional embeddings
        # -----------------------------------------------------------------
        max_mel_seq_len = self.config.max_mel_tokens + 2 + self.config.max_conditioning_inputs
        max_text_seq_len = self.config.max_text_tokens + 2

        gpt = GPT2Model(
            GPT2Config(
                vocab_size=256,  # Unused.
                n_positions=max_mel_seq_len + max_text_seq_len,
                n_ctx=max_mel_seq_len + max_text_seq_len,
                n_embd=self.config.model_dim,
                n_layer=self.config.layers,
                n_head=self.config.heads,
                use_cache=False,
            )
        )
        # `GPT2Model` initialization may sanitize config fields; set this after model
        # construction so the attribute is reliably present for downstream checks/tests.
        gpt.config.gradient_checkpointing = True
        if hasattr(gpt, "gradient_checkpointing_enable"):
            gpt.gradient_checkpointing_enable()

        # Override the built in positional embeddings
        del gpt.wpe
        gpt.wpe = NullPositionEmbedding(self.config.model_dim)

        # Built-in token embeddings are unused.
        del gpt.wte

        self.gpt = gpt
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_seq_len, self.config.model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_seq_len, self.config.model_dim)

        # -----------------------------------------------------------------
        # Output heads
        # -----------------------------------------------------------------
        self.final_norm = nn.LayerNorm(self.config.model_dim)
        self.text_head = nn.Linear(self.config.model_dim, self.config.number_text_tokens + 1)
        self.mel_head = nn.Linear(self.config.model_dim, self.config.number_mel_codes)

        # Runtime state
        self.use_accel = use_accel
        self.accel_engine = None
        self.inference_model = None

    # -------------------------------------------------------------------------
    # Post-initialization for Inference
    # -------------------------------------------------------------------------

    def post_init_gpt2_config(self, half: bool = False) -> None:
        """Initialize inference components after model loading."""
        gpt_config = GPT2Config(
            vocab_size=self.config.number_mel_codes,
            n_positions=self.config.seq_length,
            n_ctx=self.config.seq_length,
            n_embd=self.config.model_dim,
            n_layer=self.config.layers,
            n_head=self.config.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )

        if self.use_accel and torch.cuda.is_available():
            if importlib.util.find_spec("flash_attn") is None:
                raise ImportError(
                    "flash_attn is required for acceleration but not installed. "
                    "Please install from https://github.com/Dao-AILab/flash-attention/releases/"
                )

            from indextts.accel import AccelInferenceEngine, GPT2AccelModel  # noqa: PLC0415

            accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)
            accel_gpt = (accel_gpt.half() if half else accel_gpt).cuda().eval()

            lm_head_with_norm = nn.Sequential(self.final_norm, self.mel_head)
            self.accel_engine = AccelInferenceEngine(
                model=accel_gpt,
                lm_head=lm_head_with_norm,
                num_layers=self.config.layers,
                num_heads=self.config.heads,
                head_dim=self.config.head_dim,
                block_size=256,
                num_blocks=16,  # 16 * 256 = 4096 tokens capacity
                use_cuda_graph=True,
            )
            logger.info("acceleration engine initialized")

        inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=True,
        )
        self.inference_model = inference_model.eval()

        self.gpt.wte = self.mel_embedding

    # -------------------------------------------------------------------------
    # Conditioning Extraction
    # -------------------------------------------------------------------------

    def get_conditioning(self, speech_conditioning_input: Tensor, cond_mel_lengths: Tensor) -> Tensor:
        """Extract speaker conditioning latents from speech input.

        Args:
            speech_conditioning_input: (batch, frames, dim) speech features
            cond_mel_lengths: (batch,) lengths of each conditioning sequence

        Returns:
            (batch, cond_num, model_dim) conditioning latents
        """
        encoded, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2), cond_mel_lengths)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        return self.perceiver_encoder(encoded, conds_mask)

    def get_emo_conditioning(self, speech_conditioning_input: Tensor, cond_mel_lengths: Tensor) -> Tensor:
        """Extract emotion conditioning from speech input.

        Args:
            speech_conditioning_input: (batch, frames, dim) speech features
            cond_mel_lengths: (batch,) lengths of each conditioning sequence

        Returns:
            (batch, dim) emotion conditioning vector
        """
        encoded, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2), cond_mel_lengths)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(encoded, conds_mask)
        return conds.squeeze(1)

    def get_emovec(self, emo_speech_conditioning_latent: Tensor, emo_cond_lengths: Tensor) -> Tensor:
        """Extract and project emotion vector from speech conditioning."""
        emo_vec = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1, 2), emo_cond_lengths)
        return self.emo_layer(self.emovec_layer(emo_vec))

    def merge_emovec(
        self,
        speech_conditioning_latent: Tensor,
        emo_speech_conditioning_latent: Tensor,
        cond_lengths: Tensor,
        emo_cond_lengths: Tensor,
        alpha: float = 1.0,
    ) -> Tensor:
        """Blend base and emotion vectors using linear interpolation."""
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)
        return base_vec + alpha * (emo_vec - base_vec)

    def _build_conditioning_concat(
        self,
        speech_conditioning_latent: Tensor,
        emo_vec: Tensor,
        use_speed: Tensor,
    ) -> Tensor:
        """Build concatenated conditioning tensor with emotion and duration embeddings."""
        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        return torch.cat(
            (
                speech_conditioning_latent + emo_vec.unsqueeze(1),
                duration_emb_half.unsqueeze(1),
                duration_emb.unsqueeze(1),
            ),
            dim=1,
        )

    @override
    def forward(
        self,
        speech_conditioning_latent: Tensor,
        text_inputs: Tensor,
        text_lengths: Tensor,
        mel_codes: Tensor,
        mel_codes_lengths: Tensor,
        emo_speech_conditioning_latent: Tensor,
        cond_mel_lengths: Tensor,
        emo_cond_mel_lengths: Tensor,
        emo_vec: Tensor | None = None,
        use_speed: Tensor | None = None,
        do_spk_cond: bool = False,
    ) -> Tensor:
        """Forward pass combining text and voice conditioning.

        Args:
            speech_conditioning_latent: Speaker conditioning (batch, dim, frames) or (batch, cond_num, dim)
            text_inputs: Text token IDs (batch, text_len)
            text_lengths: Actual text lengths (batch,)
            mel_codes: Mel token codes (batch, mel_len)
            mel_codes_lengths: Actual mel lengths (batch,)
            emo_speech_conditioning_latent: Emotion conditioning (batch, dim, frames)
            cond_mel_lengths: Conditioning mel lengths (batch,)
            emo_cond_mel_lengths: Emotion conditioning lengths (batch,)
            emo_vec: Pre-computed emotion vector or None
            use_speed: Speed control tensor (batch,)
            do_spk_cond: If True, compute speaker conditioning from raw input

        Returns:
            Mel latent representations (batch, mel_len, dim)
        """
        # Compute speaker conditioning if needed
        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(
                speech_conditioning_latent.transpose(1, 2), cond_mel_lengths
            )

        # Compute emotion vector if not provided
        if emo_vec is None:
            emo_vec = self.get_emo_conditioning(
                emo_speech_conditioning_latent.transpose(1, 2),
                emo_cond_mel_lengths,
            )
            emo_vec = self.emo_layer(self.emovec_layer(emo_vec))

        # Prepare text and mel tokens
        text_inputs = set_token_padding(text_inputs, text_lengths, self.config.stop_text_token)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.config.stop_text_token)

        mel_codes = set_token_padding(mel_codes, mel_codes_lengths, self.config.stop_mel_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.config.stop_mel_token)

        # Build conditioning
        assert use_speed is not None
        conds = self._build_conditioning_concat(speech_conditioning_latent, emo_vec, use_speed)

        # Build aligned inputs
        text_inputs = F.pad(text_inputs, (1, 0), value=self.config.start_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        mel_codes = F.pad(mel_codes, (1, 0), value=self.config.start_mel_token)

        mel_emb = self.mel_embedding(mel_codes) + self.mel_pos_embedding(mel_codes)

        # Get latent representations
        parts = [conds, text_emb, mel_emb]
        emb = torch.cat(parts, dim=1)

        # GPT forward pass
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True)
        assert isinstance(gpt_out, BaseModelOutputWithPastAndCrossAttentions)

        # Extract encoded representations (skip conditioning)
        offset = conds.shape[1]
        assert gpt_out.last_hidden_state is not None
        enc = self.final_norm(gpt_out.last_hidden_state[:, offset:])

        mel_latent = enc[:, -mel_emb.shape[1] :]

        # Strip the two tokens added by padding
        return mel_latent[:, :-2]

    def prepare_gpt_inputs(
        self,
        conditional_latents: Tensor,
        text_inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare inputs for GPT2InferenceModel.generate().

        Args:
            conditional_latents: (batch, cond_num, dim) conditioning embeddings
            text_inputs: (batch, text_len) text token IDs

        Returns:
            input_ids: (batch, seq_len+1) fake input IDs for generation
            inputs_embeds: (batch, seq_len, dim) input embeddings
            attention_mask: (batch, seq_len+1) attention mask
        """
        batch_size, text_len = text_inputs.shape[:2]
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1

        if not single_cond:
            assert conditional_latents.shape[0] == batch_size, (
                f"batch size mismatch: {conditional_latents.shape[0]} vs {batch_size}"
            )

        target_len = conditional_latents.shape[1] + text_len + 2
        batched_mel_embs: list[Tensor] = []
        attention_masks: list[Tensor] = []

        for i in range(batch_size):
            cond_latent = conditional_latents.squeeze(0) if single_cond else conditional_latents[i]
            text_input = text_inputs[i]
            # Filter out special tokens and add start/stop
            valid_mask = (text_input != self.config.stop_text_token) & (text_input != self.config.start_text_token)
            text_input = text_input[valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.config.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.config.stop_text_token)

            # Compute text embeddings
            text_pos = torch.arange(text_input.size(-1), device=cond_latent.device, dtype=torch.long)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_pos)

            # Build sequence: [optional_pad][cond][text]
            parts: list[Tensor] = [cond_latent, text_emb]
            attn_mask = torch.ones(target_len + 1, dtype=torch.long, device=text_emb.device)

            # Add left padding if needed
            padding = text_len + 2 - text_input.size(-1)
            if padding > 0:
                pad = torch.zeros(
                    (padding, cond_latent.size(-1)),
                    dtype=text_emb.dtype,
                    device=text_emb.device,
                )
                parts.insert(0, pad)
                attn_mask[:padding] = 0

            mel_emb = torch.cat(parts)
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"

            batched_mel_embs.append(mel_emb)
            attention_masks.append(attn_mask)

        # Stack batched outputs
        batched_mel_emb = torch.stack(batched_mel_embs, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)

        # Create fake input IDs with start_mel_token at the end
        fake_inputs = torch.ones((batch_size, target_len + 1), dtype=torch.long, device=batched_mel_emb.device)
        fake_inputs[:, -1] = self.config.start_mel_token

        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(
        self,
        speech_condition: Tensor,
        text_inputs: Tensor,
        emo_speech_condition: Tensor | None = None,
        cond_lengths: Tensor | None = None,
        emo_cond_lengths: Tensor | None = None,
        emo_vec: Tensor | None = None,
        input_tokens: Tensor | None = None,
        num_return_sequences: int = 1,
        max_generate_length: int | None = None,
        typical_sampling: bool = False,
        typical_mass: float = 0.9,
        **hf_generate_kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Generate speech tokens from text and conditioning.

        Args:
            speech_condition: (batch, dim, frames) or (dim, frames) speaker conditioning
            text_inputs: (batch, text_len) text token IDs
            emo_speech_condition: Optional emotion conditioning
            cond_lengths: Lengths of conditioning sequences
            emo_cond_lengths: Lengths of emotion conditioning sequences
            emo_vec: Pre-computed emotion vector
            input_tokens: Additional tokens for generation
            num_return_sequences: Number of sequences to generate
            max_generate_length: Maximum generation length
            typical_sampling: Use typical sampling
            typical_mass: Mass for typical sampling
            **hf_generate_kwargs: Additional HuggingFace generate kwargs

        Returns:
            generated_tokens: (batch, generated_len) generated mel tokens
            speech_conditioning_latent: (batch, cond_num, dim) conditioning used
        """
        t0 = time.perf_counter()

        # Normalize input dimensions
        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=emo_speech_condition.device)

        # Compute conditioning latents
        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1, 2), cond_lengths)
        logger.info(f"get_conditioning: {time.perf_counter() - t0:.4f}s")

        # Compute or use provided emotion vector
        if emo_vec is None:
            t1 = time.perf_counter()
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1, 2), emo_cond_lengths)
            emo_vec = self.emo_layer(self.emovec_layer(emo_vec))
            logger.info(f"get_emo_conditioning: {time.perf_counter() - t1:.4f}s")
        else:
            logger.info("Using provided emotion vector")

        # Build conditioning latent
        conds_latent = self._build_conditioning_concat(
            speech_conditioning_latent,
            emo_vec,
            torch.zeros(text_inputs.size(0), dtype=torch.long, device=text_inputs.device),
        )

        # Prepare GPT inputs
        t2 = time.perf_counter()
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        assert self.inference_model is not None
        self.inference_model.store_mel_emb(inputs_embeds)
        logger.info(f"prepare_gpt_inputs: {time.perf_counter() - t2:.4f}s")

        # Handle additional input tokens
        if input_tokens is None:
            inputs = input_ids
        else:
            batch_size = text_inputs.shape[0]
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)

            assert num_return_sequences % input_tokens.shape[0] == 0, (
                "num_return_sequences must be divisible by input_tokens batch size"
            )
            assert num_return_sequences % batch_size == 0, (
                "num_return_sequences must be divisible by text_inputs batch size"
            )

            repeat_factor = num_return_sequences // input_ids.shape[0]
            if repeat_factor > 1:
                input_ids = input_ids.repeat(repeat_factor, 1)
                attention_mask = attention_mask.repeat(repeat_factor, 1)

            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)

        # Setup generation parameters
        trunc_index = inputs.shape[1]
        logits_processor = LogitsProcessorList()

        if typical_sampling:
            if not (0.0 < typical_mass < 1.0):
                msg = f"`typical_mass` must be > 0 and < 1, got {typical_mass}"
                raise ValueError(msg)
            min_tokens = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens))

        max_length = (
            trunc_index + self.config.max_mel_tokens - 1
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        # Generate
        t3 = time.perf_counter()

        if self.accel_engine is not None and num_return_sequences == 1:
            output = self.accel_engine.generate(
                inputs,
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=hf_generate_kwargs.get("temperature", 1),
                stop_tokens=[self.config.stop_mel_token],
                tts_embeddings=inputs_embeds,
                tts_mel_embedding=self.inference_model.embeddings,
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,
            )
        else:
            output = self.inference_model.generate(
                inputs,
                bos_token_id=self.config.start_mel_token,
                pad_token_id=self.config.stop_mel_token,
                eos_token_id=self.config.stop_mel_token,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=logits_processor,
                num_return_sequences=num_return_sequences,
                **hf_generate_kwargs,
            )

        assert isinstance(output, Tensor)

        logger.info(f"generation: {time.perf_counter() - t3:.4f}s")
        logger.info(f"total inference_speech: {time.perf_counter() - t0:.4f}s")

        return output[:, trunc_index:], speech_conditioning_latent

    @patch_call(forward)
    def __call__(self) -> None: ...
