from __future__ import annotations

import importlib.util
import logging
import time
import typing
from typing import Any, Final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model, LogitsProcessorList
from transformers.generation.logits_process import TypicalLogitsWarper
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.inference_model import GPT2InferenceModel, NullPositionEmbedding
from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings
from indextts.gpt.perceiver import PerceiverResampler
from indextts.util import patch_call

if typing.TYPE_CHECKING:
    from indextts.accel.accel_engine import AccelInferenceEngine

logger = logging.getLogger(__name__)


# =============================================================================
# Model helpers
# =============================================================================


def get_embedding(
    tokens: Tensor,
    start_token: int,
    stop_token: int,
    embeddings: nn.Embedding,
    positional_embeddings: LearnedPositionEmbeddings,
) -> Tensor:
    mask = (tokens == stop_token).cummax(dim=-1).values
    tokens = tokens.masked_fill_(mask, stop_token)
    tokens = F.pad(tokens, (0, 1), value=stop_token)
    tokens = F.pad(tokens, (1, 0), value=start_token)
    text_emb = embeddings(tokens) + positional_embeddings(tokens)
    return text_emb


# Token vocabulary sizes
NUMBER_TEXT_TOKENS: Final = 12000
NUMBER_MEL_CODES: Final = 8194
# Special tokens
STOP_TEXT_TOKEN: Final = 1
START_TEXT_TOKEN: Final = 0
STOP_MEL_TOKEN: Final = NUMBER_MEL_CODES - 1
START_MEL_TOKEN: Final = STOP_MEL_TOKEN - 1
# Model architecture
LAYERS: Final = 24
HEADS: Final = 20
MODEL_DIM: Final = 1280
# Sequence lengths
MAX_MEL_TOKENS: Final = 1815
MAX_TEXT_TOKENS: Final = 600
MAX_CONDITIONING_INPUTS: Final = 1
# Conditioning
COND_NUM: Final = 32
MEL_LENGTH_COMPRESSION: Final = 1024
HEAD_DIM: Final = MODEL_DIM // HEADS
SEQ_LENGTH: Final = MAX_MEL_TOKENS + MAX_TEXT_TOKENS + 2


class UnifiedVoice(nn.Module):
    """Unified voice synthesis model combining GPT-2 with conditioning encoders."""

    inference_model: GPT2InferenceModel | None
    use_accel: bool
    accel_engine: AccelInferenceEngine | None = None

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

    def __init__(self, *, use_accel: bool = False) -> None:
        super().__init__()

        self.use_accel = use_accel

        # -----------------------------------------------------------------
        # Conditioning encoders
        # -----------------------------------------------------------------
        self.cond_mask_pad = nn.ConstantPad1d((COND_NUM, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        # Speaker conditioning encoder
        self.conditioning_encoder = ConformerEncoder(linear_units=2048, attention_heads=8, num_blocks=6)
        self.perceiver_encoder = PerceiverResampler(MODEL_DIM, heads=8, num_latents=COND_NUM)

        # Emotion conditioning encoder (smaller architecture)
        self.emo_conditioning_encoder = ConformerEncoder(linear_units=1024, attention_heads=4, num_blocks=4)
        self.emo_perceiver_encoder = PerceiverResampler(1024, heads=4, num_latents=1)

        # -----------------------------------------------------------------
        # Embeddings
        # -----------------------------------------------------------------
        self.text_embedding = nn.Embedding(NUMBER_TEXT_TOKENS + 1, MODEL_DIM)
        self.mel_embedding = nn.Embedding(NUMBER_MEL_CODES, MODEL_DIM)

        # Emotion projection layers
        self.emo_layer = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.emovec_layer = nn.Linear(1024, MODEL_DIM)

        # Speed/duration embedding (initialized to zero)
        self.speed_emb = nn.Embedding(2, MODEL_DIM)
        self.speed_emb.weight.data.zero_()

        # GPT-2 style initialization
        for emb in [self.text_embedding, self.mel_embedding]:
            emb.weight.data.normal_(mean=0.0, std=0.02)

        # -----------------------------------------------------------------
        # GPT-2 transformer + positional embeddings
        # -----------------------------------------------------------------
        max_mel_sequence_len = MAX_MEL_TOKENS + 2 + MAX_CONDITIONING_INPUTS
        max_text_sequence_len = MAX_TEXT_TOKENS + 2

        gpt = GPT2Model(
            GPT2Config(
                vocab_size=256,  # Unused.
                n_positions=max_mel_sequence_len + max_text_sequence_len,
                n_ctx=max_mel_sequence_len + max_text_sequence_len,
                n_embd=MODEL_DIM,
                n_layer=LAYERS,
                n_head=HEADS,
                use_cache=False,
            )
        )

        # Override the built in positional embeddings
        del gpt.wpe
        gpt.wpe = NullPositionEmbedding(MODEL_DIM)

        # Built-in token embeddings are unused.
        del gpt.wte

        self.gpt = gpt
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_sequence_len, MODEL_DIM)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_sequence_len, MODEL_DIM)

        # -----------------------------------------------------------------
        # Output heads
        # -----------------------------------------------------------------
        self.final_norm = nn.LayerNorm(MODEL_DIM)
        self.text_head = nn.Linear(MODEL_DIM, NUMBER_TEXT_TOKENS + 1)
        self.mel_head = nn.Linear(MODEL_DIM, NUMBER_MEL_CODES)

        # Runtime state
        self.accel_engine = None
        self.inference_model = None

    def post_init_gpt2_config(self, half: bool = False) -> None:
        """Initialize inference components after model loading."""
        gpt_config = GPT2Config(
            vocab_size=NUMBER_MEL_CODES,
            n_positions=SEQ_LENGTH,
            n_ctx=SEQ_LENGTH,
            n_embd=MODEL_DIM,
            n_layer=LAYERS,
            n_head=HEADS,
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
                num_layers=LAYERS,
                num_heads=HEADS,
                head_dim=HEAD_DIM,
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

    def _build_conditioning_concat(
        self,
        speech_conditioning_latent: Tensor,
        emotion_vector: Tensor,
        use_speed: Tensor,
    ) -> Tensor:
        """Build concatenated conditioning tensor with emotion and duration embeddings."""
        return torch.cat(
            (
                speech_conditioning_latent + emotion_vector.unsqueeze(1),
                self.speed_emb(torch.ones_like(use_speed)).unsqueeze(1),
                self.speed_emb(torch.zeros_like(use_speed)).unsqueeze(1),
            ),
            dim=1,
        )

    @override
    def forward(
        self,
        speech_conditioning_latent: Tensor,
        text_inputs: Tensor,
        mel_codes: Tensor,
        emotion_vector: Tensor,
        use_speed: Tensor,
    ) -> Tensor:
        """Forward pass combining text and voice conditioning.

        Args:
            speech_conditioning_latent: Speaker conditioning (batch, dim, frames) or (batch, cond_num, dim)
            text_inputs: Text token IDs (batch, text_len)
            mel_codes: Mel token codes (batch, mel_len)
            emo_vec: Pre-computed emotion vector or None
            use_speed: Speed control tensor (batch,)

        Returns:
            Mel latent representations (batch, mel_len, dim)
        """
        # Prepare text and mel tokens
        text_emb = get_embedding(
            text_inputs,
            START_TEXT_TOKEN,
            STOP_TEXT_TOKEN,
            self.text_embedding,
            self.text_pos_embedding,
        )
        mel_emb = get_embedding(
            mel_codes,
            START_MEL_TOKEN,
            STOP_MEL_TOKEN,
            self.mel_embedding,
            self.mel_pos_embedding,
        )

        # Get latent representations
        conds = self._build_conditioning_concat(speech_conditioning_latent, emotion_vector, use_speed)
        emb = torch.cat([conds, text_emb, mel_emb], dim=1)

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
            valid_mask = (text_input != STOP_TEXT_TOKEN) & (text_input != START_TEXT_TOKEN)
            text_input = text_input[valid_mask]
            text_input = F.pad(text_input, (1, 0), value=START_TEXT_TOKEN)
            text_input = F.pad(text_input, (0, 1), value=STOP_TEXT_TOKEN)

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
        fake_inputs[:, -1] = START_MEL_TOKEN

        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(
        self,
        speech_condition: Tensor,
        text_inputs: Tensor,
        emo_vec: Tensor,
        input_tokens: Tensor | None = None,
        max_generate_length: int | None = None,
        typical_sampling: bool = False,
        typical_mass: float = 0.9,
        num_beams: int = 1,
        temperature: float = 1.0,
        **hf_generate_kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Generate speech tokens from text and conditioning.

        Args:
            speech_condition: (batch, dim, frames) or (dim, frames) speaker conditioning
            text_inputs: (batch, text_len) text token IDs
            emo_speech_condition: Optional emotion conditioning
            emo_vec: Pre-computed emotion vector
            input_tokens: Additional tokens for generation
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

        # Compute conditioning latents
        encoded, mask = self.conditioning_encoder(speech_condition)
        speech_conditioning_latent = self.perceiver_encoder(encoded, self.cond_mask_pad(mask.squeeze(1)))
        logger.info(f"get_conditioning: {time.perf_counter() - t0:.4f}s")

        # Prepare GPT inputs
        t2 = time.perf_counter()
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(
            self._build_conditioning_concat(
                speech_conditioning_latent,
                emo_vec,
                torch.zeros(text_inputs.size(0), dtype=torch.long, device=text_inputs.device),
            ),
            text_inputs,
        )
        assert self.inference_model is not None
        self.inference_model.store_mel_emb(inputs_embeds)
        logger.info(f"prepare_gpt_inputs: {time.perf_counter() - t2:.4f}s")

        # Handle additional input tokens
        if input_tokens is None:
            inputs = input_ids
        else:
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)

            input_tokens = input_tokens.repeat(1, 1)
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)

        # Setup generation parameters
        logits_processor = LogitsProcessorList()

        if typical_sampling:
            if not (0.0 < typical_mass < 1.0):
                raise ValueError(f"`typical_mass` must be > 0 and < 1, got {typical_mass}")
            min_tokens = 2 if num_beams > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens))

        trunc_index = inputs.shape[1]
        max_length = (
            trunc_index + MAX_MEL_TOKENS - 1 if max_generate_length is None else trunc_index + max_generate_length
        )

        # Generate
        t3 = time.perf_counter()

        if self.accel_engine is not None and 1 == 1:
            output = self.accel_engine.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_length - trunc_index,
                stop_tokens=[STOP_MEL_TOKEN],
                temperature=temperature,
                tts_embeddings=inputs_embeds,
                tts_mel_embedding=self.inference_model.embeddings,
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,
            )
        else:
            output = self.inference_model.generate(
                inputs,
                attention_mask=attention_mask,
                bos_token_id=START_MEL_TOKEN,
                eos_token_id=STOP_MEL_TOKEN,
                logits_processor=logits_processor,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=STOP_MEL_TOKEN,
                **hf_generate_kwargs,
            )

        logger.info(f"generation: {time.perf_counter() - t3:.4f}s")
        logger.info(f"total inference_speech: {time.perf_counter() - t0:.4f}s")

        return output[:, trunc_index:], speech_conditioning_latent

    @patch_call(forward)
    def __call__(self) -> None: ...
