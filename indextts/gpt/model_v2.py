from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, LogitsProcessorList
from transformers.generation.logits_process import TypicalLogitsWarper
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from indextts.gpt import GPT2InferenceModel
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.gpt.utils import (
    build_hf_gpt_transformer,
    set_token_padding,
)
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

    # Type annotations for dynamically initialized attributes
    gst_encoder: nn.Module | None
    inference_model: GPT2InferenceModel | None
    ds_engine: object

    def __init__(self, use_accel: bool = False, config: VoiceModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or VoiceModelConfig()
        self._init_conditioning_encoders()
        self._init_embeddings()
        self._init_gpt_transformer()
        self._init_output_heads()

        # Runtime state
        self.use_accel = use_accel
        self.accel_engine = None
        self.inference_model = None
        self.gst_encoder = None
        self.ds_engine = None

    # -------------------------------------------------------------------------
    # Initialization Helpers
    # -------------------------------------------------------------------------

    def _init_conditioning_encoders(self) -> None:
        """Initialize speaker and emotion conditioning encoders."""
        self.cond_mask_pad = nn.ConstantPad1d((self.cfg.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        # Speaker conditioning encoder
        self.conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=512,
            linear_units=2048,
            attention_heads=8,
            num_blocks=6,
            input_layer="conv2d2",
        )
        self.perceiver_encoder = PerceiverResampler(
            self.cfg.model_dim,
            dim_context=512,
            ff_mult=2,
            heads=8,
            num_latents=self.cfg.cond_num,
        )

        # Emotion conditioning encoder (smaller architecture)
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=512,
            linear_units=1024,
            attention_heads=4,
            num_blocks=4,
            input_layer="conv2d2",
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=512,
            ff_mult=2,
            heads=4,
            num_latents=1,
        )

    def _init_embeddings(self) -> None:
        """Initialize text and mel embeddings with GPT-2 initialization."""
        self.text_embedding = nn.Embedding(self.cfg.number_text_tokens + 1, self.cfg.model_dim)
        self.mel_embedding = nn.Embedding(self.cfg.number_mel_codes, self.cfg.model_dim)

        # Emotion projection layers
        self.emo_layer = nn.Linear(self.cfg.model_dim, self.cfg.model_dim)
        self.emovec_layer = nn.Linear(1024, self.cfg.model_dim)

        # Speed/duration embedding (initialized to zero)
        self.speed_emb = nn.Embedding(2, self.cfg.model_dim)
        self.speed_emb.weight.data.zero_()

        # GPT-2 style initialization
        for emb in [self.text_embedding, self.mel_embedding]:
            emb.weight.data.normal_(mean=0.0, std=0.02)

    def _init_gpt_transformer(self) -> None:
        """Initialize GPT-2 transformer and positional embeddings."""
        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            self.cfg.layers,
            self.cfg.model_dim,
            self.cfg.heads,
            self.cfg.max_mel_tokens + 2 + self.cfg.max_conditioning_inputs,
            self.cfg.max_text_tokens + 2,
        )

    def _init_output_heads(self) -> None:
        """Initialize output projection heads and normalization."""
        self.final_norm = nn.LayerNorm(self.cfg.model_dim)
        self.text_head = nn.Linear(self.cfg.model_dim, self.cfg.number_text_tokens + 1)
        self.mel_head = nn.Linear(self.cfg.model_dim, self.cfg.number_mel_codes)

    # -------------------------------------------------------------------------
    # Post-initialization for Inference
    # -------------------------------------------------------------------------

    def post_init_gpt2_config(
        self,
        use_deepspeed: bool = False,
        kv_cache: bool = False,
        half: bool = False,
    ) -> None:
        """Initialize inference components after model loading."""
        gpt_config = self._create_gpt_config()

        if self.use_accel and torch.cuda.is_available():
            self._init_accel_engine(gpt_config, half)

        self._init_inference_model(gpt_config, kv_cache, use_deepspeed, half)
        self.gpt.wte = self.mel_embedding

    def _create_gpt_config(self) -> GPT2Config:
        """Create GPT2Config for inference model."""
        return GPT2Config(
            vocab_size=self.cfg.number_mel_codes,
            n_positions=self.cfg.seq_length,
            n_ctx=self.cfg.seq_length,
            n_embd=self.cfg.model_dim,
            n_layer=self.cfg.layers,
            n_head=self.cfg.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )

    def _init_accel_engine(self, gpt_config: GPT2Config, half: bool) -> None:
        """Initialize flash attention acceleration engine."""
        if importlib.util.find_spec("flash_attn") is None:
            msg = (
                "flash_attn is required for acceleration but not installed. "
                "Please install from https://github.com/Dao-AILab/flash-attention/releases/"
            )
            raise ImportError(msg)

        from indextts.accel import AccelInferenceEngine, GPT2AccelModel

        accel_gpt = GPT2AccelModel(gpt_config)
        accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)
        accel_gpt = (accel_gpt.half() if half else accel_gpt).cuda().eval()

        lm_head_with_norm = nn.Sequential(self.final_norm, self.mel_head)
        self.accel_engine = AccelInferenceEngine(
            model=accel_gpt,
            lm_head=lm_head_with_norm,
            num_layers=self.cfg.layers,
            num_heads=self.cfg.heads,
            head_dim=self.cfg.head_dim,
            block_size=256,
            num_blocks=16,  # 16 * 256 = 4096 tokens capacity
            use_cuda_graph=True,
        )
        logger.info("acceleration engine initialized")

    def _init_inference_model(
        self,
        gpt_config: GPT2Config,
        kv_cache: bool,
        use_deepspeed: bool,
        half: bool,
    ) -> None:
        """Initialize the GPT2 inference model with optional DeepSpeed."""
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )

        if use_deepspeed and torch.cuda.is_available():
            self._apply_deepspeed(half)
        else:
            self.inference_model = self.inference_model.eval()

    def _apply_deepspeed(self, half: bool) -> None:
        """Apply DeepSpeed inference optimization."""
        import deepspeed  # ty:ignore[unresolved-import]  # noqa: PLC0415

        self.ds_engine = deepspeed.init_inference(
            model=self.inference_model,
            mp_size=1,
            replace_with_kernel_inject=True,
            dtype=torch.float16 if half else torch.float32,
        )
        self.inference_model = self.ds_engine.module.eval()

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

    # -------------------------------------------------------------------------
    # GPT Logits Computation
    # -------------------------------------------------------------------------

    def get_logits(
        self,
        speech_conditioning_inputs: Tensor,
        first_inputs: Tensor,
        first_head: nn.Linear,
        second_inputs: Tensor | None = None,
        second_head: nn.Linear | None = None,
        get_attns: bool = False,
        return_latent: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Compute logits or latents from GPT forward pass.

        Args:
            speech_conditioning_inputs: Conditioning embeddings (batch, cond_len, dim)
            first_inputs: First sequence embeddings (batch, seq1_len, dim)
            first_head: Linear head for first sequence output
            second_inputs: Optional second sequence embeddings
            second_head: Linear head for second sequence output
            get_attns: If True, return attention weights only
            return_latent: If True, return latent encodings instead of logits

        Returns:
            Logits, latents, or attention weights depending on flags
        """
        # Concatenate all inputs
        parts = [speech_conditioning_inputs, first_inputs]
        if second_inputs is not None:
            parts.append(second_inputs)
        emb = torch.cat(parts, dim=1)

        # GPT forward pass
        gpt_out = self.gpt.forward(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        assert isinstance(gpt_out, BaseModelOutputWithPastAndCrossAttentions)

        if get_attns:
            assert gpt_out.attentions is not None
            return gpt_out.attentions

        # Extract encoded representations (skip conditioning)
        offset = speech_conditioning_inputs.shape[1]
        assert gpt_out.last_hidden_state is not None
        enc = self.final_norm(gpt_out.last_hidden_state[:, offset:])

        if return_latent:
            assert second_inputs is not None
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        # Compute logits
        first_logits = first_head(enc[:, : first_inputs.shape[1]]).permute(0, 2, 1)

        if second_inputs is not None:
            assert second_head is not None
            second_logits = second_head(enc[:, -second_inputs.shape[1] :]).permute(0, 2, 1)
            return first_logits, second_logits
        return first_logits

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------

    def _compute_emo_vec(
        self,
        emo_speech_conditioning_latent: Tensor,
        emo_cond_mel_lengths: Tensor,
    ) -> Tensor:
        """Compute emotion vector from conditioning latent."""
        emo_vec = self.get_emo_conditioning(
            emo_speech_conditioning_latent.transpose(1, 2),
            emo_cond_mel_lengths,
        )
        return self.emo_layer(self.emovec_layer(emo_vec))

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
            emo_vec = self._compute_emo_vec(emo_speech_conditioning_latent, emo_cond_mel_lengths)

        # Prepare text and mel tokens
        text_inputs = set_token_padding(text_inputs, text_lengths, self.cfg.stop_text_token)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.cfg.stop_text_token)

        mel_codes = set_token_padding(mel_codes, mel_codes_lengths, self.cfg.stop_mel_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.cfg.stop_mel_token)

        # Build conditioning
        assert use_speed is not None
        conds = self._build_conditioning_concat(speech_conditioning_latent, emo_vec, use_speed)

        # Build aligned inputs
        text_inputs = F.pad(text_inputs, (1, 0), value=self.cfg.start_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        mel_codes = F.pad(mel_codes, (1, 0), value=self.cfg.start_mel_token)

        mel_emb = self.mel_embedding(mel_codes) + self.mel_pos_embedding(mel_codes)

        # Get latent representations
        _, mel_latent = self.get_logits(
            conds,
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            get_attns=False,
            return_latent=True,
        )
        # Strip the two tokens added by padding
        return mel_latent[:, :-2]

    # -------------------------------------------------------------------------
    # Inference Input Preparation
    # -------------------------------------------------------------------------

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
            mel_emb, attn_mask = self._prepare_single_gpt_input(
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_inputs[i],
                text_len,
                target_len,
            )
            batched_mel_embs.append(mel_emb)
            attention_masks.append(attn_mask)

        # Stack batched outputs
        batched_mel_emb = torch.stack(batched_mel_embs, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)

        # Create fake input IDs with start_mel_token at the end
        fake_inputs = torch.ones((batch_size, target_len + 1), dtype=torch.long)
        fake_inputs[:, -1] = self.cfg.start_mel_token

        return fake_inputs, batched_mel_emb, attention_mask

    def _prepare_single_gpt_input(
        self,
        cond_latent: Tensor,
        text_input: Tensor,
        max_text_len: int,
        target_len: int,
    ) -> tuple[Tensor, Tensor]:
        """Prepare GPT input for a single sequence.

        Args:
            cond_latent: (cond_num, dim) conditioning for this sequence
            text_input: (text_len,) text tokens for this sequence
            max_text_len: Maximum text length in batch
            target_len: Target sequence length

        Returns:
            mel_emb: (target_len, dim) concatenated embeddings
            attention_mask: (target_len+1,) attention mask
        """
        # Filter out special tokens and add start/stop
        valid_mask = (text_input != self.cfg.stop_text_token) & (text_input != self.cfg.start_text_token)
        text_input = text_input[valid_mask]
        text_input = F.pad(text_input, (1, 0), value=self.cfg.start_text_token)
        text_input = F.pad(text_input, (0, 1), value=self.cfg.stop_text_token)

        # Compute text embeddings
        text_pos = torch.arange(text_input.size(-1))
        text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_pos)

        # Build sequence: [optional_pad][cond][text]
        parts: list[Tensor] = [cond_latent, text_emb]
        attention_mask = torch.ones(target_len + 1, dtype=torch.long)

        # Add left padding if needed
        padding = max_text_len + 2 - text_input.size(-1)
        if padding > 0:
            pad = torch.zeros((padding, cond_latent.size(-1)), dtype=text_emb.dtype)
            parts.insert(0, pad)
            attention_mask[:padding] = 0

        mel_emb = torch.cat(parts)
        assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"

        return mel_emb, attention_mask

    # -------------------------------------------------------------------------
    # Speech Generation
    # -------------------------------------------------------------------------

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
        **hf_generate_kwargs: Any,  # pyright: ignore[reportAny]
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
            cond_lengths = torch.tensor([speech_condition.shape[-1]])
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]])

        # Compute conditioning latents
        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1, 2), cond_lengths)
        logger.info("get_conditioning: %.4fs", time.perf_counter() - t0)

        # Compute or use provided emotion vector
        if emo_vec is None:
            t1 = time.perf_counter()
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1, 2), emo_cond_lengths)
            emo_vec = self.emo_layer(self.emovec_layer(emo_vec))
            logger.info("get_emo_conditioning: %.4fs", time.perf_counter() - t1)
        else:
            logger.info("Using provided emotion vector")

        # Build conditioning latent
        conds_latent = self._build_conditioning_concat(
            speech_conditioning_latent,
            emo_vec,
            torch.zeros(text_inputs.size(0), dtype=torch.long),
        )

        # Prepare GPT inputs
        t2 = time.perf_counter()
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        assert self.inference_model is not None
        self.inference_model.store_mel_emb(inputs_embeds)
        logger.info("prepare_gpt_inputs: %.4fs", time.perf_counter() - t2)

        # Handle additional input tokens
        inputs, attention_mask = self._prepare_generation_inputs(
            input_ids, attention_mask, input_tokens, num_return_sequences, text_inputs.shape[0]
        )

        # Setup generation parameters
        trunc_index = inputs.shape[1]
        logits_processor = self._build_logits_processor(typical_sampling, typical_mass, hf_generate_kwargs)
        max_length = (
            trunc_index + self.cfg.max_mel_tokens - 1
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        # Generate
        t3 = time.perf_counter()
        output = self._run_generation(
            inputs,
            attention_mask,
            inputs_embeds,
            max_length,
            trunc_index,
            num_return_sequences,
            logits_processor,
            hf_generate_kwargs,
        )
        logger.info("generation: %.4fs", time.perf_counter() - t3)
        logger.info("total inference_speech: %.4fs", time.perf_counter() - t0)

        return output[:, trunc_index:], speech_conditioning_latent

    def _prepare_generation_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        input_tokens: Tensor | None,
        num_return_sequences: int,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Prepare inputs for generation, handling additional tokens."""
        if input_tokens is None:
            return input_ids, attention_mask

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

        return inputs, attention_mask

    def _build_logits_processor(
        self,
        typical_sampling: bool,
        typical_mass: float,
        hf_generate_kwargs: dict[str, Any],
    ) -> LogitsProcessorList:
        """Build logits processor list for generation."""
        logits_processor = LogitsProcessorList()

        if typical_sampling:
            if not (0.0 < typical_mass < 1.0):
                msg = f"`typical_mass` must be > 0 and < 1, got {typical_mass}"
                raise ValueError(msg)
            min_tokens = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens))

        return logits_processor

    def _run_generation(
        self,
        inputs: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Tensor,
        max_length: int,
        trunc_index: int,
        num_return_sequences: int,
        logits_processor: LogitsProcessorList,
        hf_generate_kwargs: dict[str, Any],
    ) -> Tensor:
        """Run generation using accel engine or standard inference model."""
        assert self.inference_model is not None

        if self.accel_engine is not None and num_return_sequences == 1:
            output = self.accel_engine.generate(
                inputs,
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=hf_generate_kwargs.get("temperature", 1),
                stop_tokens=[self.cfg.stop_mel_token],
                tts_embeddings=inputs_embeds,
                tts_mel_embedding=self.inference_model.embeddings,
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,
            )
        else:
            output = self.inference_model.generate(
                inputs,
                bos_token_id=self.cfg.start_mel_token,
                pad_token_id=self.cfg.stop_mel_token,
                eos_token_id=self.cfg.stop_mel_token,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=logits_processor,
                num_return_sequences=num_return_sequences,
                **hf_generate_kwargs,
            )

        assert isinstance(output, Tensor)
        return output

    @patch_call(forward)
    def __call__(self) -> None: ...
