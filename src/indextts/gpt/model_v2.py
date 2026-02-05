from typing import TYPE_CHECKING, Final, cast, override

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn

from indextts.config import UnifiedVoiceConfig
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.inference import GPT2InferenceModel
from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings
from indextts.gpt.perceiver import PerceiverResampler
from indextts.util import patch_call, unwrap

if TYPE_CHECKING:
    from indextts.accel import AccelInferenceEngine


class UnifiedVoice(nn.Module):
    """Unified voice/text GPT model.

    Attribute descriptions are documented inline (directly beneath each annotated attribute).
    """

    if TYPE_CHECKING:
        accel_engine: AccelInferenceEngine | None
        """Optional accelerated generation engine (CUDA/flash-attn path, initialized lazily)."""

    emo_layer: nn.Linear
    """Final projection applied to the emotion vector in the GPT embedding space (dim -> dim)."""
    emovec_layer: nn.Linear
    """Projects the emotion-conditioning latent (1024-d) into the GPT embedding space (1024 -> dim)."""
    final_norm: nn.LayerNorm
    """LayerNorm applied to transformer hidden states before projecting to logits."""
    gpt: transformers.GPT2Model
    """Core GPT-2 transformer backbone that consumes concatenated conditioning/text/mel embeddings."""
    inference_model: GPT2InferenceModel  # pyright: ignore[reportUninitializedInstanceVariable]
    """Generation-oriented wrapper around the transformer (caching/positioning + `generate`)."""
    mel_head: nn.Linear
    """Output projection from hidden size (dim) to the mel-code vocabulary size."""
    text_head: nn.Linear
    """Output projection from hidden size (dim) to the text vocabulary size."""
    speed_emb: nn.Embedding
    """Embeddings for speed/control tokens that are appended to the conditioning prefix."""
    mel_embedding: nn.Embedding
    """Token embedding table for mel-code ids."""
    text_embedding: nn.Embedding
    """Token embedding table for text token ids."""
    mel_pos_embedding: LearnedPositionEmbeddings
    """Learned positional embeddings for the mel-code segment."""
    text_pos_embedding: LearnedPositionEmbeddings
    """Learned positional embeddings for the text segment."""
    heads: Final = 20
    """Number of attention heads in the GPT transformer."""
    layers: Final = 24
    """Number of transformer layers in the GPT stack."""
    max_mel_tokens: int
    """Maximum mel-code tokens supported (used to size positional embeddings / generation limits)."""
    max_text_tokens: int
    """Maximum text tokens supported (used to size positional embeddings / padding logic)."""
    cond_mask_pad: nn.ConstantPad1d
    """Pads the conditioning attention mask to account for inserted conditioning latents."""
    conditioning_encoder: ConformerEncoder
    """Conformer encoder that processes speech conditioning features before Perceiver resampling."""
    emo_cond_mask_pad: nn.ConstantPad1d
    """Pads the emotion-conditioning attention mask (single-latent Perceiver)."""
    emo_conditioning_encoder: ConformerEncoder
    """Conformer encoder for emotion-specific conditioning features."""
    emo_perceiver_encoder: PerceiverResampler
    """Perceiver resampler that reduces emotion conditioning to a single latent token."""
    perceiver_encoder: PerceiverResampler
    """Perceiver resampler that reduces speech conditioning to `condition_num_latent` latent tokens."""
    cfg: UnifiedVoiceConfig
    """Model configuration (token ids, vocab sizes, architecture hyperparameters, limits)."""
    dim: Final = 1280
    """Model hidden dimension (GPT embedding size)."""
    use_accel: bool
    """Whether to use the acceleration engine (if available)."""

    def __init__(
        self, cfg: UnifiedVoiceConfig, condition_num_latent: int = 32, use_accel: bool = False, dim: int = 1280
    ) -> None:
        """
        Args:
            layers: Number of layers in transformer stack.
            heads: Number of transformer heads. Must be divisible by 1280. Recommend 1280//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
        """
        super().__init__()
        self.cfg = cfg
        self.max_mel_tokens = cfg.max_mel_tokens
        self.max_text_tokens = cfg.max_text_tokens
        self.cond_mask_pad = nn.ConstantPad1d((condition_num_latent, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.conditioning_encoder = ConformerEncoder(dim=512, linear_units=2048, attention_heads=8, num_blocks=6)
        self.perceiver_encoder = PerceiverResampler(dim, heads=8, num_latents=condition_num_latent)

        self.emo_conditioning_encoder = ConformerEncoder(dim=512, linear_units=1024, attention_heads=4, num_blocks=4)
        self.emo_perceiver_encoder = PerceiverResampler(1024, heads=4, num_latents=1)

        self.emo_layer = nn.Linear(dim, dim)
        self.emovec_layer = nn.Linear(1024, dim)

        self.text_embedding = nn.Embedding(cfg.number_text_tokens + 1, dim)
        self.mel_embedding = nn.Embedding(cfg.number_mel_codes, dim)
        max_mel_seq_len = self.max_mel_tokens + 3
        max_text_seq_len = self.max_text_tokens + 2

        self.gpt = transformers.GPT2Model(
            transformers.GPT2Config(
                vocab_size=256,  # Unused.
                n_positions=max_mel_seq_len + max_text_seq_len,
                n_ctx=max_mel_seq_len + max_text_seq_len,
                n_embd=dim,
                n_layer=self.layers,
                n_head=self.heads,
            )
        )
        # Override the built in positional embeddings
        del self.gpt.wpe

        def wpe_override(x: Tensor) -> Tensor:
            return torch.zeros((x.shape[0], x.shape[1], dim), device=x.device)

        self.gpt.wpe = cast(nn.Embedding, wpe_override)
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_seq_len)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_seq_len)

        self.final_norm = nn.LayerNorm(dim)
        self.text_head = nn.Linear(dim, cfg.number_text_tokens + 1)
        self.mel_head = nn.Linear(dim, cfg.number_mel_codes)

        self.speed_emb = nn.Embedding(2, dim)
        self.speed_emb.weight.data.normal_(std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding, self.mel_embedding]
        for module in embeddings:
            module.weight.data.normal_(std=0.02)

        self.use_accel = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config

    def post_init_gpt2_config(self, half: bool) -> None:
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = transformers.GPT2Config(
            vocab_size=self.cfg.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.dim,
            n_layer=self.layers,
            n_head=self.heads,
        )

        if self.use_accel and torch.cuda.is_available():
            # Check if flash attention is available
            try:
                import flash_attn  # noqa: F401  # pyright: ignore
            except ImportError as err:
                raise ImportError(
                    "flash_attn is required for acceleration but not installed. Please install from https://github.com/Dao-AILab/flash-attention/releases/"
                ) from err

            from indextts.accel import AccelInferenceEngine, GPT2AccelModel

            # Create accel model
            accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)

            if half:
                accel_gpt = accel_gpt.half()

            lm_head_with_norm = nn.Sequential(self.final_norm, self.mel_head)
            self.accel_engine = AccelInferenceEngine(
                model=accel_gpt.cuda().eval(),
                lm_head=lm_head_with_norm,
                num_layers=self.layers,
                num_heads=self.heads,
                head_dim=self.dim // self.heads,
                block_size=256,
                num_blocks=16,  # Reduce to save memory (16*256 = 4096 tokens capacity)
            )
            print("acceleration engine initialized")
        self.inference_model = GPT2InferenceModel(
            gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head
        )
        self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    @override
    def forward(
        self, speech_conditioning_latent: Tensor, text_inputs: Tensor, mel_codes: Tensor, emo_vec: Tensor
    ) -> Tensor:
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        text_inputs = F.pad(text_inputs, (1, 0), value=self.cfg.start_text_token)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.cfg.stop_text_token)

        mel_codes = F.pad(mel_codes, (1, 0), value=self.cfg.start_mel_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.cfg.stop_mel_token)

        mel_emb = self.mel_embedding(mel_codes) + self.mel_pos_embedding(mel_codes.shape[1])
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs.shape[1])

        conds = self.combine_latents(speech_conditioning_latent, emo_vec, text_inputs)
        output = self.gpt(
            inputs_embeds=torch.cat([conds, text_emb, mel_emb], dim=1), return_dict=True, output_attentions=False
        )
        assert not isinstance(output, tuple)

        offset = conds.shape[1]
        enc = unwrap(output.last_hidden_state)[:, offset:]
        enc = self.final_norm(enc)

        # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.
        return enc[:, -mel_emb.shape[1] : -2]

    def prepare_gpt_inputs(self, latent: Tensor, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Prepare the inputs for the GPT2InferenceModel to generate.
        Args:
            conds_latent: (B, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (B, L)
        Returns:
            input_ids: (B, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (B, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (B, s+1) the attention mask for the GPT2InferenceModel.generate()
        """
        is_single_condition = latent.ndim == 3 and latent.shape[0] == 1
        assert is_single_condition or latent.shape[0] == inputs.shape[0], (
            f"batch size mismatch: {latent.shape[0]} vs {inputs.shape[0]}"
        )
        batched_mel_emb: list[Tensor] = []
        attention_masks: list[Tensor] = []
        target_len = latent.shape[1] + inputs.shape[1] + 2
        for i, row in enumerate(inputs):
            valid_mask = (row != self.cfg.stop_text_token) & (row != self.cfg.start_text_token)

            text_input = row[valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.cfg.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.cfg.stop_text_token)
            text_input_pos = torch.arange(text_input.size(-1), device=inputs.device)

            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)

            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [latent.squeeze(0) if is_single_condition else latent[i], text_emb]

            # +1 for the start_mel_token
            attention_mask = inputs.new_ones(target_len + 1)

            # check this text input is padded
            padding = inputs.shape[1] + 2 - text_input.size(-1)

            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = inputs.new_zeros((padding, latent.size(-1)))  # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            batched_mel_emb.append(torch.cat(conds_text_emb))
            attention_masks.append(attention_mask)
        mel_embedding_batch = torch.stack(batched_mel_emb)
        attention_mask = torch.stack(attention_masks)
        fake_inputs = inputs.new_ones((mel_embedding_batch.shape[0], mel_embedding_batch.shape[1] + 1))
        fake_inputs[:, -1] = self.cfg.start_mel_token
        return fake_inputs, mel_embedding_batch, attention_mask

    def combine_latents(self, speech_conditioning_latent: Tensor, emo_vec: Tensor, text_inputs: Tensor) -> Tensor:
        template = text_inputs.new_zeros(text_inputs.shape[0])
        return torch.cat(
            (
                speech_conditioning_latent + emo_vec.unsqueeze(1),
                self.speed_emb(torch.ones_like(template)).unsqueeze(1),
                self.speed_emb(torch.zeros_like(template)).unsqueeze(1),
            ),
            dim=1,
        )

    def inference_speech(
        self,
        speech_conditioning_latent: Tensor,
        text_inputs: Tensor,
        *,
        emo_vec: Tensor,
        max_generate_length: int | None = None,
        **hf_generate_kwargs: object,
    ) -> Tensor:
        """
        Args:
            speech_condition: (B, D, frames) or (D, frames)
            text_inputs: (B, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (B,) or (1,)
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """
        conds_latent = self.combine_latents(speech_conditioning_latent, emo_vec, text_inputs)
        inputs_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        self.inference_model.cached_mel_emb = inputs_embeds
        trunc_index = inputs_ids.shape[1]
        max_length = (
            (trunc_index + self.max_mel_tokens - 1)
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        # Use accel engine if available (single sequence only)
        if self.accel_engine is not None:
            output = self.accel_engine.generate(
                inputs_ids,  # fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=cast(float, hf_generate_kwargs.get("temperature", 1)),
                stop_tokens=[self.cfg.stop_mel_token],
                tts_embeddings=inputs_embeds,  # [pad][cond][text] embeddings (87 tokens, NO start_mel_token)
                tts_mel_embedding=self.inference_model.embeddings,  # mel_embedding layer
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,  # text_pos_embedding layer
            )
        else:
            output = self.inference_model.generate(
                inputs_ids,
                bos_token_id=self.cfg.start_mel_token,
                pad_token_id=self.cfg.stop_mel_token,
                eos_token_id=self.cfg.stop_mel_token,
                attention_mask=attention_mask,
                max_length=max_length,
                **hf_generate_kwargs,  # pyright: ignore
            )
        return output[:, trunc_index:]

    def process_speech_condition(self, condition: Tensor) -> Tensor:
        if condition.ndim == 2:
            condition = condition.unsqueeze(0)

        input, mask = self.conditioning_encoder.__call__(condition)
        mask = self.cond_mask_pad(mask.squeeze(1))
        return self.perceiver_encoder(input, mask)

    def get_emo_vec(self, latent: Tensor) -> Tensor:
        input, mask = self.emo_conditioning_encoder.__call__(latent)
        mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(input, mask)
        vector = self.emovec_layer(conds.squeeze(1))
        return self.emo_layer(vector)

    @patch_call(forward)
    def __call__(self) -> None: ...
