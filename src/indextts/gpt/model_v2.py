from typing import TYPE_CHECKING, ClassVar, Final, cast, override

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.inference import GPT2InferenceModel
from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings
from indextts.gpt.perceiver import PerceiverResampler
from indextts.util import patch_call, unwrap

if TYPE_CHECKING:
    from indextts.accel import AccelInferenceEngine

MAX_MEL_TOKENS: Final = 1815
START_MEL_TOKEN: Final = 8192
STOP_MEL_TOKEN: Final = START_MEL_TOKEN + 1
NUMBER_MEL_CODES: Final = STOP_MEL_TOKEN + 1

MAX_TEXT_TOKENS: Final = 600
START_TEXT_TOKEN: Final = 0
STOP_TEXT_TOKEN: Final = START_TEXT_TOKEN + 1
NUMBER_TEXT_TOKENS: Final = 12000

MAX_MEL_SEQ_LEN: Final = MAX_MEL_TOKENS + 3
MAX_TEXT_SEQ_LEN: Final = MAX_TEXT_TOKENS + 2
SEQ_LENGTH: Final = MAX_MEL_SEQ_LEN + MAX_TEXT_SEQ_LEN


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
    use_accel: bool
    """Whether to use the acceleration engine (if available)."""

    voice_dim: ClassVar[int] = 1280
    heads: ClassVar[int] = 20
    """Number of attention heads in the GPT transformer."""
    layers: ClassVar[int] = 24
    """Number of transformer layers in the GPT stack."""

    def __init__(self, dim: int = 512, n_latent: int = 32, use_accel: bool = False) -> None:
        super().__init__()

        self.cond_mask_pad = nn.ConstantPad1d((n_latent, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.conditioning_encoder = ConformerEncoder(dim, linear_units=2048, attention_heads=8, num_blocks=6)
        self.perceiver_encoder = PerceiverResampler(dim, self.voice_dim, heads=8, num_latents=n_latent)

        self.emo_conditioning_encoder = ConformerEncoder(dim, linear_units=1024, attention_heads=4, num_blocks=4)
        self.emo_perceiver_encoder = PerceiverResampler(dim, 1024, heads=4, num_latents=1)

        self.emo_layer = nn.Linear(self.voice_dim, self.voice_dim)
        self.emovec_layer = nn.Linear(1024, self.voice_dim)

        self.text_embedding = nn.Embedding(NUMBER_TEXT_TOKENS + 1, self.voice_dim)
        self.mel_embedding = nn.Embedding(NUMBER_MEL_CODES, self.voice_dim)

        self.gpt = transformers.GPT2Model(
            transformers.GPT2Config(
                vocab_size=256,  # Unused.
                n_positions=SEQ_LENGTH,
                n_ctx=SEQ_LENGTH,
                n_embd=self.voice_dim,
                n_layer=self.layers,
                n_head=self.heads,
            )
        )
        # Override the built in positional embeddings
        del self.gpt.wpe

        def wpe_override(x: Tensor) -> Tensor:
            return torch.zeros((x.shape[0], x.shape[1], self.voice_dim), device=x.device)

        self.gpt.wpe = cast(nn.Embedding, wpe_override)
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.mel_pos_embedding = LearnedPositionEmbeddings(MAX_MEL_SEQ_LEN)
        self.text_pos_embedding = LearnedPositionEmbeddings(MAX_TEXT_SEQ_LEN)

        self.final_norm = nn.LayerNorm(self.voice_dim)
        self.text_head = nn.Linear(self.voice_dim, NUMBER_TEXT_TOKENS + 1)
        self.mel_head = nn.Linear(self.voice_dim, NUMBER_MEL_CODES)

        self.speed_emb = nn.Embedding(2, self.voice_dim)
        self.speed_emb.weight.data.normal_(std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding, self.mel_embedding]
        for module in embeddings:
            module.weight.data.normal_(std=0.02)

        self.use_accel = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config

    @override
    def forward(
        self, speech_conditioning_latent: Tensor, text_inputs: Tensor, mel_codes: Tensor, emo_vec: Tensor
    ) -> Tensor:
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        text_inputs = F.pad(text_inputs, (1, 0), value=START_TEXT_TOKEN)
        text_inputs = F.pad(text_inputs, (0, 1), value=STOP_TEXT_TOKEN)

        mel_codes = F.pad(mel_codes, (1, 0), value=START_MEL_TOKEN)
        mel_codes = F.pad(mel_codes, (0, 1), value=STOP_MEL_TOKEN)

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
            valid_mask = (row != STOP_TEXT_TOKEN) & (row != START_TEXT_TOKEN)

            text_input = row[valid_mask]
            text_input = F.pad(text_input, (1, 0), value=START_TEXT_TOKEN)
            text_input = F.pad(text_input, (0, 1), value=STOP_TEXT_TOKEN)
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
        fake_inputs[:, -1] = START_MEL_TOKEN
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
        max_generate_length: int,
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
        max_length = trunc_index + max_generate_length

        # Use accel engine if available (single sequence only)
        if self.accel_engine is not None:
            output = self.accel_engine.generate(
                inputs_ids,  # fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=cast(float, hf_generate_kwargs.get("temperature", 1)),
                stop_tokens=[STOP_MEL_TOKEN],
                tts_embeddings=inputs_embeds,  # [pad][cond][text] embeddings (87 tokens, NO start_mel_token)
                tts_mel_embedding=self.inference_model.embeddings,  # mel_embedding layer
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,  # text_pos_embedding layer
            )
        else:
            output = self.inference_model.generate(
                inputs_ids,
                bos_token_id=START_MEL_TOKEN,
                pad_token_id=STOP_MEL_TOKEN,
                eos_token_id=STOP_MEL_TOKEN,
                attention_mask=attention_mask,
                max_length=max_length,
                **hf_generate_kwargs,  # pyright: ignore
            )
        return output[:, trunc_index:]  # pyright: ignore[reportUnknownVariableType]

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


def post_init_gpt2_config(model: UnifiedVoice) -> None:
    gpt_config = transformers.GPT2Config(
        vocab_size=NUMBER_MEL_CODES,
        n_positions=SEQ_LENGTH,
        n_ctx=SEQ_LENGTH,
        n_embd=model.voice_dim,
        n_layer=model.layers,
        n_head=model.heads,
    )

    if model.use_accel and torch.cuda.is_available():
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
        accel_gpt.load_state_dict(model.gpt.state_dict(), strict=False)

        lm_head_with_norm = nn.Sequential(model.final_norm, model.mel_head)
        model.accel_engine = AccelInferenceEngine(
            model=accel_gpt.cuda().eval(),
            lm_head=lm_head_with_norm,
            num_layers=model.layers,
            num_heads=model.heads,
            head_dim=model.voice_dim // model.heads,
            block_size=256,
            num_blocks=16,  # Reduce to save memory (16*256 = 4096 tokens capacity)
        )
        print("acceleration engine initialized")
    model.inference_model = GPT2InferenceModel(
        gpt_config, model.gpt, model.mel_pos_embedding, model.mel_embedding, model.final_norm, model.mel_head
    )
    model.inference_model = model.inference_model.eval()

    model.gpt.wte = model.mel_embedding
