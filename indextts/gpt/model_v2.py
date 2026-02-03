from typing import TYPE_CHECKING, Any, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model, LogitsProcessorList

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
    gpt: GPT2Model
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

    def __init__(self, condition_num_latent: int = 32, use_accel: bool = False, dim: int = 1280) -> None:
        """
        Args:
            layers: Number of layers in transformer stack.
            heads: Number of transformer heads. Must be divisible by 1280. Recommend 1280//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
        """
        super().__init__()
        self.cond_mask_pad = nn.ConstantPad1d((condition_num_latent, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.conditioning_encoder = ConformerEncoder(linear_units=2048, attention_heads=8, num_blocks=6)
        self.perceiver_encoder = PerceiverResampler(1280, heads=8, num_latents=condition_num_latent)

        self.emo_conditioning_encoder = ConformerEncoder(linear_units=1024, attention_heads=4, num_blocks=4)
        self.emo_perceiver_encoder = PerceiverResampler(1024, heads=4, num_latents=1)

        self.emo_layer = nn.Linear(1280, 1280)
        self.emovec_layer = nn.Linear(1024, 1280)

        self.text_embedding = nn.Embedding(12000 + 1, 1280)
        self.mel_embedding = nn.Embedding(8194, 1280)
        max_mel_seq_len = 1815 + 3
        max_text_seq_len = 600 + 2

        self.gpt = GPT2Model(
            GPT2Config(
                vocab_size=256,  # Unused.
                n_positions=max_mel_seq_len + max_text_seq_len,
                n_ctx=max_mel_seq_len + max_text_seq_len,
                n_embd=1280,
                n_layer=24,
                n_head=20,
            )
        )
        # Override the built in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = lambda x: torch.zeros((x.shape[0], x.shape[1], 1280), device=x.device)  # type: ignore
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_seq_len)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_seq_len)

        self.final_norm = nn.LayerNorm(1280)
        self.text_head = nn.Linear(1280, 12000 + 1)
        self.mel_head = nn.Linear(1280, 8194)

        self.speed_emb = nn.Embedding(2, 1280)
        self.speed_emb.weight.data.normal_(std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding, self.mel_embedding]
        for module in embeddings:
            module.weight.data.normal_(std=0.02)

        self.use_accel = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config

    def post_init_gpt2_config(self, half: bool) -> None:
        seq_length = 1815 + 600 + 2
        gpt_config = GPT2Config(
            vocab_size=8194, n_positions=seq_length, n_ctx=seq_length, n_embd=1280, n_layer=24, n_head=20
        )

        if self.use_accel and torch.cuda.is_available():
            # Check if flash attention is available
            try:
                import flash_attn  # noqa: F401  # type: ignore
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
                num_layers=24,
                num_heads=20,
                head_dim=1280 // 20,
            )
            print("acceleration engine initialized")
        self.inference_model = GPT2InferenceModel(
            gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head
        )
        self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    @override
    def forward(
        self,
        speech_conditioning_latent: Float[Tensor, "B S D"],
        text_inputs: Int[Tensor, "B L"],
        mel_codes: Int[Tensor, "B M"],
        emo_vec: Float[Tensor, "B D"],
        device: torch.device,
    ) -> Tensor:
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        text_inputs = F.pad(text_inputs, (1, 0), value=0)
        text_inputs = F.pad(text_inputs, (0, 1), value=1)

        mel_codes = F.pad(mel_codes, (1, 0), value=8192)
        mel_codes = F.pad(mel_codes, (0, 1), value=8193)

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

    def prepare_gpt_inputs(
        self, latent: Float[Tensor, "B S D"], inputs: Int[Tensor, "B T"]
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        B = inputs.size(0)
        L = inputs.size(1)
        device = inputs.device
        single_cond = latent.ndim == 3 and latent.shape[0] == 1
        if not single_cond:
            assert latent.shape[0] == B, f"batch size mismatch: {latent.shape[0]} vs {B}"
        batched_mel_emb: list[Tensor] = []
        attention_masks: list[Tensor] = []
        target_len = latent.shape[1] + L + 2
        for i in range(inputs.size(0)):
            valid_mask = (inputs[i] != 1) & (inputs[i] != 0)

            text_input = inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=0)
            text_input = F.pad(text_input, (0, 1), value=1)
            text_input_pos = torch.arange(text_input.size(-1), device=device)

            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)

            # concatenate [conditional latents][text embeddings]
            conds_text_emb: list[Tensor] = [latent.squeeze(0) if single_cond else latent[i], text_emb]

            # +1 for the start_mel_token
            attention_mask = torch.ones(target_len + 1, dtype=torch.long, device=device)

            # check this text input is padded
            padding: int = L + 2 - text_input.size(-1)

            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros((padding, latent.size(-1)), dtype=text_emb.dtype, device=device)  # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb)  # [s, dim]
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # [B, s, dim]
        batched_mel_emb_tensor = torch.stack(batched_mel_emb)
        # [B, s + 1]
        attention_mask = torch.stack(attention_masks)
        # [B, s + 1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb_tensor.shape[0],
                batched_mel_emb_tensor.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = 8192
        return fake_inputs, batched_mel_emb_tensor, attention_mask

    def combine_latents(
        self,
        speech_conditioning_latent: Float[Tensor, "B S D"],
        emo_vec: Float[Tensor, "B D"],
        text_inputs: Int[Tensor, "B T"],
    ) -> Tensor:
        template = text_inputs.new_zeros(text_inputs.size(0))
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
        speech_conditioning_latent: Float[Tensor, "B S D"],
        text_inputs: Int[Tensor, "B T"],
        *,
        emo_vec: Float[Tensor, "B D"],
        max_generate_length: int | None = None,
        **hf_generate_kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
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
        max_length = (trunc_index + 1815 - 1) if max_generate_length is None else trunc_index + max_generate_length

        # Use accel engine if available (single sequence only)
        if self.accel_engine is not None:
            output = self.accel_engine.generate(
                inputs_ids,  # fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=float(hf_generate_kwargs.get("temperature", 1)),  # pyright: ignore
                stop_tokens=[8193],
                tts_embeddings=inputs_embeds,  # [pad][cond][text] embeddings (87 tokens, NO start_mel_token)
                tts_mel_embedding=self.inference_model.embeddings,  # mel_embedding layer
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,  # text_pos_embedding layer
            )
        else:
            output = self.inference_model.generate(
                inputs_ids,
                bos_token_id=8192,
                pad_token_id=8193,
                eos_token_id=8193,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=LogitsProcessorList(),
                num_return_sequences=1,
                **hf_generate_kwargs,  # pyright: ignore
            )
        return output[:, trunc_index:]

    def process_speech_condition(self, condition: Float[Tensor, "B T D"]) -> Tensor:
        if condition.ndim == 2:
            condition = condition.unsqueeze(0)

        input, mask = self.conditioning_encoder.__call__(condition)
        mask = self.cond_mask_pad(mask.squeeze(1))
        return self.perceiver_encoder(input, mask)

    def get_emo_vec(self, latent: Float[Tensor, "B T D"]) -> Tensor:
        input, mask = self.emo_conditioning_encoder.__call__(latent)
        mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(input, mask)
        vector = self.emovec_layer(conds.squeeze(1))
        return self.emo_layer(vector)

    @patch_call(forward)
    def __call__(self) -> None: ...
