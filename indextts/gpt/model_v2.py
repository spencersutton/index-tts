import functools
from typing import TYPE_CHECKING, Any, assert_type

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model, LogitsProcessorList

from indextts.config import (
    NUMBER_MEL_CODES,
    NUMBER_TEXT_TOKENS,
    START_MEL_TOKEN,
    START_TEXT_TOKEN,
    STOP_MEL_TOKEN,
    STOP_TEXT_TOKEN,
)
from indextts.gpt.inference import GPT2InferenceModel
from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings

if TYPE_CHECKING:
    from indextts.accel import AccelInferenceEngine
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.util import patch_call


def null_position_embeddings(range: Tensor, dim: int) -> Tensor:
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


def build_hf_gpt_transformer(
    layers: int, model_dim: int, heads: int, max_mel_seq_len: int, max_text_seq_len: int
) -> tuple[GPT2Model, LearnedPositionEmbeddings, LearnedPositionEmbeddings]:
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
    )


class UnifiedVoice(nn.Module):
    if TYPE_CHECKING:
        accel_engine: AccelInferenceEngine | None
    ds_engine: Any

    emo_layer: nn.Linear
    emovec_layer: nn.Linear
    final_norm: nn.LayerNorm
    gpt: GPT2Model
    inference_model: GPT2InferenceModel
    mel_head: nn.Linear
    speed_emb: nn.Embedding

    mel_embedding: nn.Embedding
    text_embedding: nn.Embedding

    mel_pos_embedding: LearnedPositionEmbeddings
    text_pos_embedding: LearnedPositionEmbeddings

    cond_num: int
    heads: int
    layers: int
    max_conditioning_inputs: int
    max_mel_tokens: int
    max_text_tokens: int
    mel_length_compression: int
    model_dim: int

    cond_mask_pad: nn.ConstantPad1d
    conditioning_encoder: ConformerEncoder
    emo_cond_mask_pad: nn.ConstantPad1d
    emo_conditioning_encoder: ConformerEncoder
    emo_perceiver_encoder: PerceiverResampler
    perceiver_encoder: PerceiverResampler

    def __init__(
        self,
        layers: int = 8,
        model_dim: int = 512,
        heads: int = 8,
        max_text_tokens: int = 120,
        max_mel_tokens: int = 250,
        max_conditioning_inputs: int = 1,
        mel_length_compression: int = 1024,
        condition_num_latent: int = 32,
        use_accel: bool = False,
    ) -> None:
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
        """
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.conditioning_encoder = ConformerEncoder(linear_units=2048, attention_heads=8, num_blocks=6)
        self.perceiver_encoder = PerceiverResampler(model_dim, heads=8, num_latents=self.cond_num)

        self.emo_conditioning_encoder = ConformerEncoder(linear_units=1024, attention_heads=4, num_blocks=4)
        self.emo_perceiver_encoder = PerceiverResampler(1024, heads=4, num_latents=1)

        self.text_embedding = nn.Embedding(NUMBER_TEXT_TOKENS + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        self.mel_embedding = nn.Embedding(NUMBER_MEL_CODES, model_dim)
        (self.gpt, self.mel_pos_embedding, self.text_pos_embedding) = build_hf_gpt_transformer(
            layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs, self.max_text_tokens + 2
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, NUMBER_TEXT_TOKENS + 1)
        self.mel_head = nn.Linear(model_dim, NUMBER_MEL_CODES)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings: list[nn.Embedding] = [self.text_embedding]
        embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

        self.use_accel: bool = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config

    def post_init_gpt2_config(self, use_deepspeed: bool, kv_cache: bool, half: bool, model_dim: int) -> None:
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=NUMBER_MEL_CODES,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=model_dim,
            n_layer=self.layers,
            n_head=self.heads,
        )

        if self.use_accel and torch.cuda.is_available():
            # Check if flash attention is available
            try:
                import flash_attn  # noqa: F401 # type: ignore
            except ImportError:
                raise ImportError(
                    "flash_attn is required for acceleration but not installed. Please install from https://github.com/Dao-AILab/flash-attention/releases/"
                )

            from indextts.accel import AccelInferenceEngine, GPT2AccelModel

            # Create accel model
            accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)

            if half:
                accel_gpt = accel_gpt.half().cuda()
            else:
                accel_gpt = accel_gpt.cuda()
            accel_gpt.eval()

            lm_head_with_norm = nn.Sequential(self.final_norm, self.mel_head)
            self.accel_engine = AccelInferenceEngine(
                model=accel_gpt,
                lm_head=lm_head_with_norm,
                num_layers=self.layers,
                num_heads=self.heads,
                head_dim=model_dim // self.heads,
                block_size=256,
                num_blocks=16,  # Reduce to save memory (16*256 = 4096 tokens capacity)
                use_cuda_graph=True,
            )
            print("acceleration engine initialized")
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed  # type: ignore

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model, mp_size=1, replace_with_kernel_inject=True, dtype=torch.float16
            )
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed  # type: ignore

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model, mp_size=1, replace_with_kernel_inject=True, dtype=torch.float32
            )
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    def set_mel_padding(self, mel_input_tokens: Tensor, mel_lengths: Tensor) -> Tensor:
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = STOP_MEL_TOKEN
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens: Tensor, text_lengths: Tensor) -> Tensor:
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = STOP_TEXT_TOKEN
        return text_input_tokens

    def get_logits(
        self, speech_conditioning_inputs: Tensor, first_inputs: Tensor, second_inputs: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=False)

        assert not isinstance(gpt_out, tuple) and gpt_out.last_hidden_state is not None
        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

    def get_emo_conditioning(self, speech_conditioning_input: Tensor, cond_mel_lengths: Tensor) -> Tensor:
        speech_conditioning_input, mask = self.emo_conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)

    def forward(
        self,
        speech_conditioning_latent: Tensor,
        text_inputs: Tensor,
        text_lengths: Tensor,
        mel_codes: Tensor,
        mel_codes_lengths: Tensor,
        emo_speech_conditioning_latent: Tensor,
        emo_cond_mel_lengths: Tensor,
        emo_vec: Tensor,
        use_speed: Tensor,
    ) -> Tensor:
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(
                emo_speech_conditioning_latent.transpose(1, 2), emo_cond_mel_lengths
            )
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=STOP_TEXT_TOKEN)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=STOP_MEL_TOKEN)

        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat(
            (
                speech_conditioning_latent + emo_vec.unsqueeze(1),
                duration_emb_half.unsqueeze(1),
                duration_emb.unsqueeze(1),
            ),
            1,
        )
        text_inputs = F.pad(text_inputs, (1, 0), value=START_TEXT_TOKEN)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes = F.pad(mel_codes, (1, 0), value=START_MEL_TOKEN)

        mel_emb: Tensor = self.mel_embedding(mel_codes)
        mel_emb += self.mel_pos_embedding.forward(mel_codes)

        _text_logits, mel_logits = self.get_logits(conds, text_emb, mel_emb)
        # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.
        return mel_logits[:, :-2]

    def prepare_gpt_inputs(self, conditional_latents: Tensor, text_inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Prepare the inputs for the GPT2InferenceModel to generate.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
        Returns:
            input_ids: (b, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (b, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (b, s+1) the attention mask for the GPT2InferenceModel.generate()
        """
        b, L = text_inputs.shape[:2]
        device = text_inputs.device
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            valid_mask = (text_inputs[i] != STOP_TEXT_TOKEN) & (text_inputs[i] != START_TEXT_TOKEN)
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=START_TEXT_TOKEN)
            text_input = F.pad(text_input, (0, 1), value=STOP_TEXT_TOKEN)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [conditional_latents.squeeze(0) if single_cond else conditional_latents[i], text_emb]
            # +1 for the start_mel_token
            attention_mask = torch.ones(target_len + 1, dtype=torch.long, device=device)
            # check this text input is padded
            padding: int = L + 2 - text_input.size(-1)
            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros(
                    (padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device
                )  # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb)  # [s, dim]
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # [b, s+1]
        attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = START_MEL_TOKEN
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(
        self,
        speech_condition: Tensor,
        text_inputs: Tensor,
        emo_speech_condition: Tensor | None = None,
        emo_vec: Tensor | None = None,
        input_tokens: Tensor | None = None,
        num_return_sequences: int = 1,
        max_generate_length: int | None = None,
        **hf_generate_kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            speech_condition: (b, d, frames) or (d, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
            input_tokens: additional tokens for generation in shape (b, s) or (s,)
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """

        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition

        speech_conditioning_input, mask = self.conditioning_encoder(
            speech_condition, torch.tensor([speech_condition.shape[-1]], device=text_inputs.device)
        )
        speech_conditioning_latent = self.perceiver_encoder(
            speech_conditioning_input, self.cond_mask_pad(mask.squeeze(1))
        )

        if emo_vec is None:
            print("compute emo vec")
            emo_vec = self.get_emo_conditioning(
                emo_speech_condition.transpose(1, 2),
                torch.tensor([emo_speech_condition.shape[-1]], device=text_inputs.device),
            )
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
        else:
            print("Use the specified emotion vector")

        tmp = torch.zeros(text_inputs.size(0)).to(text_inputs.device)
        duration_emb = self.speed_emb(torch.zeros_like(tmp).long())
        duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())
        conds_latent = torch.cat(
            (
                speech_conditioning_latent + emo_vec.unsqueeze(1),
                duration_emb_half.unsqueeze(1),
                duration_emb.unsqueeze(1),
            ),
            1,
        )
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        self.inference_model.cached_mel_emb = inputs_embeds
        if input_tokens is None:
            inputs = input_ids
        else:
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)
            assert num_return_sequences % input_tokens.shape[0] == 0, (
                "The num_return_sequences must be divisible by the batch number of input_tokens"
            )
            assert num_return_sequences % text_inputs.shape[0] == 0, (
                "The num_return_sequences must be divisible by the batch number of text_inputs"
            )
            b = num_return_sequences // input_ids.shape[0]
            if b > 1:
                input_ids = input_ids.repeat(b, 1)
                attention_mask = attention_mask.repeat(b, 1)
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)
        trunc_index = inputs.shape[1]
        max_length = (
            (trunc_index + self.max_mel_tokens - 1)
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        # Use accel engine if available (single sequence only)
        if self.accel_engine is not None and num_return_sequences == 1:
            output = self.accel_engine.generate(
                inputs,  # fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=float(hf_generate_kwargs.get("temperature", 1)),
                stop_tokens=[STOP_MEL_TOKEN],
                tts_embeddings=inputs_embeds,  # [pad][cond][text] embeddings (87 tokens, NO start_mel_token)
                tts_mel_embedding=self.inference_model.embeddings,  # mel_embedding layer
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,  # text_pos_embedding layer
            )
        else:
            logits_processor = LogitsProcessorList()
            output = self.inference_model.generate(
                inputs,
                bos_token_id=START_MEL_TOKEN,
                pad_token_id=STOP_MEL_TOKEN,
                eos_token_id=STOP_MEL_TOKEN,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=logits_processor,
                num_return_sequences=num_return_sequences,
                **hf_generate_kwargs,
            )
        if isinstance(output, Tensor):
            return output[:, trunc_index:], speech_conditioning_latent
        assert False, "Unexpected output type from GPT2InferenceModel.generate()"
        # GenerateOutput
        output.sequences = output.sequences[:, trunc_index:]
        return output, speech_conditioning_latent

    def get_emo_vec(self, emo_speech_conditioning_latent: Tensor, emo_cond_lengths: Tensor) -> Tensor:
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1, 2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        return self.emo_layer(emo_vec_syn)

    def merge_emo_vec(
        self,
        speech_conditioning_latent: Tensor,
        emo_speech_conditioning_latent: Tensor,
        cond_lengths: Tensor,
        emo_cond_lengths: Tensor,
        alpha: float = 1.0,
    ) -> Tensor:
        emo_vec = self.get_emo_vec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emo_vec(speech_conditioning_latent, cond_lengths)

        return base_vec + alpha * (emo_vec - base_vec)

    @patch_call(forward)
    def __call__(self) -> None: ...
