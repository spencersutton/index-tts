from __future__ import annotations

import importlib.util
import logging
import time
from typing import TYPE_CHECKING, Any, cast, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    Cache,
    GenerationMixin,
    GPT2Config,
    GPT2Model,
    GPT2PreTrainedModel,
    LogitsProcessorList,
)
from transformers.generation.logits_process import TypicalLogitsWarper
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings
from indextts.gpt.perceiver import PerceiverResampler
from indextts.util import patch_call

logger = logging.getLogger(__name__)


class NullPositionEmbedding(nn.Embedding):
    def __init__(self, dim: int) -> None:
        super().__init__(1, dim)
        del self.weight

    @override
    def forward(self, input: Tensor) -> Tensor:
        return torch.zeros((input.shape[0], input.shape[1], self.embedding_dim))

    @patch_call(forward)
    def __call__(self) -> None: ...


class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    if TYPE_CHECKING:
        lm_head: nn.Sequential[nn.LayerNorm | nn.Linear]
    text_pos_embedding: LearnedPositionEmbeddings
    transformer: GPT2Model
    kv_cache: bool
    cached_mel_emb: Tensor | None
    device_map: dict[int, int] | None
    model_parallel: bool

    def __init__(
        self,
        config: GPT2Config,
        gpt: GPT2Model,
        text_pos_emb: LearnedPositionEmbeddings,
        embeddings: nn.Embedding,
        norm: nn.LayerNorm,
        linear: nn.Linear,
        kv_cache: bool = False,
    ) -> None:
        super().__init__(config)
        # Note: the argument named `text_pos_emb` here actually represents the mel position embedding
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    def parallelize(self, device_map: dict[int, int] | None = None) -> None:
        self.device_map = (
            get_device_map(
                len(self.transformer.h),
                range(max(1, torch.cuda.device_count())),
            )
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head
        self.model_parallel = True

    def deparallelize(self) -> None:
        self.transformer.deparallelize()
        self.transformer = self.transformer
        self.lm_head = self.lm_head
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @override
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def store_mel_emb(self, mel_emb: torch.Tensor) -> None:
        self.cached_mel_emb = mel_emb

    @override
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Tensor,
    ) -> dict[str, Any]:
        inputs_embeds = kwargs.get("inputs_embeds")  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1].unsqueeze(-1)

        position_ids = kwargs.get("position_ids")

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": inputs_embeds,
        }

    @override
    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithCrossAttentions | tuple[Tensor, ...]:
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb += self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(text_emb.shape[0] // self.cached_mel_emb.shape[0], 0)
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            assert attention_mask is not None
            emb += self.text_pos_embedding.get_fixed_embedding(attention_mask.shape[1] - mel_len)
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        assert not isinstance(transformer_outputs, tuple)
        hidden_states = cast(Tensor, transformer_outputs[0])

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head[1].weight.device)

        lm_logits = cast(torch.FloatTensor, self.lm_head(hidden_states))

        if not return_dict:
            return (lm_logits, *transformer_outputs[1:])  # pyright: ignore[reportUnknownVariableType]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @patch_call(forward)
    def __call__(self) -> None: ...


def _build_hf_gpt_transformer(
    layers: int,
    model_dim: int,
    heads: int,
    max_mel_seq_len: int,
    max_text_seq_len: int,
) -> tuple[GPT2Model, LearnedPositionEmbeddings, LearnedPositionEmbeddings, None, None]:
    """GPT-2 implemented by the HuggingFace library."""

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=True,
        use_cache=False,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = NullPositionEmbedding(model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
        None,
        None,
    )


class UnifiedVoice(nn.Module):
    gst_encoder: nn.Module | None
    inference_model: GPT2InferenceModel | None
    number_text_tokens = 12000
    start_text_token = 0
    stop_text_token = 1
    number_mel_codes = 8194
    start_mel_token = 8192
    stop_mel_token = 8193
    layers = 24
    heads = 20
    max_mel_tokens = 1815
    max_text_tokens = 600
    model_dim = 1280
    max_conditioning_inputs = 1
    mel_length_compression = 1024
    cond_num = 32
    mel_solo_embedding = 0
    text_solo_embedding = 0
    ds_engine: object = None

    def __init__(self, use_accel: bool = False) -> None:
        super().__init__()

        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=512,
            linear_units=2048,
            attention_heads=8,
            num_blocks=6,
            input_layer="conv2d2",
        )
        self.perceiver_encoder = PerceiverResampler(
            self.model_dim,
            dim_context=512,
            ff_mult=2,
            heads=8,
            num_latents=self.cond_num,
        )
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

        self.text_embedding = nn.Embedding(self.number_text_tokens + 1, self.model_dim)
        self.emo_layer = nn.Linear(self.model_dim, self.model_dim)
        self.emovec_layer = nn.Linear(1024, self.model_dim)

        self.mel_embedding = nn.Embedding(self.number_mel_codes, self.model_dim)
        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = _build_hf_gpt_transformer(
            self.layers,
            self.model_dim,
            self.heads,
            self.max_mel_tokens + 2 + self.max_conditioning_inputs,
            self.max_text_tokens + 2,
        )

        self.final_norm = nn.LayerNorm(self.model_dim)
        self.text_head = nn.Linear(self.model_dim, self.number_text_tokens + 1)
        self.mel_head = nn.Linear(self.model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, self.model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

        self.use_accel = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config
        self.inference_model = None
        self.gst_encoder = None

    def post_init_gpt2_config(self, use_deepspeed: bool = False, kv_cache: bool = False, half: bool = False) -> None:
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )

        if self.use_accel and torch.cuda.is_available():
            # Check if flash attention is available
            if importlib.util.find_spec("flash_attn") is None:
                msg = "flash_attn is required for acceleration but not installed. Please install from https://github.com/Dao-AILab/flash-attention/releases/"
                raise ImportError(msg)

            from indextts.accel import AccelInferenceEngine, GPT2AccelModel  # noqa: PLC0415

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
                head_dim=self.model_dim // self.heads,
                block_size=256,
                num_blocks=16,  # Reduce to save memory (16*256 = 4096 tokens capacity)
                use_cuda_graph=True,
            )
            logger.info("acceleration engine initialized")
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        assert self.inference_model is not None
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]

            self.ds_engine = deepspeed.init_inference(  # pyright: ignore[reportUnknownMemberType]
                model=self.inference_model,
                mp_size=1,
                replace_with_kernel_inject=True,
                dtype=torch.float16,
            )
            self.inference_model = self.ds_engine.module.eval()  # pyright: ignore
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]

            self.ds_engine = deepspeed.init_inference(  # pyright: ignore[reportUnknownMemberType]
                model=self.inference_model,
                mp_size=1,
                replace_with_kernel_inject=True,
                dtype=torch.float32,
            )
            self.inference_model = self.ds_engine.module.eval()  # pyright: ignore
        else:
            self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(
        self,
        input: Tensor,  # noqa: A002
        start_token: int,
        stop_token: int,
    ) -> tuple[Tensor, Tensor]:
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens: Tensor, mel_lengths: Tensor) -> Tensor:
        """Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens: Tensor, text_lengths: Tensor) -> Tensor:
        """Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

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
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt.forward(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        assert isinstance(gpt_out, BaseModelOutputWithPastAndCrossAttentions)
        if get_attns:
            assert gpt_out.attentions is not None
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        assert gpt_out.last_hidden_state is not None
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm.forward(enc)

        if return_latent:
            assert second_inputs is not None
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head.forward(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            assert second_head is not None
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head.forward(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        return first_logits

    def get_conditioning(self, speech_conditioning_input: Tensor, cond_mel_lengths: Tensor) -> Tensor:
        speech_conditioning_input, mask = self.conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )  # (b, s, d), (b, 1, s)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        return self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)

    def get_emo_conditioning(self, speech_conditioning_input: Tensor, cond_mel_lengths: Tensor) -> Tensor:
        speech_conditioning_input, mask = self.emo_conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)

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
        """Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode.

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """
        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(
                speech_conditioning_latent.transpose(1, 2), cond_mel_lengths
            )

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(
                emo_speech_conditioning_latent.transpose(1, 2),
                emo_cond_mel_lengths,
            )
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)
            assert emo_vec is not None

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        assert use_speed is not None
        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        assert isinstance(duration_emb, Tensor)
        assert isinstance(duration_emb_half, Tensor)
        conds = torch.cat(
            (
                speech_conditioning_latent + emo_vec.unsqueeze(1),
                duration_emb_half.unsqueeze(1),
                duration_emb.unsqueeze(1),
            ),
            1,
        )
        text_inputs, _text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, _mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )

        mel_emb = self.mel_embedding(mel_codes)
        mel_emb += self.mel_pos_embedding(mel_codes)

        _text_logits, mel_logits = self.get_logits(
            conds,
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            get_attns=False,
            return_latent=True,
        )
        return mel_logits[
            :, :-2
        ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def prepare_gpt_inputs(
        self,
        conditional_latents: Tensor,
        text_inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare the inputs for the GPT2InferenceModel to generate.

        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)

        Returns:
            input_ids: (b, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (b, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (b, s+1) the attention mask for the GPT2InferenceModel.generate()

        """
        b, L = text_inputs.shape[:2]
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb_list: list[Tensor] = []
        attention_masks: list[Tensor] = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1))
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]
            # +1 for the start_mel_token
            attention_mask = torch.ones(target_len + 1, dtype=torch.long)
            # check this text input is padded
            padding: int = L + 2 - text_input.size(-1)
            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros(
                    (padding, conditional_latents.size(-1)),
                    dtype=text_emb.dtype,
                )  # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb)  # [s, dim]
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb_list.append(mel_emb)
            attention_masks.append(attention_mask)
        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb_list, dim=0)
        # [b, s+1]
        attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
        )
        fake_inputs[:, -1] = self.start_mel_token
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
        **hf_generate_kwargs: Any,  # pyright: ignore[reportAny]
    ) -> tuple[Tensor, Tensor]:
        t0 = time.perf_counter()
        """Args:
        speech_condition: (b, d, frames) or (d, frames)
        text_inputs: (b, L)
        cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
        input_tokens: additional tokens for generation in shape (b, s) or (s,)
        max_generate_length: limit the number of generated tokens
        hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`.

        """
        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]])
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]])

        t1 = time.perf_counter()
        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1, 2), cond_lengths)
        logger.info("get_conditioning: %.4fs", time.perf_counter() - t1)

        if emo_vec is None:
            logger.info("compute emo vec")
            t2 = time.perf_counter()
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1, 2), emo_cond_lengths)
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
            assert emo_vec is not None
            logger.info("get_emo_conditioning: %.4fs", time.perf_counter() - t2)
        else:
            logger.info("Use the specified emotion vector")

        tmp = torch.zeros(text_inputs.size(0))
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
        t3 = time.perf_counter()
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        assert self.inference_model is not None
        self.inference_model.store_mel_emb(inputs_embeds)
        logger.info("prepare_gpt_inputs: %.4fs", time.perf_counter() - t3)
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
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            # employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                msg = f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}"
                raise ValueError(msg)
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
        max_length = (
            (trunc_index + self.max_mel_tokens - 1)
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        t4 = time.perf_counter()
        # Use accel engine if available (single sequence only)
        if self.accel_engine is not None and num_return_sequences == 1:
            output = self.accel_engine.generate(
                inputs,  # fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=hf_generate_kwargs.get("temperature", 1),  # pyright: ignore[reportAny]
                stop_tokens=[self.stop_mel_token],
                tts_embeddings=inputs_embeds,  # [pad][cond][text] embeddings (87 tokens, NO start_mel_token)
                tts_mel_embedding=self.inference_model.embeddings,  # mel_embedding layer
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,  # text_pos_embedding layer
            )
        else:
            output = self.inference_model.generate(
                inputs,
                bos_token_id=self.start_mel_token,
                pad_token_id=self.stop_mel_token,
                eos_token_id=self.stop_mel_token,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=logits_processor,
                num_return_sequences=num_return_sequences,
                **hf_generate_kwargs,  # pyright: ignore[reportAny]
            )
        logger.info("generation: %.4fs", time.perf_counter() - t4)
        logger.info("total inference_speech: %.4fs", time.perf_counter() - t0)

        assert isinstance(output, Tensor)
        return output[:, trunc_index:], speech_conditioning_latent

    def get_emovec(self, emo_speech_conditioning_latent: Tensor, emo_cond_lengths: Tensor) -> Tensor:
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1, 2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        return self.emo_layer(emo_vec_syn)

    def merge_emovec(
        self,
        speech_conditioning_latent: Tensor,
        emo_speech_conditioning_latent: Tensor,
        cond_lengths: Tensor,
        emo_cond_lengths: Tensor,
        alpha: float = 1.0,
    ) -> Tensor:
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        return base_vec + alpha * (emo_vec - base_vec)

    @patch_call(forward)
    def __call__(self) -> None: ...
