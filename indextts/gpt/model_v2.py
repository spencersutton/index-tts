import functools
import importlib.util

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.generation.logits_process import TypicalLogitsWarper
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    lm_head: nn.Sequential
    text_pos_embedding: LearnedPositionEmbeddings
    transformer: GPT2Model
    kv_cache: bool
    cached_mel_emb: torch.Tensor | None
    device_map: dict[int, int] | None
    model_parallel: bool

    def __init__(self, config, gpt: GPT2Model, text_pos_emb, embeddings, norm, linear, kv_cache=False) -> None:
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

    def parallelize(self, device_map=None) -> None:
        self.device_map = (
            get_device_map(len(self.transformer.h), range(max(1, torch.cuda.device_count())))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self) -> None:
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")  # ty:ignore[invalid-argument-type]
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings) -> None:
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb) -> None:
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        token_type_ids = kwargs.get("token_type_ids")  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

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
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
            emb += self.text_pos_embedding.get_fixed_embedding(attention_mask.shape[1] - mel_len, attention_mask.device)
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
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)  # ty:ignore[invalid-argument-type]
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head[1].weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits, *transformer_outputs[1:])

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class ConditioningEncoder(nn.Module):
    def __init__(
        self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, do_checkpointing=False, mean=False
    ) -> None:
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        attn = [AttentionBlock(embedding_dim, num_attn_heads) for a in range(attn_blocks)]
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h


def build_hf_gpt_transformer(
    layers, model_dim, heads, max_mel_seq_len, max_text_seq_len
) -> tuple[GPT2Model, LearnedPositionEmbeddings, LearnedPositionEmbeddings, None, None]:
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model

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
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
        None,
        None,
    )


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2) -> None:
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
            nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]),
            nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 16, channels // 2),
            nn.ReLU(),
            nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]),
            nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 8, channels),
            nn.ReLU(),
            nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
        )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


class UnifiedVoice(nn.Module):
    gst_encoder: nn.Module
    inference_model: GPT2InferenceModel
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
        ) = build_hf_gpt_transformer(
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

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False) -> None:
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
                raise ImportError(
                    "flash_attn is required for acceleration but not installed. Please install from https://github.com/Dao-AILab/flash-attention/releases/"
                )

            from indextts.accel import AccelInferenceEngine, GPT2AccelModel

            # Create accel model
            accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)

            accel_gpt = accel_gpt.half().cuda() if half else accel_gpt.cuda()  # ty:ignore[missing-argument]
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
            import deepspeed  # ty:ignore[unresolved-import]

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model, mp_size=1, replace_with_kernel_inject=True, dtype=torch.float16
            )
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed  # ty:ignore[unresolved-import]

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model, mp_size=1, replace_with_kernel_inject=True, dtype=torch.float32
            )
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
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
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
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
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(
        self,
        speech_conditioning_inputs: torch.Tensor,
        first_inputs: torch.Tensor,
        first_head: nn.Module,
        second_inputs: torch.Tensor | None = None,
        second_head: nn.Module | None = None,
        get_attns: bool = False,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out: BaseModelOutputWithPastAndCrossAttentions = self.gpt(
            inputs_embeds=emb, return_dict=True, output_attentions=get_attns
        )
        if get_attns:
            assert gpt_out.attentions is not None
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        assert gpt_out.last_hidden_state is not None
        enc = gpt_out.last_hidden_state[:, offset:]
        enc: torch.Tensor = self.final_norm(enc)

        if return_latent:
            assert second_inputs is not None
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits: torch.Tensor = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            assert second_head is not None
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits: torch.Tensor = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(
        self, speech_conditioning_input: torch.Tensor, cond_mel_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        speech_conditioning_input, mask = self.conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )  # (b, s, d), (b, 1, s)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        return conds

    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)

    def forward(
        self,
        speech_conditioning_latent: torch.Tensor,
        text_inputs: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_codes: torch.Tensor,
        mel_codes_lengths: torch.Tensor,
        emo_speech_conditioning_latent: torch.Tensor,
        cond_mel_lengths: torch.Tensor | None = None,
        emo_cond_mel_lengths: torch.Tensor | None = None,
        emo_vec: torch.Tensor | None = None,
        use_speed: torch.Tensor | None = None,
        do_spk_cond: bool = False,
    ) -> torch.Tensor:
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

        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(
                speech_conditioning_latent.transpose(1, 2), cond_mel_lengths
            )
        else:
            speech_conditioning_latent = speech_conditioning_latent

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(
                emo_speech_conditioning_latent.transpose(1, 2), emo_cond_mel_lengths
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
        assert isinstance(duration_emb, torch.Tensor)
        assert isinstance(duration_emb_half, torch.Tensor)
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
            conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=False, return_latent=True
        )
        return mel_logits[
            :, :-2
        ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]
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
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(
        self,
        speech_condition: torch.Tensor,
        text_inputs: torch.Tensor,
        emo_speech_condition: torch.Tensor | None = None,
        cond_lengths: torch.Tensor | None = None,
        emo_cond_lengths: torch.Tensor | None = None,
        emo_vec: torch.Tensor | None = None,
        input_tokens: torch.Tensor | None = None,
        num_return_sequences: int = 1,
        max_generate_length: int | None = None,
        typical_sampling: bool = False,
        typical_mass: float = 0.9,
        **hf_generate_kwargs,
    ):
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
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device)

        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1, 2), cond_lengths)
        if emo_vec is None:
            print("compute emo vec")
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1, 2), emo_cond_lengths)
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
        self.inference_model.store_mel_emb(inputs_embeds)
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
                raise ValueError(f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
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
                temperature=hf_generate_kwargs.get("temperature", 1),
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
                **hf_generate_kwargs,
            )
        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:], speech_conditioning_latent
        # GenerateOutput
        output.sequences = output.sequences[:, trunc_index:]
        return output, speech_conditioning_latent

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1, 2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(
        self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha=1.0
    ):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
