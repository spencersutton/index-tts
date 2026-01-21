from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model, LogitsProcessorList

from indextts.config import UnifiedVoiceConfig
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.inference import GPT2InferenceModel
from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings
from indextts.gpt.perceiver import PerceiverResampler
from indextts.util import patch_call, unwrap

if TYPE_CHECKING:
    from indextts.accel import AccelInferenceEngine

DIM = 1280


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
    max_mel_tokens: int
    max_text_tokens: int

    cond_mask_pad: nn.ConstantPad1d
    conditioning_encoder: ConformerEncoder
    emo_cond_mask_pad: nn.ConstantPad1d
    emo_conditioning_encoder: ConformerEncoder
    emo_perceiver_encoder: PerceiverResampler
    perceiver_encoder: PerceiverResampler

    cfg: UnifiedVoiceConfig

    def __init__(self, cfg: UnifiedVoiceConfig, condition_num_latent: int = 32, use_accel: bool = False) -> None:
        """
        Args:
            layers: Number of layers in transformer stack.
            heads: Number of transformer heads. Must be divisible by DIM. Recommend DIM//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
        """
        super().__init__()
        self.cfg = cfg
        self.layers = cfg.layers
        self.heads = cfg.heads
        self.max_mel_tokens = cfg.max_mel_tokens
        self.max_text_tokens = cfg.max_text_tokens
        self.cond_mask_pad = nn.ConstantPad1d((condition_num_latent, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.conditioning_encoder = ConformerEncoder(linear_units=2048, attention_heads=8, num_blocks=6)
        self.perceiver_encoder = PerceiverResampler(DIM, heads=8, num_latents=condition_num_latent)

        self.emo_conditioning_encoder = ConformerEncoder(linear_units=1024, attention_heads=4, num_blocks=4)
        self.emo_perceiver_encoder = PerceiverResampler(1024, heads=4, num_latents=1)

        self.emo_layer = nn.Linear(DIM, DIM)
        self.emovec_layer = nn.Linear(1024, DIM)

        self.text_embedding = nn.Embedding(cfg.number_text_tokens + 1, DIM)
        self.mel_embedding = nn.Embedding(cfg.number_mel_codes, DIM)
        max_mel_seq_len = self.max_mel_tokens + 3
        max_text_seq_len = self.max_text_tokens + 2

        self.gpt = GPT2Model(
            GPT2Config(
                vocab_size=256,  # Unused.
                n_positions=max_mel_seq_len + max_text_seq_len,
                n_ctx=max_mel_seq_len + max_text_seq_len,
                n_embd=DIM,
                n_layer=cfg.layers,
                n_head=cfg.heads,
            )
        )
        # Override the built in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = lambda x: torch.zeros((x.shape[0], x.shape[1], DIM), device=x.device)
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_seq_len)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_seq_len)

        self.final_norm = nn.LayerNorm(DIM)
        self.text_head = nn.Linear(DIM, cfg.number_text_tokens + 1)
        self.mel_head = nn.Linear(DIM, cfg.number_mel_codes)

        self.speed_emb = nn.Embedding(2, DIM)
        self.speed_emb.weight.data.normal_(std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings: list[nn.Embedding] = [self.text_embedding, self.mel_embedding]
        for module in embeddings:
            module.weight.data.normal_(std=0.02)

        self.use_accel = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config

    def post_init_gpt2_config(self, use_deepspeed: bool, half: bool) -> None:
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.cfg.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=DIM,
            n_layer=self.layers,
            n_head=self.heads,
        )

        if self.use_accel and torch.cuda.is_available():
            # Check if flash attention is available
            try:
                import flash_attn  # noqa: F401 # type: ignore
            except ImportError as err:
                raise ImportError(
                    "flash_attn is required for acceleration but not installed. Please install from https://github.com/Dao-AILab/flash-attention/releases/"
                ) from err

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
                head_dim=DIM // self.heads,
                block_size=256,
                num_blocks=16,  # Reduce to save memory (16*256 = 4096 tokens capacity)
                use_cuda_graph=True,
            )
            print("acceleration engine initialized")
        self.inference_model = GPT2InferenceModel(
            gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head
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

    def set_padding(self, input_tokens: Tensor, lengths: Tensor, token: int) -> Tensor:
        """
        Given tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with `token` in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = lengths[b]
            if actual_end < input_tokens.shape[-1]:
                input_tokens[b, actual_end:] = token
        return input_tokens

    def forward(
        self,
        speech_conditioning_latent: Tensor,
        text_inputs: Tensor,
        mel_codes: Tensor,
        emo_speech_conditioning_latent: Tensor,
        emo_vec: Tensor,
        use_speed: int,
        device: torch.types.Device,
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

        text_lengths = torch.tensor([text_inputs.shape[-1]], device=device)
        text_inputs = self.set_padding(text_inputs, text_lengths, self.cfg.stop_text_token)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.cfg.stop_text_token)

        mel_codes_lengths = torch.tensor([mel_codes.shape[-1]], device=device)
        mel_codes = self.set_padding(mel_codes, mel_codes_lengths, self.cfg.stop_mel_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.cfg.stop_mel_token)

        text_inputs = F.pad(text_inputs, (1, 0), value=self.cfg.start_text_token)
        mel_codes = F.pad(mel_codes, (1, 0), value=self.cfg.start_mel_token)

        mel_emb = self.mel_embedding(mel_codes) + self.mel_pos_embedding(mel_codes)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        conds = self.combine_latents(speech_conditioning_latent, emo_vec, text_inputs)
        output = self.gpt(
            inputs_embeds=torch.cat([conds, text_emb, mel_emb], dim=1), return_dict=True, output_attentions=False
        )

        offset = conds.shape[1]
        enc = unwrap(output.last_hidden_state)[:, offset:]
        enc = self.final_norm(enc)

        # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.
        return enc[:, -mel_emb.shape[1] : -2]

    def prepare_gpt_inputs(self, latent: Tensor, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        b = inputs.size(0)
        L = inputs.size(1)
        device = inputs.device
        single_cond = latent.ndim == 3 and latent.shape[0] == 1
        if not single_cond:
            assert latent.shape[0] == b, f"batch size mismatch: {latent.shape[0]} vs {b}"
        batched_mel_emb: list[Tensor] = []
        attention_masks: list[Tensor] = []
        target_len = latent.shape[1] + L + 2
        for i in range(inputs.size(0)):
            valid_mask = (inputs[i] != self.cfg.stop_text_token) & (inputs[i] != self.cfg.start_text_token)

            text_input = inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.cfg.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.cfg.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)

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
        # [b, s, dim]
        batched_mel_emb_tensor = torch.stack(batched_mel_emb)
        # [b, s + 1]
        attention_mask = torch.stack(attention_masks)
        # [b, s + 1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb_tensor.shape[0],
                batched_mel_emb_tensor.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.cfg.start_mel_token
        return fake_inputs, batched_mel_emb_tensor, attention_mask

    def combine_latents(self, speech_conditioning_latent: Tensor, emo_vec: Tensor, text_inputs: Tensor) -> Tensor:
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
        speech_conditioning_latent: Tensor,
        text_inputs: Tensor,
        emo_speech_condition: Tensor,
        *,
        emo_vec: Tensor,
        input_tokens: None = None,
        num_return_sequences: int = 1,
        max_generate_length: int | None = None,
        **hf_generate_kwargs: Any,
    ) -> Tensor:
        """
        Args:
            speech_condition: (b, d, frames) or (d, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
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
        if self.accel_engine is not None and num_return_sequences == 1:
            output = self.accel_engine.generate(
                inputs_ids,  # fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=float(hf_generate_kwargs.get("temperature", 1)),
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
                logits_processor=LogitsProcessorList(),
                num_return_sequences=num_return_sequences,
                **hf_generate_kwargs,
            )
        return output[:, trunc_index:]

    def process_speech_condition(self, condition: Tensor) -> Tensor:
        if condition.ndim == 2:
            condition = condition.unsqueeze(0)

        input, mask = self.conditioning_encoder(condition)
        mask = self.cond_mask_pad(mask.squeeze(1))
        return self.perceiver_encoder(input, mask)

    def get_emo_vec(self, latent: Tensor) -> Tensor:
        input, mask = self.emo_conditioning_encoder(latent)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(input, conds_mask)
        emotion_vector = self.emovec_layer(conds.squeeze(1))
        return self.emo_layer(emotion_vector)

    @patch_call(forward)
    def __call__(self) -> None: ...
