import torch
from torch import Tensor
from torch.nn import Module

class Wav2Vec2Model(Module):
    def __init__(
        self,
        feature_extractor: Module,
        encoder: Module,
        aux: Module | None = ...,
    ) -> None: ...
    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = ...,
        num_layers: int | None = ...,
    ) -> tuple[list[Tensor], Tensor | None]: ...
    def forward(self, waveforms: Tensor, lengths: Tensor | None = ...) -> tuple[Tensor, Tensor | None]: ...

class HuBERTPretrainModel(Module):
    def __init__(
        self,
        wav2vec2: Wav2Vec2Model,
        mask_generator: Module,
        logit_generator: Module,
        feature_grad_mult: float | None,
    ) -> None: ...
    def forward(
        self,
        waveforms: Tensor,
        labels: Tensor,
        audio_lengths: Tensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...

def wav2vec2_model(
    extractor_mode: str,
    extractor_conv_layer_config: list[tuple[int, int, int]] | None,
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: int | None,
) -> Wav2Vec2Model: ...
def wav2vec2_base(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def wav2vec2_large(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def wav2vec2_large_lv60k(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def hubert_base(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def hubert_large(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def hubert_xlarge(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def hubert_pretrain_model(
    extractor_mode: str,
    extractor_conv_layer_config: list[tuple[int, int, int]] | None,
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    mask_prob: float,
    mask_selection: str,
    mask_other: float,
    mask_length: int,
    no_mask_overlap: bool,
    mask_min_space: int,
    mask_channel_prob: float,
    mask_channel_selection: str,
    mask_channel_other: float,
    mask_channel_length: int,
    no_mask_channel_overlap: bool,
    mask_channel_min_space: int,
    skip_masked: bool,
    skip_nomask: bool,
    num_classes: int,
    final_dim: int,
    feature_grad_mult: float | None,
) -> HuBERTPretrainModel: ...
def hubert_pretrain_base(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    mask_prob: float = ...,
    mask_channel_prob: float = ...,
    mask_channel_length: int = ...,
    feature_grad_mult: float | None = ...,
    num_classes: int = ...,
) -> HuBERTPretrainModel: ...
def hubert_pretrain_large(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    mask_prob: float = ...,
    mask_channel_prob: float = ...,
    mask_channel_length: int = ...,
    feature_grad_mult: float | None = ...,
) -> HuBERTPretrainModel: ...
def hubert_pretrain_xlarge(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    mask_prob: float = ...,
    mask_channel_prob: float = ...,
    mask_channel_length: int = ...,
    feature_grad_mult: float | None = ...,
) -> HuBERTPretrainModel: ...
def wavlm_model(
    extractor_mode: str,
    extractor_conv_layer_config: list[tuple[int, int, int]] | None,
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_num_buckets: int,
    encoder_max_distance: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: int | None,
) -> Wav2Vec2Model: ...
def wavlm_base(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def wavlm_large(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def wav2vec2_xlsr_300m(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def wav2vec2_xlsr_1b(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
def wav2vec2_xlsr_2b(
    encoder_projection_dropout: float = ...,
    encoder_attention_dropout: float = ...,
    encoder_ff_interm_dropout: float = ...,
    encoder_dropout: float = ...,
    encoder_layer_drop: float = ...,
    aux_num_out: int | None = ...,
) -> Wav2Vec2Model: ...
