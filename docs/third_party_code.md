# Third-Party Code Inventory — IndexTTS / IndexTTS2

This document catalogues all third-party code vendored inside the `indextts/`
package, classifies whether it is novel, vendored, or a paper implementation,
and − where possible − identifies the upstream source and any PyPI package or
Git submodule that could replace it.

---

## Classification Key

| Category | Meaning |
|---|---|
| **Novel** | Written specifically for this repository; no known upstream. |
| **Vendored** | A copy (possibly lightly modified) of someone else's library. |
| **Paper impl.** | Code that implements a published research paper, originally from an external research codebase. |

---

## 1. IndexTTS Core (Novel)

These files are unique to this project and constitute the novel contribution of
the Bilibili IndexTTS team.

| Path | Description |
|---|---|
| `indextts/infer.py` | `IndexTTS` inference class (IndexTTS 1.x). |
| `indextts/infer_v2.py` | `IndexTTS2` inference class with emotion and duration control. |
| `indextts/cli.py` | Command-line entry point. |
| `indextts/gpt/model.py` | `UnifiedVoice` — GPT-style autoregressive TTS model (IndexTTS 1.x). |
| `indextts/gpt/model_v2.py` | `UnifiedVoice` for IndexTTS 2 with flash-attention acceleration path. |
| `indextts/utils/front.py` | `TextNormalizer` / `TextTokenizer` — CJK+English text normalization, pinyin injection, SentencePiece tokenization. |
| `indextts/utils/feature_extractors.py` | `MelSpectrogramFeatures` — custom mel-spectrogram extractor. |
| `indextts/utils/text_utils.py` | Syllable counting mixing Chinese character count and `textstat`. |
| `indextts/utils/maskgct_utils.py` | Helper to build semantic codec and semantic model from Amphion MaskGCT. |
| `indextts/utils/common.py` | General audio loading and CJK tokenization utilities. |
| `indextts/utils/webui_utils.py` | Gradio WebUI helper functions. |
| `indextts/utils/arch_util.py` | `AttentionBlock` and `GroupNorm32` glue. |
| `indextts/utils/utils.py` | Miscellaneous utilities. |
| `indextts/s2mel/modules/commons.py` | Shared commons (sequence masks, `MyModel`, distributed checkpoint loader). |
| `indextts/s2mel/modules/diffusion_transformer.py` | `DiT` — diffusion transformer wiring `gpt_fast` `Transformer` with `WaveNet`. |
| `indextts/s2mel/modules/flow_matching.py` | `BASECFM` / Conditional Flow Matching wrapper around `DiT`. |
| `indextts/s2mel/modules/length_regulator.py` | Length regulator with F0 coarsening. |
| `indextts/s2mel/modules/layers.py` | Auxiliary layer utilities (activations, ISTFT, etc.). |
| `indextts/s2mel/modules/audio.py` | Mel-spectrogram computation wrappers. |
| `indextts/s2mel/modules/quantize.py` | VQ-based quantizer glue over DAC/encodec primitives. |
| `indextts/s2mel/modules/wavenet.py` | `WN` — WaveNet conditioner built on top of `encodec.py`'s `SConv1d`. |
| `indextts/s2mel/optimizers.py` | `MultiOptimizer` trainer helper. |
| `indextts/s2mel/hf_utils.py` | HuggingFace Hub download helper. |
| `indextts/s2mel/wav2vecbert_extract.py` | Wav2Vec-BERT semantic feature extractor (dev/training utility). |
| `webui.py` | Gradio WebUI entrypoint. |
| `tools/gpu_check.py` | GPU diagnostics helper. |
| `tools/i18n/` | Internationalisation scan/loader. |

---

## 2. Acceleration Layer (Novel)

`indextts/accel/` — custom paged KV-cache inference engine for GPT inference
acceleration, using Triton + FlashAttention.  All files appear to be novel
IndexTTS code (no upstream attribution notices).

| Path | Description |
|---|---|
| `indextts/accel/accel_engine.py` | Batched prefill/decode engine with paged KV cache. |
| `indextts/accel/attention.py` | Triton-backed attention with variable-length sequences (`flash_attn`). |
| `indextts/accel/gpt2_accel.py` | Drop-in replacement GPT2 attention block using the paged engine. |
| `indextts/accel/kv_manager.py` | Block-level KV cache memory manager with prefix hashing. |

---

## 3. Vendored: BigVGAN (NVIDIA)

**Source:** https://github.com/NVIDIA/BigVGAN  
**License:** MIT  
**Paper:** *BigVGAN: A Universal Neural Vocoder with Large-Scale Training*
(Lee et al., 2022 / 2024)  
**PyPI replacement:** No official `bigvgan` PyPI package exists (as of early 2026). NVIDIA does publish the model on HuggingFace Hub, but the library must be installed from source.  
**Submodule candidate:** `https://github.com/NVIDIA/BigVGAN` could be added as a Git submodule.

Two separate copies of BigVGAN are vendored:

### 3a. `indextts/BigVGAN/`

Used by `IndexTTS` (v1). Modified to integrate the ECAPA-TDNN speaker-embedding
path.

| Path | Notes |
|---|---|
| `bigvgan.py` | Main generator; copyright NVIDIA 2024; adapted from jik876/hifi-gan. |
| `models.py` | Discriminators; copyright NVIDIA 2022. |
| `activations.py` | Snake / SnakeBeta activations (adapted from EdwardDixon/snake). |
| `utils.py` | Weight-norm helpers (adapted from jik876/hifi-gan). |
| `alias_free_activation/torch/` | Alias-free resampling (adapted from junjun3518/alias-free-torch, Apache-2.0). |
| `alias_free_activation/cuda/` | CUDA kernel for alias-free activation; copyright NVIDIA 2024. |
| `alias_free_torch/` | Second copy of alias-free-torch for legacy compatibility. |
| `ECAPA_TDNN.py` | Speaker encoder (see §5). |
| `nnet/` | SpeechBrain primitives (see §6). |

### 3b. `indextts/s2mel/modules/bigvgan/`

Used by the IndexTTS2 `s2mel` pipeline. Same upstream source as §3a.

---

## 4. Vendored: DAC — Descript Audio Codec

**Source:** https://github.com/descriptinc/descript-audio-codec  
**License:** MIT  
**Paper:** *High-Fidelity Audio Compression with Improved RVQGAN* (Kumar et al., 2023)  
**PyPI replacement:** `descript-audio-codec` — **exists on PyPI** but is not listed directly in `pyproject.toml`; `descript-audiotools` is listed and installed. The entire `indextts/s2mel/dac/` subtree is a vendored copy of the DAC library.  
**Submodule candidate:** `https://github.com/descriptinc/descript-audio-codec`

| Path | Notes |
|---|---|
| `indextts/s2mel/dac/__init__.py` | Package root; sets `audiotools` INTERN/EXTERN lists. |
| `indextts/s2mel/dac/model/dac.py` | DAC model class. |
| `indextts/s2mel/dac/model/encodec.py` | Meta/Facebook Encodec convolutional wrappers (see §8). |
| `indextts/s2mel/dac/model/discriminator.py` | Multi-band discriminator. |
| `indextts/s2mel/dac/nn/layers.py` | `Snake1d`, `WNConv1d`, `WNConvTranspose1d`. |
| `indextts/s2mel/dac/nn/quantize.py` | `ResidualVectorQuantize`, `VectorQuantize`. |
| `indextts/s2mel/dac/utils/` | Model download and encode/decode utilities. |

---

## 5. Vendored: ECAPA-TDNN (SpeechBrain)

**Source:** https://github.com/speechbrain/speechbrain  
**License:** Apache-2.0  
**Paper:** *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification* (Desplanques et al., 2020)  
**PyPI replacement:** `speechbrain` — **exists on PyPI** (`pip install speechbrain`). The `ECAPA_TDNN` class and its `nnet/` helpers could be imported directly.  
**Submodule candidate:** https://github.com/speechbrain/speechbrain

| Path | Notes |
|---|---|
| `indextts/BigVGAN/ECAPA_TDNN.py` | Speaker encoder; authors "Hwidong Na 2020". |
| `indextts/BigVGAN/nnet/CNN.py` | SpeechBrain Conv layer (authors Ravanelli, Zhong, et al.). |
| `indextts/BigVGAN/nnet/linear.py` | SpeechBrain linear layer. |
| `indextts/BigVGAN/nnet/normalization.py` | SpeechBrain BatchNorm1d. |

---

## 6. Vendored: Conformer Encoder (WeNet / ESPnet)

**Source:** https://github.com/wenet-e2e/wenet (originally from https://github.com/espnet/espnet)  
**License:** Apache-2.0  
**PyPI replacement:** `wenet` — available on PyPI. `ESPnet` also on PyPI (`espnet`). The conformer building blocks can be imported from either.  
**Submodule candidate:** https://github.com/wenet-e2e/wenet

| Path | Notes |
|---|---|
| `indextts/gpt/conformer/attention.py` | Copyright Shigeki Karita 2019, Mobvoi Inc / Binbin Zhang 2020. |
| `indextts/gpt/conformer/embedding.py` | Copyright Mobvoi Inc 2020; modified from ESPnet. |
| `indextts/gpt/conformer/subsampling.py` | Copyright Mobvoi Inc 2021; modified from ESPnet. |
| `indextts/gpt/conformer_encoder.py` | `ConformerEncoder` — uses the conformer submodules above. Appears novel. |
| `indextts/utils/checkpoint.py` | Checkpoint load/save utility; copyright Mobvoi / Binbin Zhang 2020. |

---

## 7. Vendored: HuggingFace Transformers (GPT-2 + generation)

**Source:** https://github.com/huggingface/transformers  
**License:** Apache-2.0  
**PyPI replacement:** `transformers` — **already listed in `pyproject.toml`** and installed. These files are copies of older Transformers internal modules, likely frozen to preserve a specific generation API.  

> **Note:** These four files total ~13 000 lines. They shadow the installed
> `transformers` package and carry drift risk. A future cleanup could remove
> them in favour of importing directly from `transformers`.

| Path | Notes |
|---|---|
| `indextts/gpt/transformers_gpt2.py` | Copyright OpenAI / HuggingFace 2018; GPT-2 model. |
| `indextts/gpt/transformers_generation_utils.py` | Copyright Google / Facebook / HuggingFace 2020; generation logic. |
| `indextts/gpt/transformers_beam_search.py` | Copyright HuggingFace 2020; beam-search scorers. |
| `indextts/gpt/transformers_modeling_utils.py` | Copyright Google / Facebook / HuggingFace 2018; base `PreTrainedModel`. |

---

## 8. Vendored: Meta Encodec Convolutional Wrappers

**Source:** https://github.com/facebookresearch/encodec  
**License:** MIT  
**Paper:** *High Fidelity Neural Audio Compression* (Défossez et al., 2022)  
**PyPI replacement:** `encodec` — **exists on PyPI** (`pip install encodec`). The `SConv1d`/`SConvTranspose1d`/`SLSTM` helpers could be imported from the package.

| Path | Notes |
|---|---|
| `indextts/s2mel/modules/encodec.py` | Copyright Meta Platforms; streaming conv wrappers. |
| `indextts/s2mel/dac/model/encodec.py` | Same file vendored again inside the DAC subtree. |

---

## 9. Vendored: gpt-fast (Meta)

**Source:** https://github.com/pytorch-labs/gpt-fast  
**License:** BSD-3-Clause  
**PyPI replacement:** No PyPI package; small research repo. Code is ~200 lines, so vendoring is reasonable.  
**Submodule candidate:** https://github.com/pytorch-labs/gpt-fast (though the repo is largely a reference; the files used here are self-contained).

| Path | Notes |
|---|---|
| `indextts/s2mel/modules/gpt_fast/model.py` | `Transformer` / `TransformerBlock` / `Attention` + `KVCache`; copyright Meta Platforms. |
| `indextts/s2mel/modules/gpt_fast/generate.py` | Speculative decoding + compiled sampling; copyright Meta Platforms. |
| `indextts/s2mel/modules/gpt_fast/quantize.py` | Int4/Int8 weight-only quantization; copyright Meta Platforms. |

---

## 10. Vendored: alias-free-torch (junjun3518)

**Source:** https://github.com/junjun3518/alias-free-torch  
**License:** Apache-2.0  
**Paper:** *Alias-Free Generative Adversarial Networks* (Karras et al., 2021)  
**PyPI replacement:** `alias-free-torch` — **exists on PyPI** (`pip install alias-free-torch`).

This four-file module (`act.py`, `filter.py`, `resample.py`, `__init__.py`) is
duplicated **five times** across the repository:

| Location |
|---|
| `indextts/BigVGAN/alias_free_activation/torch/` |
| `indextts/BigVGAN/alias_free_torch/` |
| `indextts/s2mel/modules/alias_free_torch/` |
| `indextts/s2mel/modules/bigvgan/alias_free_activation/torch/` |
| `indextts/utils/maskgct/models/codec/facodec/alias_free_torch/` |
| `indextts/utils/maskgct/models/codec/ns3_codec/alias_free_torch/` |

---

## 11. Vendored: OpenVoice

**Source:** https://github.com/myshell-ai/OpenVoice  
**License:** MIT  
**Paper:** *OpenVoice: Versatile Instant Voice Cloning* (Qin et al., 2023)  
**PyPI replacement:** No official `openvoice` package on PyPI. A fork named `openvoice` exists on PyPI but is not the official source.  
**Submodule candidate:** https://github.com/myshell-ai/OpenVoice

| Path | Notes |
|---|---|
| `indextts/s2mel/modules/openvoice/` | Full copy: `api.py`, `models.py`, `attentions.py`, `modules.py`, `commons.py`, `transforms.py`, `mel_processing.py`, `se_extractor.py`, `utils.py`, plus an `openvoice_app.py` and checkpoint stubs. Used for the tone-colour (timbre) extractor (`ToneColorConverter`). |

---

## 12. Vendored: Vocos Vocoder

**Source:** https://github.com/hubert-siuzdak/vocos  
**License:** MIT  
**Paper:** *Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders* (Siuzdak, 2023)  
**PyPI replacement:** `vocos` — **exists on PyPI** (`pip install vocos`).

| Path | Notes |
|---|---|
| `indextts/s2mel/modules/vocos/` | Full copy: `models.py`, `modules.py`, `heads.py`, `loss.py`, `pretrained.py`, `spectral_ops.py`, `helpers.py`. |

---

## 13. Vendored: CAMPPlus Speaker Encoder (Alibaba 3D-Speaker)

**Source:** https://github.com/alibaba-damo-academy/3D-Speaker  
**License:** Apache-2.0  
**Paper:** *CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking* (Wang et al., 2023)  
**PyPI replacement:** `3dspeaker` — available on PyPI via `pip install 3dspeaker`, though the package structure may differ.  
**Submodule candidate:** https://github.com/alibaba-damo-academy/3D-Speaker

| Path | Notes |
|---|---|
| `indextts/s2mel/modules/campplus/DTDNN.py` | `CAMPPlus` model class. |
| `indextts/s2mel/modules/campplus/classifier.py` | Speaker classifier head. |
| `indextts/s2mel/modules/campplus/layers.py` | TDNN / DenseLayer / StatsPool building blocks. |

---

## 14. Vendored: HiFi-GAN from CosyVoice (Alibaba)

**Source:** https://github.com/FunAudioLLM/CosyVoice  
**License:** Apache-2.0  
**Paper:** *CosyVoice: A Scalable Multilingual Zero-Shot Text-to-Speech Synthesizer Using Supervised Semantic Tokens* (Du et al., 2024)  
**PyPI replacement:** No `cosyvoice` PyPI package.  
**Submodule candidate:** https://github.com/FunAudioLLM/CosyVoice

| Path | Notes |
|---|---|
| `indextts/s2mel/modules/hifigan/generator.py` | HiFi-GAN vocoder; copyright Alibaba 2024 (Xiang Lyu, Kai Hu). |
| `indextts/s2mel/modules/hifigan/f0_predictor.py` | `ConvRNNF0Predictor`; copyright Alibaba 2024. |

---

## 15. Vendored: Amphion MaskGCT Codec Stack

**Source:** https://github.com/open-mmlab/Amphion  
**License:** MIT  
**Paper:** *MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer* (Wang et al., 2024)  
**PyPI replacement:** No `amphion` PyPI package.  
**Submodule candidate:** https://github.com/open-mmlab/Amphion (large monorepo; a targeted subdirectory copy/submodule is more feasible)

The entire `indextts/utils/maskgct/` tree is ported from Amphion (47 Python
files carry `Copyright (c) 2023/2024 Amphion`). It provides the semantic codec
and MaskGCT S2A model used by IndexTTS2 for emotion control.

| Sub-path | Content |
|---|---|
| `models/tts/maskgct/maskgct_s2a.py` | `MaskGCT_S2A` — semantic-to-acoustic model. |
| `models/tts/maskgct/llama_nar.py` | `DiffLlama` — non-autoregressive LLaMA-based diffusion decoder. |
| `models/codec/amphion_codec/` | `CodecEncoder` / `CodecDecoder` + RVQ / LFQ / FVQ quantizers. |
| `models/codec/kmeans/repcodec_model.py` | `RepCodec` — K-means semantic codec. |
| `models/codec/kmeans/vocos.py` | Vocos backbone inside RepCodec (Amphion version). |
| `models/codec/speechtokenizer/` | SpeechTokenizer (modified from ZhangXInFD/SpeechTokenizer; core RVQ layers copied from Meta Encodec). |
| `models/codec/facodec/` | FACodec for NaturalSpeech 3 (modified from sh-lee-prml/HierSpeechpp + yl4579/PitchExtractor). |
| `models/codec/ns3_codec/` | FACodec variant for NS3 (also alias-free-torch copy). |
| `models/codec/vevo/vevo_repcodec.py` | VQ from ByteDance AudioDec (CC BY-NC). |

---

## 16. Vendored: DiscreteVAE / DVAE (from Tortoise-TTS / DALLE-pytorch)

**Source:** https://github.com/lucidrains/DALLE-pytorch (lucidrains), re-used in
[neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts) and then
[coqui-ai/TTS (XTTS)](https://github.com/coqui-ai/TTS)  
**License:** MIT  
**PyPI replacement:** `tortoise-tts` exists on PyPI but ships the whole TTS pipeline. The DVAE alone is best sourced directly.  
**Note:** This file is currently commented out in `infer.py` and appears to be a legacy artefact from IndexTTS 1.x development.

| Path | Notes |
|---|---|
| `indextts/vqvae/xtts_dvae.py` | `DiscreteVAE`, `Quantize`, `DiscretizationLoss`. File comment: *"partially derived from lucidrains DALLE implementation"*. |

---

## 17. Vendored: PerceiverResampler (lucidrains / NaturalSpeech2)

**Source:** https://github.com/lucidrains/naturalspeech2-pytorch  
**License:** MIT  
**PyPI replacement:** `naturalspeech2-pytorch` exists on PyPI.

| Path | Notes |
|---|---|
| `indextts/gpt/perceiver.py` | `PerceiverResampler`; adapted from lucidrains/naturalspeech2-pytorch. Used to downsample reference audio tokens. |

---

## 18. Vendored: x-transformers (partial)

**Source:** https://github.com/lucidrains/x-transformers  
**License:** MIT  
**PyPI replacement:** `x-transformers` — **exists on PyPI**.

| Path | Notes |
|---|---|
| `indextts/utils/xtransformers.py` | `RelativePositionBias` and associated utilities; appears to be a partial snapshot of x-transformers. |

---

## Summary Table

| Component | Type | Upstream | PyPI Package | Submodule? |
|---|---|---|---|---|
| `IndexTTS` core (`infer.py`, front-end, etc.) | Novel | — | — | — |
| `accel/` (paged KV cache engine) | Novel | — | — | — |
| **BigVGAN** (`BigVGAN/`, `s2mel/modules/bigvgan/`) | Vendored + Paper impl. | github.com/NVIDIA/BigVGAN | ❌ No PyPI | ✅ Submodule candidate |
| **DAC** (`s2mel/dac/`) | Vendored | github.com/descriptinc/descript-audio-codec | ✅ `descript-audio-codec` | ✅ |
| **ECAPA-TDNN** (`BigVGAN/ECAPA_TDNN.py` + `nnet/`) | Vendored | github.com/speechbrain/speechbrain | ✅ `speechbrain` | ✅ |
| **Conformer encoder** (`gpt/conformer/`) | Vendored | github.com/wenet-e2e/wenet | ✅ `wenet` / `espnet` | ✅ |
| **HuggingFace Transformers** (`gpt/transformers_*.py`) | Vendored | github.com/huggingface/transformers | ✅ `transformers` *(already a dep)* | — |
| **Meta Encodec wrappers** (`encodec.py` × 2) | Vendored | github.com/facebookresearch/encodec | ✅ `encodec` | ✅ |
| **gpt-fast** (`s2mel/modules/gpt_fast/`) | Vendored | github.com/pytorch-labs/gpt-fast | ❌ No PyPI | ✅ Submodule candidate |
| **alias-free-torch** (×6 copies) | Vendored | github.com/junjun3518/alias-free-torch | ✅ `alias-free-torch` | ✅ |
| **OpenVoice** (`s2mel/modules/openvoice/`) | Vendored | github.com/myshell-ai/OpenVoice | ❌ No official PyPI | ✅ Submodule candidate |
| **Vocos** (`s2mel/modules/vocos/`) | Vendored | github.com/hubert-siuzdak/vocos | ✅ `vocos` | ✅ |
| **CAMPPlus** (`s2mel/modules/campplus/`) | Vendored + Paper impl. | github.com/alibaba-damo-academy/3D-Speaker | ⚠️ `3dspeaker` (unofficial) | ✅ |
| **HiFi-GAN (CosyVoice)** (`s2mel/modules/hifigan/`) | Vendored | github.com/FunAudioLLM/CosyVoice | ❌ No PyPI | ✅ Submodule candidate |
| **Amphion MaskGCT** (`utils/maskgct/`) | Vendored + Paper impl. | github.com/open-mmlab/Amphion | ❌ No PyPI | ⚠️ Large monorepo |
| **DiscreteVAE / DVAE** (`vqvae/xtts_dvae.py`) | Vendored | github.com/lucidrains/DALLE-pytorch | ⚠️ `tortoise-tts` (full pipeline) | — |
| **PerceiverResampler** (`gpt/perceiver.py`) | Vendored | github.com/lucidrains/naturalspeech2-pytorch | ✅ `naturalspeech2-pytorch` | — |
| **x-transformers** (`utils/xtransformers.py`) | Vendored | github.com/lucidrains/x-transformers | ✅ `x-transformers` | — |

---

## Notable Issues

1. **alias-free-torch is duplicated six times.** Replacing all six occurrences
   with `pip install alias-free-torch` and a single import alias would
   significantly reduce the code surface.

2. **HuggingFace Transformers files are ~13 000 lines of frozen code.** The
   project already lists `transformers==4.52.1` as a hard dependency. These
   files likely exist to pin custom behaviour in the generation loop; they
   should be audited to determine whether the needed modifications can be
   expressed via the stock API (monkey-patching, custom
   `LogitsProcessor`, etc.).

3. **DAC (`descript-audio-codec`) is listed as a PyPI dep
   (`descript-audiotools`), but the library itself (`descript-audio-codec`) is
   vendored.** The vendored copy locks an exact API; it should be checked
   whether a pinned `descript-audio-codec` package import would work instead.

4. **BigVGAN appears twice** (`indextts/BigVGAN/` for IndexTTS 1.x and
   `indextts/s2mel/modules/bigvgan/` for IndexTTS 2). The two copies could be
   consolidated into one with appropriate configuration shims.

5. **`indextts/utils/maskgct/models/codec/vevo/vevo_repcodec.py`** carries a
   CC BY-NC licence (from ByteDance AudioDec), which is **not** a permissive
   open-source licence. This may restrict commercial use of IndexTTS2 beyond
   what the Bilibili model licence already stipulates.
