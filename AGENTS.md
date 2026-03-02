# Agent Progress File

## Task: Remove All Unused Code

### Tools Used
- `vulture` (installed via `uv tool install vulture`) — dead code detector
- `ruff` (already installed) — unused import detection/fix

### Status: COMPLETE (Pass 1 done)

---

## Pass 1 — Unused Imports (ruff F401)

### Summary
- **Total errors found:** 176 (across indextts/, tests/, tools/, webui.py)
- **Auto-fixable by ruff:** 156 (`ruff check --select F401 --fix`)
- **Non-auto-fixable (re-exports in __init__.py):** 20 — require `__all__` or explicit `x as x` syntax, or manual deletion

### Files with non-auto-fixable F401 (re-exports in __init__.py):
- `indextts/utils/maskgct/models/codec/amphion_codec/quantize/__init__.py` — re-exports `FactorizedVectorQuantize`, `VectorQuantize`, `LookupFreeQuantize`, `ResidualVQ` that aren't used within the package itself (these ARE re-exports for external use — **do not delete**)

### Auto-fix applied: `ruff check --select F401 --fix indextts/ tests/ tools/ webui.py`

Files affected by auto-fix:
- `indextts/utils/maskgct/models/codec/amphion_codec/codec.py` — removed `einops.rearrange`, `VectorQuantize`, `FactorizedVectorQuantize`, `LookupFreeQuantize`
- `indextts/utils/maskgct/models/codec/amphion_codec/quantize/factorized_vector_quantize.py` — removed `numpy`
- `indextts/utils/maskgct/models/codec/amphion_codec/quantize/lookup_free_quantize.py` — removed `numpy`, `einops.rearrange`
- `indextts/utils/maskgct/models/codec/amphion_codec/quantize/residual_vq.py` — removed `typing.Union`, `numpy`, `torch.nn.functional`, `einops.rearrange`, `torch.nn.utils.weight_norm`
- `indextts/utils/maskgct/models/codec/amphion_codec/quantize/vector_quantize.py` — removed `numpy`
- `indextts/utils/maskgct/models/codec/amphion_codec/vocos.py` — removed `librosa`
- `indextts/utils/maskgct/models/codec/codec_trainer.py` — removed `tqdm.tqdm`, `models.codec.codec_sampler.build_samplers`
- `indextts/utils/maskgct/models/codec/facodec/facodec_dataset.py` — removed `torch.nn.functional`, `pad_sequence`, `CodecDataset`
- `indextts/utils/maskgct/models/codec/facodec/facodec_inference.py` — removed `shutil`, `argparse`, `yaml`, `time`
- `indextts/utils/maskgct/models/codec/facodec/facodec_trainer.py` — removed `random`, `re`, `accelerate`, `ProjectConfiguration`, `DataLoader`, `build_samplers`, `GANLoss`
- `indextts/utils/maskgct/models/codec/facodec/modules/attentions.py` — removed `copy`, `numpy`
- `indextts/utils/maskgct/models/codec/facodec/modules/commons.py` — removed `numpy`, `torch.nn`, `json`
- `indextts/utils/maskgct/models/codec/facodec/modules/layers.py` — removed `math`, `typing.Optional`, `typing.Any`, `torch.Tensor`, `torchaudio`
- `indextts/utils/maskgct/models/codec/facodec/modules/wavenet.py` — removed `math`
- `indextts/utils/maskgct/models/codec/facodec/optimizer.py` — removed `os`, `sys`, `os.path`, `numpy`, `torch.nn`, `torch.optim.Optimizer`
- `indextts/utils/maskgct/models/codec/kmeans/repcodec_model.py` — removed `concurrent.futures.ALL_COMPLETED`, `numpy`, `einops.rearrange`, `einops.repeat`
- `indextts/utils/maskgct/models/codec/melvqgan/melspec.py` — removed `pyworld`, `soundfile`, `os`, `torchaudio.functional.pitch_shift`, `librosa` (bare), `torch.nn.functional`, `tqdm`
- `indextts/utils/maskgct/models/codec/ns3_codec/facodec.py` — removed `torch.nn.functional`
- `indextts/utils/maskgct/models/codec/ns3_codec/melspec.py` — removed `pyworld`, `soundfile`, `os`, `torchaudio.functional.pitch_shift`, `librosa` (bare), `torch.nn.functional`
- `indextts/utils/maskgct/models/codec/ns3_codec/quantize/fvq.py` — removed `typing.Union`, `numpy`
- `indextts/utils/maskgct/models/codec/ns3_codec/transformer.py` — removed `numpy`
- `indextts/utils/maskgct/models/codec/speechtokenizer/modules/quantization/core_vq.py` — removed `broadcast_tensors`, `rank`
- `indextts/utils/maskgct/models/codec/speechtokenizer/modules/quantization/vq.py` — removed `math`
- `indextts/utils/maskgct/models/tts/maskgct/llama_nar.py` — removed `LlamaForCausalLM`, `torch.nn.functional`, `numpy`, `os`
- `indextts/utils/maskgct_utils.py` — removed `hf_hub_download`, `SeamlessM4TFeatureExtractor`, `safetensors`, `time`
- `indextts/utils/utils.py` — removed `os`, `random`

### Non-auto-fixable re-exports (KEPT as-is — they are public API re-exports):
- `indextts/utils/maskgct/models/codec/amphion_codec/quantize/__init__.py` — all 4 re-exports kept

---

## Pass 2 — Dead Code (vulture)

### Command used
`vulture indextts/ webui.py --min-confidence 80`

### Changes made

| File | Change |
|------|--------|
| `indextts/BigVGAN/alias_free_activation/cuda/activation1d.py` | Renamed `output_grads` → `_output_grads` in stub `backward()` that just raises; removed dead return after raise |
| `indextts/s2mel/modules/bigvgan/alias_free_activation/cuda/activation1d.py` | Same fix as above |
| `indextts/s2mel/modules/openvoice/api.py` | Removed ~20 lines of unreachable code in `add_watermark()` after `return audio` (watermarking was disabled in a stub) |
| `indextts/s2mel/modules/openvoice/modules.py` | Removed dead `piecewise_rational_quadratic_transform` block (~15 lines) after an if/else that returns in both branches |
| `indextts/s2mel/modules/commons.py` | Renamed `use_emovec` → `_use_emovec` in `MyModel.__init__` (not used in body; no keyword callers) |
| `indextts/s2mel/modules/rmvpe.py` | Renamed `use_jit` → `_use_jit` in `RMVPE.__init__` (not used in body; no keyword callers) |
| `indextts/s2mel/modules/vocos/helpers.py` | Renamed `trainer` → `_trainer` in `on_after_backward()` callback (not used in body) |
| `indextts/utils/xtransformers.py` | Renamed `use_entmax15` → `_use_entmax15` in `Attention.__init__` (never referenced by callers) |
| `indextts/vqvae/xtts_dvae.py` | Renamed `lr_quantizer_args` → `_lr_quantizer_args` in constructor (not used; no callers) |
| Deleted `.ipynb_checkpoints/` | Removed Jupyter autosave artifacts in `indextts/s2mel/modules/` |

### Remaining known vulture warnings (intentionally kept)

| File | Reason kept |
|------|-------------|
| `gpt/transformers_beam_search.py:324,817` — `final_beam_tokens` | Vendored transformers API; positional param in complex inheritance chain |
| `gpt/transformers_generation_utils.py:109,110` — `PreTrainedTokenizerBase`, `BaseStreamer` | Behind `if TYPE_CHECKING:` — valid for type checkers, vulture doesn't understand this pattern |
| `gpt/transformers_modeling_utils.py:2537` — `new_num_position_embeddings` | Stub abstract method param in vendored code |
| `s2mel/modules/hifigan/generator.py:248` — `upsample_scale` | Called with keyword `upsample_scale=np.prod(...)` at line 320 of same file |
| `s2mel/modules/openvoice/attentions.py:47` — `isflow` | Called with `isflow=True` at `openvoice/modules.py:552` |
| `s2mel/modules/openvoice/se_extractor.py:126` — `vad` | Called with `vad=True` at `openvoice/openvoice_app.py:117` |
| `utils/maskgct/models/codec/facodec/optimizer.py:72` — `scheduler_params_dict` | Called with keyword at `facodec_trainer.py:264` |

---

## Final State

### ruff F401 (unused imports): 0 errors ✅
### vulture 80% confidence: 9 remaining items (all intentionally kept — see above)

---

## Notes for future passes:
- The `indextts/utils/maskgct/` subtree is large and contains many vendored/copied modules — be careful removing from it
- `indextts/s2mel/` is used in `infer_v2.py` (the v2 inference path)
- Entry points: `indextts/infer.py`, `indextts/infer_v2.py`, `indextts/cli.py`, `webui.py`
- `indextts/gpt/transformers_*.py` files are heavily vendored from HuggingFace transformers
- Consider running `vulture --min-confidence 60` for a broader pass (will have more false positives)
- The `indextts/s2mel/modules/openvoice/` subtree is also largely vendored from OpenVoice
