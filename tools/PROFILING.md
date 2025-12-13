# Profiling Tools for IndexTTS

This directory contains smart profiling tools that help you analyze performance without generating huge trace files. The tools filter output to show **your code** and **top-level library calls** while filtering out deep library internals.

## Quick Start

### 1. Profile CLI with Python cProfile (Recommended for CPU profiling)

```bash
# Basic profiling
python tools/profile_cli.py -v outputs/mizora.ogg "Hello world"

# Show top 50 functions
python tools/profile_cli.py -v outputs/mizora.ogg "test text" --top 50

# With acceleration options
python tools/profile_cli.py -v outputs/mizora.ogg "test" --use-accel --fp16

# Save full unfiltered profile for later analysis
python tools/profile_cli.py -v outputs/mizora.ogg "test" --save-full-profile full_profile.pstats
```

**View saved profiles:**
```bash
python -m pstats full_profile.pstats
# Then in pstats shell:
# > sort cumulative
# > stats 30
```

### 2. Profile Benchmark Runs

```bash
# Profile 5 benchmark runs
python tools/profile_benchmark.py -v outputs/mizora.ogg --runs 5

# Profile with torch.compile
python tools/profile_benchmark.py -v outputs/mizora.ogg --runs 3 --use-torch-compile --top 50

# Custom text and save full profile
python tools/profile_benchmark.py -v outputs/mizora.ogg --text "Long text to test" --save-full-profile bench_profile.pstats
```

### 3. CLI with Optional Torch Profiling (For GPU/CUDA profiling)

```bash
# Normal run (no profiling)
python cli_with_profiling.py -v outputs/mizora.ogg "test"

# With profiling (creates Chrome trace)
python cli_with_profiling.py -v outputs/mizora.ogg "test" --profile

# Custom trace output and more detailed summary
python cli_with_profiling.py -v outputs/mizora.ogg "test" --profile --profile-output my_trace.json --profile-rows 50

# With memory profiling (larger trace file)
python cli_with_profiling.py -v outputs/mizora.ogg "test" --profile --profile-memory
```

**View Chrome traces:**
- Open Chrome browser
- Go to `chrome://tracing`
- Click "Load" and select your `.json` trace file
- Or use https://ui.perfetto.dev/

### 4. Use Profiling Utils in Your Own Code

```python
from indextts.utils.profiling import create_filtered_profiler, print_filtered_table

# Method 1: Context manager
with create_filtered_profiler("my_trace.json") as prof:
    tts.infer(spk_audio_prompt="voice.wav", text="Hello", output_path="out.wav")
    prof.step()  # Trigger trace save

print_filtered_table(prof, sort_by="cuda_time_total", row_limit=30)

# Method 2: Quick function profiling
from indextts.utils.profiling import profile_function_calls

result = profile_function_calls(
    tts.infer,
    spk_audio_prompt="voice.wav",
    text="Hello world",
    output_path="out.wav",
    trace_path="my_profile.json",
    table_rows=50,
)
```

## What Gets Filtered?

### ✅ Included in Output
- All `indextts/` code (your project)
- Top-level library calls:
  - `torch.nn.*` operations
  - `transformers.models.*` model code
  - `transformers.generation.*` generation logic

### ❌ Excluded from Output (but still counted in totals)
- Deep library internals:
  - `torch._*` internal functions
  - `torch.autograd.*` autograd engine
  - `torch.cuda.*` CUDA internals
  - `transformers._*` internal helpers
- Python stdlib internals (typing, abc, contextlib, etc.)
- Built-in functions

**Note:** Excluded functions are still timed and included in parent function cumulative times. They just don't clutter your output.

## Profiling Modes Comparison

| Tool | Best For | Output Format | File Size | GPU Support |
|------|----------|---------------|-----------|-------------|
| `profile_cli.py` | Quick CPU profiling | Text table | Small | No |
| `profile_benchmark.py` | Multi-run analysis | Text table + benchmarks | Small | No |
| `cli_with_profiling.py` | GPU/CUDA analysis | Chrome trace + table | Medium-Large | Yes |
| `indextts/utils/profiling.py` | Custom scripts | Chrome trace + table | Medium-Large | Yes |

## Understanding the Output

### cProfile Output (profile_cli.py, profile_benchmark.py)

```
Top functions by CUMULATIVE time (includes time in called functions):
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       1    0.010    0.010   12.450   12.450 indextts/infer_v2.py:123(infer)
      10    0.025    0.003    8.230    0.823 indextts/gpt/model_v2.py:456(forward)
     100    0.150    0.002    5.100    0.051 torch/nn/modules/linear.py:103(forward)
```

- **ncalls**: Number of times function was called
- **tottime**: Total time spent in function itself (excluding subfunctions)
- **cumtime**: Cumulative time (including all subfunctions)
- **percall**: Average time per call

### Torch Profiler Output

```
--------------------------  ------------  ------------  ------------  ------------
Name                        Self CPU %    Self CPU      CPU total    CUDA total
--------------------------  ------------  ------------  ------------  ------------
aten::linear                2.5%         150.2ms       45.3%        2.34s
aten::addmm                 15.2%        890.1ms       38.1%        2.01s
```

- **Self CPU/CUDA**: Time spent in this operation alone
- **CPU/CUDA total**: Time including all child operations

## Tips

1. **Start with cProfile** (`profile_cli.py`) - it's lightweight and fast
2. **Use torch profiler** (`cli_with_profiling.py`) when you need GPU kernel details
3. **Adjust --top N** to see more or fewer functions (default: 30)
4. **Save full profiles** with `--save-full-profile` for later deep analysis
5. **For long runs**, torch profiler only records first few steps (see schedule in profiling.py)

## Customizing Filters

Edit the `should_include_function()` in the profiling scripts to adjust what you see:

```python
def should_include_function(filename: str, project_root: str) -> bool:
    # Add patterns you want to include
    include_patterns = [
        "torch/nn/",
        "transformers/models/",
        "my_other_module/",  # Add your patterns here
    ]
    
    # Add patterns to exclude
    exclude_patterns = [
        "site-packages/torch/_",
        "my_module/debug/",  # Exclude specific submodules
    ]
    
    # ... filtering logic ...
```

## Advanced: Analyzing Full Profiles

If you saved a full profile with `--save-full-profile`:

```bash
# Interactive analysis
python -m pstats profile.pstats

# In pstats shell:
> sort cumulative      # Sort by cumulative time
> stats 50             # Show top 50
> sort time            # Sort by internal time
> stats 20             # Show top 20
> callers infer        # See who calls 'infer'
> callees infer        # See what 'infer' calls
```

## Troubleshooting

**Problem:** Trace files too large
- **Solution:** Disable `--profile-memory` and reduce warmup/runs

**Problem:** Not seeing my code in output
- **Solution:** Check that your code is in `indextts/` directory or adjust filters

**Problem:** Want to see specific library
- **Solution:** Add pattern to `include_patterns` in profiling script

**Problem:** Chrome trace won't load
- **Solution:** Try https://ui.perfetto.dev/ (handles larger files better)

## Files

- `tools/profile_cli.py` - Profile single CLI runs with filtered cProfile
- `tools/profile_benchmark.py` - Profile benchmark runs with filtered cProfile  
- `cli_with_profiling.py` - CLI with optional torch profiler
- `indextts/utils/profiling.py` - Reusable profiling utilities

---

**Questions?** The profiling tools are designed to give you actionable insights without drowning you in library internals. Start with the simple tools and escalate to torch profiler when you need GPU details.
