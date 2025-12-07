import math
import sys
from pathlib import Path

# ensure workspace root is on sys.path so tests can import tools.benchmark
sys.path.insert(0, (Path(__file__).parent / "..").resolve())

from tools.benchmark import BenchmarkResult


def test_rtf_stats_basic():
    # 3 runs: times (s) / durations (s)
    times = [2.0, 3.0, 4.0]
    durations = [1.0, 2.0, 2.0]
    br = BenchmarkResult(
        startup_time=0.1,
        warmup_times=[],
        inference_times=times,
        warmup_durations=[],
        inference_durations=durations,
        text="hello",
        num_runs=3,
    )

    # expected rtf list: [2.0, 1.5, 2.0]
    assert len(br.rtf_list) == 3
    assert math.isclose(br.mean_rtf, sum(br.rtf_list) / 3.0, rel_tol=1e-9)
    assert math.isclose(br.median_rtf, 2.0, rel_tol=1e-9)
    assert math.isclose(br.min_rtf, min(br.rtf_list), rel_tol=1e-9)
    assert math.isclose(br.max_rtf, max(br.rtf_list), rel_tol=1e-9)


def test_rtf_handles_zero_duration():
    # first run has zero duration -> should be ignored
    times = [1.0, 2.0]
    durations = [0.0, 2.0]
    br = BenchmarkResult(
        startup_time=0.1,
        warmup_times=[],
        inference_times=times,
        warmup_durations=[],
        inference_durations=durations,
        text="hi",
        num_runs=2,
    )

    # Only second run contributes: rtf = 2.0 / 2.0 = 1.0
    assert br.rtf_list == [1.0]
    assert math.isclose(br.mean_rtf, 1.0, rel_tol=1e-9)
    assert math.isclose(br.median_rtf, 1.0, rel_tol=1e-9)


def test_duration_stats():
    times = [1.0, 1.5]
    durations = [0.5, 0.5]
    br = BenchmarkResult(
        startup_time=0.1,
        warmup_times=[],
        inference_times=times,
        warmup_durations=[],
        inference_durations=durations,
        text="dur",
        num_runs=2,
    )

    assert math.isclose(br.mean_duration, 0.5, rel_tol=1e-9)
    assert math.isclose(br.median_duration, 0.5, rel_tol=1e-9)
