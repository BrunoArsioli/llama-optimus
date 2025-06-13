# tests/test_core.py
from core import run_optimization, estimate_max_ngl, run_llama_bench_with_csv
import pytest

def test_estimate_max_ngl_min():
    # Use a tiny model or mock llama-bench binary here
    max_ngl = estimate_max_ngl(min_ngl=0, max_ngl=1)
    assert isinstance(max_ngl, int)
    assert max_ngl >= 0

def test_invalid_llama_bench(tmp_path):
    # Simulate missing llama-bench
    # You could monkeypatch subprocess.run to raise error
    pass
