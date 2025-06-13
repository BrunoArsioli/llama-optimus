# tests/test_core.py
import pytest
import subprocess
from llama_optimus.core import estimate_max_ngl, run_llama_bench_with_csv

def test_estimate_max_ngl_min():
    max_ngl = estimate_max_ngl(min_ngl=0, max_ngl=1)
    assert isinstance(max_ngl, int)
    assert max_ngl >= 0

def test_estimate_max_ngl_binary(monkeypatch):
    monkeypatch.setattr('llama_optimus.core.subprocess.run', lambda *args, **kw: None)
    max_ngl = estimate_max_ngl(min_ngl=0, max_ngl=5)
    assert isinstance(max_ngl, int)
    assert 0 <= max_ngl <= 5

def test_invalid_llama_bench(tmp_path, monkeypatch):
    # Simulate failure in subprocess
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0])
    monkeypatch.setattr('llama_optimus.core.subprocess.run', fake_run)

    cmd = ['llama-bench', '-o', 'csv']
    with pytest.raises(RuntimeError):
        run_llama_bench_with_csv(cmd, metric='tg')
