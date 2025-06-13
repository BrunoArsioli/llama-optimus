# __init__.py
# handle version, import core functions

"""
llama_optimus
-------------
A CLI tool and core library for optimizing llama.cpp performance flags using Optuna.
"""

__version__ = "0.1.0"

# ensure functions and constants are importable from the package root
from .core import (
    estimate_max_ngl, 
    run_llama_bench_with_csv, 
    run_optimization,
    SEARCH_SPACE,
)

