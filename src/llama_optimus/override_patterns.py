# src/llama_optimus/override_patterns.py
"""
Default regex templates for --override-tensor exploration.

Keys = human-readable names exposed to Optuna / CLI.
Values = regex strings passed verbatim to llama.cpp (--override-tensor '<regex>=CPU')
"""

OVERRIDE_PATTERNS = {
    "none"          : "",
    "ffn_cpu_all"   : r"blk\.\d+\.ffn_.*_exps.=CPU",
    "ffn_cpu_even"  : r"blk\.(?:[0-9]*[02468])\.ffn_.*_exps.=CPU",
    "ffn_cpu_odd"   : r"blk\.(?:[0-9]*[13579])\.ffn_.*_exps.=CPU",
}
