# src/llama_optimus/override_patterns.py
"""
Dictionary of override-tensor patterns for use with llama.cpp.
Key: short, human-readable name.
Value: the pattern string to pass to --override-tensor.

Notes: The keys are preset names (shown in CLI as choices); 
the values are regex patterns passed to --override-tensor.

These focus on offloading the largest MoE expert tensors 
(ffn_up_exps, ffn_down_exps, etc).

Advanced users can add custom patterns as needed for advanced setups.
"""

# test version
#OVERRIDE_PATTERNS = {
#    #"none"          : "",
#    "ffn_cpu_all"   : r"blk\.\d+\.ffn_.*_exps.=CPU",
#    "ffn_cpu_even"  : r"blk\.(?:[0-9]*[02468])\.ffn_.*_exps.=CPU",
#    "ffn_cpu_odd"   : r"blk\.(?:[0-9]*[13579])\.ffn_.*_exps.=CPU",
#}

OVERRIDE_PATTERNS = {
    # pass no override-tensor flag
    "none": "",

    # Offload **all** expert FFN tensors (max VRAM savings, slowest if you have only CPU for experts)
    "ffn_cpu_all": r"blk\.\d+\.ffn_.*_exps\.=CPU",

    # Offload only experts in **even-numbered** blocks
    "ffn_cpu_even": r"blk\.(?:[0-9]*[02468])\.ffn_.*_exps\.=CPU",

    # Offload only experts in **odd-numbered** blocks
    "ffn_cpu_odd": r"blk\.(?:[0-9]*[13579])\.ffn_.*_exps\.=CPU",

    # Offload **only up- and down-projection** weights of all experts (keep gating nets on GPU; good trade-off)
    "ffn_cpu_updown": r"blk\.\d+\.ffn_(?:up|down)_exps\.=CPU",

    # Offload **only up-projection** expert weights (very light offloading, keeps most things on GPU)
    "ffn_cpu_up": r"blk\.\d+\.ffn_up_exps\.=CPU",

    # Offload **only down-projection** expert weights
    "ffn_cpu_down": r"blk\.\d+\.ffn_down_exps\.=CPU",

    # Example: Offload experts in last 25% of layers (if model has 80 layers, this targets 60-79)
    # You might want to edit this for your specific model depth.
    "ffn_cpu_last_quarter": r"blk\.(6[0-9]|7[0-9])\.ffn_.*_exps\.=CPU",

    # Example: Offload experts from block 6 onward (Unsloth/Reddit pattern)
    "ffn_cpu_from_6": r"blk\.(6|7|8|9|[1-9][0-9]+)\.ffn_.*_exps\.=CPU",
}

