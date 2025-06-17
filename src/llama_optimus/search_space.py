# src/llama_optimus/search_space.py
import os
from .override_patterns import OVERRIDE_PATTERNS  # if needed

# count number of available cpu cores
max_threads = os.cpu_count()

SEARCH_SPACE = {
    'batch_size'     : {'low': 8, 'high': 16384},   # 
    'ubatch_size'    : {'low': 4, 'high': 8192},    #  
    'threads':    {'low': 1, 'high': max_threads},  # Adjust range to your hardware
    'gpu_layers': {'low': 0, 'high': 149},          # (-ngl) Set max according to model and VRAM; The max value must be determined for each setup
    'flash_attn': [0,1],                            #  --flash-attn <0|1> ; Enables flash attention       
    'override_spc'   : list(OVERRIDE_PATTERNS.keys()) # Read list from src/llama_optimus/override_patterns.py
    #'flash_attn_type': [0, 1, 2], # Not yet merged to main llama.cpp
}