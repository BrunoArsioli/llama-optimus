SEARCH_SPACE = {
    'batch_size'     : {'low': 8, 'high': 16384},   # 
    'ubatch_size'    : {'low': 4, 'high': 8192},    #  
    'threads':    {'low': 1, 'high': max_threads},  # Adjust range to your hardware
    'gpu_layers': {'low': 0, 'high': 149},          # (-ngl) Set max to model + VRAM; The max value must be found first
    'flash_attn': [0,1],                            #  --flash-attn <0|1> ; Enables flash attention       
    'override_spc'   : list(OVERRIDE_PATTERNS.keys())  # read list from src/llama_optimus/override_patterns.py
    #'flash_attn_type': [0, 1, 2], # not yet merged to main llama.cpp
}