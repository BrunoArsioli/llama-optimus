# llama-optimus
Lightweight Python tool using Optuna for tuning llama.cpp flags: towards optimal tok/s for your machine.  


## What it does

* Runs performance benchmarks of llama.cpp across parameters (threads, batch size, GPU layers, etc.)
* Uses Optuna to maximize tokens/sec on your specific hardware
* Exports the winning config for reproducibility
* Supports Apple Silicon, Linux, (next: NVIDIA platforms)

  
## Requirements

- Python 3.10 or higher
- llama.cpp (`llama-bench` or `llama-cli`) — see [llama.cpp repo](https://github.com/ggerganov/llama.cpp) for installation

## Installation

1. Install Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Install llama.cpp:

    See [llama.cpp repository](https://github.com/ggerganov/llama.cpp) for full build instructions.

    Example (Linux/Mac):

    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make
    ```

    Make sure the built CLI (`llama-bench`, `llama-cli`, etc.) is accessible in your PATH.
