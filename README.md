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

    You should either add the llama.cpp/build/bin path to your shell’s $PATH, or edit your benchmark.py script to use the full path to llama-bench.

    2.1. Create a path to your llama.cpp/build/bin and add to your shell's path:

        Go to your llama.cpp folder; After you build, go to llama.cpp/build/bin 
        Use 'pwd' to print you path to build/bin folder, which should be something with the form
        my_path/llama.cpp/build/bin. Copy this entire path.

        For Zsh (~/.zshrc):
            ````
            echo 'export LLAMA_BENCH_PATH="my_path/llama.cpp/build/bin/llama-bench:$LLAMA_BENCH_PATH" ' >> ~/.zshrc 
            source ~/.zshrc
            ```         

        Or, use the full path in your benchmar.py script:
            Just replace "llama_bench_path" in your code with the actual path, e.g.:
            ```
            llama_bench_path = "my_path/llama.cpp/build/bin/llama-bench"
            ```



