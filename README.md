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

3. Find the path to your llama.cpp/build/bin folder:

    Go to your llama.cpp folder; After you build, go to llama.cpp/build/bin 
    Use 'pwd' to print you path to build/bin folder, which should be something with the form
    my_path/llama.cpp/build/bin. Copy this entire path.

    Do the same and find the path to your AI models (i.e. gemma3xxxx.gguf file)

4. Edit the 'set_local_paths.sh' script 

    You need to update this script, providing the paths to both llama.cpp/build/bin 
    and to your model.gguf file. 

    (Note: If you have a big model, split in two or more files, e.g. Llama-4-Scout-17B-16E-Instruct-UD-Q5_K_XL-00001-of-00002.gguf and Llama-4-Scout-17B-16E-Instruct-UD-Q5_K_XL-00002-of-00002.gguf, 
    you just need to provide a path pointing to 
    the first file: Llama-4-Scout-17B-16E-Instruct-UD-Q5_K_XL-00001-of-00002.gguf. 
    llama-bench will handle it for you)

5. Run
    ```bash
    source set_local_paths.sh
    python src/benchmark.py
    ``` 
 
 Results will show you the best configuration for token generation with llama.cpp.

 Optimus tokens/s:  

 ` (-t)   -->   'threads' `

 ` (-b)   -->   'n_batch' `

 ` (-ngl) -->   'gpu_layers' `
 
 ` (-mmp) -->   'm_map' `

 ` (-ub)  -->   'ubatch-size' `

 ` (-fa)  -->   'flash-attn' `




6. The defaut llama.cpp configuration:

  test parameters:

  -d, --n-depth                          (default: 0)

  -b, --batch-size                       (default: 2048)

  -ub, --ubatch-size                     (default: 512)

  -ctk, --cache-type-k                   (default: f16)

  -ctv, --cache-type-v                   (default: f16)

  -dt, --defrag-thold                    (default: -1)

  -t, --threads                          (default: system dependent)

  -C, --cpu-mask                         (default: 0x0)

  --cpu-strict <0|1>                     (default: 0)

  --poll <0...100>                       (default: 50)

  -ngl, --n-gpu-layers                   (default: 99)

  -rpc, --rpc                            (default: none)

  -sm, --split-mode                      (default: layer)

  -mg, --main-gpu                        (default: 0)

  -nkvo, --no-kv-offload <0|1>           (default: 0)

  -fa, --flash-attn <0|1>                (default: 0)

  -mmp, --mmap <0|1>                     (default: 1)

  -embd, --embeddings <0|1>              (default: 0)

  -ts, --tensor-split <ts0/ts1/..>       (default: 0)

  -ot --override-tensors <tensor name pattern>=<buffer type>;...
                                            (default: disabled)

  -nopo, --no-op-offload <0|1>              (default: 0)
