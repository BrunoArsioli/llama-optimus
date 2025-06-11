# llama-optimus
Lightweight Python tool using Optuna for tuning llama.cpp flags: towards optimal tok/s for your machine.  


## What it does

* Runs performance benchmarks of llama.cpp across parameters (threads, batch size, GPU layers, etc.)
* Uses Optuna to maximize tokens/sec on your specific hardware
* Exports the winning config for reproducibility
* Supports Apple Silicon, Linux, (next: NVIDIA platforms)

  
