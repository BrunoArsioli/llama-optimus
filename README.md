# llama-optimus
Lightweight Python tool using Optuna for tuning llama.cpp flags: towards optimal tok/s for your machine.  



## Features:

Llama‑optimus is a tool for automating throughput tuning of llama.cpp using Optuna.


- Automated benchmarking of inference speed (tokens/s) across parameter combinations

- Optimizes CPU threads, batch size, GPU layers (Metal/CUDA), and more

- Exportable configuration for reproducible setup

- Suitable for Apple Silicon, Linux (and NVIDIA platforms, next)
  
