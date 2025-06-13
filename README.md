# llama-optimus

> **Running Local AI?**
>
> **Maximize your token/s for prompt processing (pp) & token generation (tg)**
>
> llama-optimus is a lightweight Python tool to automatically optimize `llama.cpp` performance flags for maximum throughput.
>
> Supports Apple Silicon, Linux, and NVIDIA GPUs.
>
> **Brings Bayesian optimization (Optuna) to your local or embedded AI models.**
>
> **Find the *BEST* performance flags for *YOUR* unique hardware in minutes.**

---


<p align="center">
    <img src="assets/llama.optimus_llama_optuna_logo.png" width="600">
</p>


## What does llama-optimus do?

- **Benchmarks and tunes** `llama.cpp` models for maximum `tokens/sec`, using automated parameter search.
- **Optimizes for token generation, prompt processing, or both.**
- **Estimates or accepts user-specified GPU layer count (`-ngl`).**
- **CLI interface:** All major parameters and paths are settable via command line or environment variable.
- **Built on:** [Optuna](https://optuna.org/) for hyperparameter optimization and [llama.cpp](https://github.com/ggerganov/llama.cpp) for inference.

---

## Requirements

- Python 3.10+  
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (built, with `llama-bench` present in `/build/bin/`)
- At least one GGUF model file

---

## Installation

1. **Clone this repo:**
    ```bash
    git clone https://github.com/YOUR-USERNAME/llama-optimus.git
    cd llama-optimus
    ```

2. **(Recommended) Create and activate a Python virtualenv:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Build llama.cpp:**
    - Follow the [llama.cpp instructions](https://github.com/ggerganov/llama.cpp#build).
    - Note your full path to `llama-bench` (e.g., `/your/path/llama.cpp/build/bin/llama-bench`).

---

## ⚙️ Configuration

### Option A: **Pass as CLI flags (recommended)**
```bash
python src/optimus.py --llama-bin /path/to/llama.cpp/build/bin \
                      --model /path/to/model.gguf \
                      --metric tg \
                      --trials 20 \
                      --repeat 2 \
                      --ngl-max 32
```

- All arguments are optional except `--llama-bin` and `--model` (if not set as env variables).
- CLI flags **override environment variables**.

### Option B: **Set as environment variables**
```bash
export LLAMA_BIN=/path/to/llama.cpp/build/bin
export MODEL_PATH=/path/to/model.gguf
python src/optimus.py
```

### Option C: **Source a helper script to set your local variables**
Edit and `source set_local_paths.sh` (see template in repo).

---

## Usage

| Flag               | Default   | Description                                                                                   |
|--------------------|-----------|-----------------------------------------------------------------------------------------------|
| `--llama-bin`      | *(env)*   | Path to folder containing `llama-bench`                                                       |
| `--model`          | *(env)*   | Path to GGUF model file                                                                       |
| `--metric`         | `tg`      | Which metric to optimize: `tg` (generation), `pp` (processing), `mean` (average of both)      |
| `--trials`         | `35`      | Number of Optuna optimization trials                                                          |
| `--repeat`, `-r`   | `2`       | Number of runs per trial (higher=more robust; lower=faster)                                   |
| `--ngl-max`        | *(auto)*  | Max model layers for `-ngl`. Skip estimation if provided                                      |

### **Examples**
- **Optimize for generation speed:**  
    ```bash
    python src/optimus.py --llama-bin ... --model ... --metric tg
    ```
- **Optimize for prompt processing speed:**  
    ```bash
    python src/optimus.py --metric pp
    ```
- **Fast test:**  
    ```bash
    python src/optimus.py --trials 1 --repeat 1
    ```
- **User-known max layers:**  
    ```bash
    python src/optimus.py --ngl-max 32
    ```

---

## Output

After running, you will see:
- **The best config found** (Optuna trial with parameters)
- **Tokens/sec** for your selected metric
- Intermediate logs of each trial

> Save output for later:  
> `python src/optimus.py ... > results.txt`

---

## How it works

- By default, estimates your hardware's max `-ngl` (GPU layers). If you know your max, use `--ngl-max` to skip this step.
- Explores batch size, threads, microbatch, flash attention, and more, to maximize your chosen throughput metric.

---
## 🛟 Troubleshooting 🛟

- **`llama-bench not found`**: Check your `--llama-bin` path or `LLAMA_BIN` env var.
- **`MODEL_PATH not set`**: Use `--model` or set the env variable.
- **Zero tokens/sec**: Try reducing `--ngl-max` or verifying your model is compatible with your hardware.

---

## Contributing

PRs, suggestions, and benchmarks welcome!  
Open an issue or submit a PR.

---

## License

MIT License (see LICENSE).

---

## Acknowledgements

**Build with Optuna**

<p align="left"><img src="assets/optuna_logo.png" width="500"></p>
- [Optuna](https://optuna.org/)

<br>

**Made for llama.cpp**

<p align="left"><img src="assets/llama.cpp_logo.png" width="500"></p>
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

<br>

**llama.optimus**
<p align="left"><img src="assets/llama.optimus_logo.png" width="500"></p>
- [llama.optimus](https://github.com/BrunoArsioli/llama-optimus)

---

**Optimize your LLM inference—contribute your configs, and help the community improve!**


