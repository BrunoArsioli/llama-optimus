# llama-optimus

> **Running Local AI?**
>
> **Optimize llama.cpp hyperparameters**
>
> **Maximize your tokens/s for prompt processing (pp) & token generation (tg)**
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
    <img src="assets/llama.optimus_logo_PyPi_Optuna_Llama.png" width="600">
</p>


## What does llama-optimus do?

- **Tunes** `llama.cpp` parameters for maximum `tokens/sec`, using automated parameter search.
- **Bayesian optimization** (Optuna) is used to maximize tokens/sec for prompt processing, generation or both
- **Estimates user-specified GPU layer count (`-ngl`).**
- **Supports override patterns for --override-tensor**: allows you to optimize advanced memory offloading for large models or low VRAM systems.
- **Automatic system warmup detection** to ensure benchmarking is done under real-world, “steady-state” conditions.
- **Grid search over categorical parameters** (for flags like --override-tensor and --flash-attn ) combined with Bayesian tuning of numerical ones.
- **CLI interface:** All major parameters and paths are settable via command line or environment variable.
- **Built on:** [Optuna](https://optuna.org/) for hyperparameter optimization and [llama.cpp](https://github.com/ggerganov/llama.cpp) for inference.
- Adapts to Apple Silicon, Linux x86, and NVIDIA GPU systems
- Outputs copy-paste-ready commands for `llama-server` and `llama-bench`
- **Quick & robust benchmarks:** Controlable via `-repeat` and `--n-tokens` flags which set the number of llama-bench repetitions per trial, and the number of tokes used when estimating tokens/s velocity in prompt processing (pp) and text generation (tg). 

---

## Requirements

- Python 3.10+  
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (built, with `llama-bench` present in `/build/bin/`)
- At least one GGUF model file

---

## Installation

1. **Clone this repo:**
    ```bash
    git clone https://github.com/BrunoArsioli/llama-optimus
    cd llama-optimus
    ```

2. **(Recommended) Create and activate a Python virtualenv:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Python dependencies in editable/developer mode:**
    ```bash
    pip install -e .
    ```

    Or, to install just requirements (if not developing):
    ```bash
    pip install -r requirements.txt
    ```

4. **Build llama.cpp:**
    - Follow the [llama.cpp instructions](https://github.com/ggerganov/llama.cpp#build).
    - Note your full path to `llama-bin` (e.g., `/your_path_to/llama.cpp/build/bin`) for this tool to work.

---

## ⚙️ Configuration

- All arguments are optional except `--llama-bin` and `--model` (if not set as env variables).
- CLI flags **override environment variables**.

### Option A: **Set as environment variables**
```bash
export LLAMA_BIN=/path_to/llama.cpp/build/bin
export MODEL_PATH=/path_to/model.gguf
python src/optimus.py
```

then you only need:
```bash
llama-optimus
```

for a robust test, you can use more trials (default: 45), and more benchmark repetitions (default: 2)
```bash
llama-optimus --trials 70 -r 5 
```

to grid search all override-tensor presets and flash-attn options, and optimize numerical flags 
```bash
llama-optimus --override-mode scan --trials 20 --metric tg
```

### Quick Notes/FAQ:
What is a **trial**? It is a test of the model performance (e.g. how many tokens/s it can generate) given a configuration of llama.cpp parameters.

e.g.: When running `llama-optimus` you will see the output for every trial, like this: 

Trial 0 finished with value: 72.43 and parameters: {'batch': 32, 'flash_attn': 1, 'u_batch': 8, 'threads': 11, 'gpu_layers': 97}. Best is trial 0 with value: 72.43 

Meaning, if you run a llama-server with flags: --batchh-size 32 --flash-attn --ubatch-size 8 --threads 11 -ngl 97 ; you will get about 72.43 tokens/s in token generation. 

What is a **repetition** or `-r` ?

The result we got from Trial 0 (**72.43 tokens/s**) is a mean value; It was calculated by running this same **Trial 0** configuration `-r` times. 

Why do we need to **repeat** runs before calculating the result (tokens/s) ?

That is because, everytime you run `llama-bench` (the llama.cpp benchmark) you get a sligtly different estimate for the tokens/s metric. There is a degree of variability in the results; We calculate a mean value to base our final decision on more reliable/robust results.  

### Option B: Pass as CLI flags

```bash
llama-optimus --llama-bin ~my_path_t/llama.cpp/build/bin --model ~my_path_to/models/my-model.gguf
```

### Option C: Source a helper script
You may create a script (e.g., set_local_paths.sh) with the content:
```bash
#!/usr/bin/env bash
export LLAMA_BIN=~my_path_to/llama.cpp/build/bin
export MODEL_PATH=~my_path_to/models/my-model.gguf
```
and `source set_local_paths.sh` before running `llama-optimus`.

---

## Usage

run llama-optimus with 25 trials, 3 repetitions per trial, and only estimate token generation **tg** velocity: 
```bash
llama-optimus --llama-bin my_path_to/llama.cpp/build/bin --model my_path_to/model.gguf --trials 25 -r 3 --metric tg
```

Or, if you prefer to use environment variables :

```bash
export LLAMA_BIN=my_path_to/llama.cpp/build/bin
export MODEL_PATH=my_path_to/model.gguf
llama-optimus --trials 25 -r 3 --metric tg
```

**Common arguments:**

* `--llama-bin`  Path to your llama.cpp `build/bin` directory (or set `$LLAMA_BIN`)
* `--model`      Path to your `.gguf` model file (or set `$MODEL_PATH`)
* `--trials`   Number of optimization trials (default: 35)
* `--metric`   Which throughput metric to optimize:
  `tg` = token generation speed,
  `pp` = prompt processing speed,
  `mean` = average of both
* `-r` / `--repeat` How many repetitions per configuration (default: 2; use 1 for quick/dirty, 5 for robust)
* `--n-tokens`  Number of tokens to use for benchmarking. Larger = more stable measurements (default: 60).
* `--override-mode`  How to treat --override-tensor (default: scan):
    `none`: ignore this flag (do not scan over override-tensor space)
    `scan`: grid search all preset patterns (from override_patterns.py)
    `custom`: ([TBD]future) user-defined override tensor patterns
* `--n-warmup-tokens` Number of tokens passed to llama-bench during each warmup loop
* `--no-warmup` Skip warmup phase (for test/debug purpose)



See all options:

```bash
llama-optimus --help
```

---

## How it works

- By default, starts with a warm-up. (see extra notes)

- By default, starts by estimating your hardware's max `-ngl` (GPU layers) for a given model. If you know your max, use `--ngl-max` to skip this step.

- Uses Optuna to search for the best combination of `llama.cpp` flags (batch size, ubatch, threads, ngl, etc).

- Runs quick benchmarks (with `llama-bench`) for each config, parses results from the CSV, and feeds back to the optimizer.

- Hierarchical optimization:

    Stage 1: Bayesian search over numerical flags (batch_size, ubatch_size, threads, gpu_layers)

    Stage 2: Grid search over categorical flags (override-tensor, flash-attn), using the best numericals found

    Stage 3: Final fine-tuning over numerical flags with best categorical flags fixed

- Finds the best flags in minutes —no need to try random combinations!

- Handles errors (non-working configs are skipped).

- You can **abort Trial** at any time, and still follow the best result/configuration achieved so far. 

**Notes:** llama-optimus ensures your system is warmed up before starting optimization, to avoid misleading cold-start results (i.e. cold-starts lead to larger tokens/s counts; This a source of confugion for the community, because cold-start results are usually reported as "I just found a new configuration which improved tokens/s in 20%", which is misleading).

---

## Output

* After running, you'll see:

  * The best configuration found for your hardware/model.
  * **A ready-to-copy `llama-server` command line** for running optimal inference.
  * **A ready-to-copy `llama-bench` command** for benchmarking both prompt processing and generation speeds.
  * Example output:

```
Best config: {'batch': 4096, 'flash': 1, 'u_batch': 1024, 'threads': 4, 'gpu_layers': 93}
Best tg tokens/sec: 73.5

# You are ready to run a local llama-server:

# For optimal inference:
llama-server --model my_path_to/model.gguf -t 4 --batch-size 4096 --ubatch-size 1024 -ngl 93 --flash-attn 

# To benchmark both generation and prompt processing speeds:
llama-bench --model my_path_to/model.gguf -t 4 --batch-size 4096 --ubatch-size 1024 -ngl 93 --flash-attn 1 -n 128 -p 128 -r 3 -o csv

# In case you want Image-to-text capabilities on your server, you will need to add a flag: --mmproj my_path_to/mmproj_file.ggfu ; Every Image-to-text model has a projection file available. 
```

---

## Tip 

The default values for prompt processing `-p 50` and prompt generation `-n 50` gives fast trials.

The user can control this via `--n-tokens` parameter, which can be passed to llama-optimus during launch: lead to stable and robust results 

```bash
llama-optimus --n-tokens 128
```

Later, for a stable final score, re-run llama-bench with the best flags found (don't forget to warm-up first):
```bas'
llama-bench ... -p 512 -n 256 -r 5 --progress
```

E.g. of launching a server with optimized flags will look like this:
```bash
llama-server --model my_path_to/model.gguf -t 4 --batch-size 4096 --ubatch-size 1024 -ngl 63 --flash-attn  --override-tensor "blk\.(6|7|8|9|[1-9][0-9]+)\.ffn_.*_exps\.=CPU"
````

---

## Why Do a Warmup?

Initial runs are fast because hardware (especially Apple Silicon) is “cold”—no thermal throttling, RAM/cache is fresh, and CPU governor may be in turbo mode.

After several runs, temperature rises or RAM usage accumulates, causing the system to reduce clocks or evict caches.

If you start Trials (Stage 1 or 2) with “cold” hardware, the “best” config may just be lucky—chosen in a period of turbo clocks, giving misleading results.

For this reason, llama-optimus warms-up before scaning the parameter space with its Bayesian TPESampler. 

**Keep in mind**: never trust cold-start numbers. 
Warming up the system and waiting for stable, “saturated” (worst-case, real-world) performance will make your optimizer results much more robust and grounded to real use cases.

## 🛟 Troubleshooting 🛟

- **`llama-bench not found`**: Check your `--llama-bin` path or `LLAMA_BIN` env var.
- **`MODEL_PATH not set`**: Use `--model` or set the env variable.
- **Zero tokens/s**: Try reducing `--ngl-max` or verifying your model is compatible with your hardware.

---

## Project Structure

```bash
llama-optimus/
│
├── src/
│    └── llama_optimus/
│           ├── __init__.py
│           ├── core.py   # all optimization/benchmark logic
│           └── cli.py    # CLI interface (argparse, entrypoint)
│           └── search_space.py      # the numerical search space 
│           └── override_patterns.py # override-tensor patterns for use with llama.cpp.
│
├── test/
│   └── test_core.py
│
├── assets/
│   └── llama.optimus_logo.png
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Coments about other llama.cpp flags 

| Flag                                                       | Why it matters                                                                                                                                                                                                                                                | Suggested search values                                                                                                    | Notes                                                                                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **`--mmap / --no-mmap`** (memory-map model vs. fully load) | • On fast NVMe & Apple SSD, `--mmap 1` (default) is fine.<br>• On slower HDD/remote disks, disabling mmap (`--no-mmap` or `--mmap 0`) and loading the whole model into RAM often gives **10-20 % faster generation** (no page-fault stalls).                  | `[0, 1]` (boolean)                                                                                                         | Keep default `1`; let Optuna see if `0` wins on a given box.                                                                             |
| **`--cache-type-k / --cache-type-v`**                      | Setting key/value cache to **`f16` vs `q4`** or **`i8`** trades RAM vs speed.  Most Apple-Metal & CUDA users stick with `f16` (fast, larger).  For low-RAM CPUs increasing speed is impossible if it swaps; `q4` can shrink cache 2-3× at \~3-5 % speed cost. | `["f16","q4"]` for both k & v (skip i8 unless you target very tiny devices).                                               | Only worth searching when the user is on **CPU-only** or small-VRAM GPU. You can gate this by detecting “CUDA not found” or VRAM < 8 GB. |
| **`--main-gpu`** / **`--gpu-split`** (or `--tensor-split`) | Relevant only for multi-GPU rigs (NVIDIA).  Picking the right primary or a tensor split can cut VRAM fragmentation and enable higher `-ngl`.                                                                                                                  | If multi-GPU detected, expose `[0,1]` for `main-gpu` **and** a handful of tensor-split presets (`0,1`, `0,0.5,0.5`, etc.). | Keep disabled on single-GPU/Apple Silicon to avoid wasted trials.                                                                        |
| **[Preliminary] `--flash-attn-type 0/1/2`** (v0.2+ of llama.cpp)         | Metal + CUDA now have two flash-attention kernels (`0` ≈ old GEMM, `1` = FMHA, `2` = in-place FMHA). **!!Note!!:** Not yet merged to llama.cpp main.  Some M-series Macs get +5-8 % with type 2 vs 1.                                                                                                           | `[0,1,2]` —but **only if llama.cpp commit ≥ May 2025**.                                                                    | Add a version guard: skip the flag on older builds.                                                                                      |




---

## FAQ
**Q:** Does this tune all `llama.cpp` options? 
**A:** No, just the most performance-relevant (batch sizes, threads, GPU layers, Flash-Attn). Feel free to extend the SEARCH_SPACE dictionary.

**Q:** Can I use this with non-Meta models?
**A:** Yes! Any GGUF model supported by your llama.cpp build.

**Q:** Is it safe for my hardware?
**A:** Yes. Non-working configs are skipped, and benchmarks use conservative timeouts.

---

## Development & Testing
Write your own test scripts in `/test` (see `test_core.py` for simple mocking).

All major code logic is in `llama_optimus/core.py`.

Install in dev/edit mode: `pip install -e .`

Run CLI directly: `llama-optimus ...` or `python -m llama_optimus.cli ...`

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

<br>

**PyPi**

<p align="left"><img src="assets/logo-PyPi.svg" width="500"></p>
- [PyPi](https://pypi.org/)

---

**Optimize your LLM inference—contribute your configs, and help the community improve!**


