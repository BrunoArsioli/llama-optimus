import subprocess
import re
import optuna
import os
import shutil
import pandas as pd 
import tempfile
import argparse 

# count number of available cpus
max_threads = os.cpu_count()

# You can later move these to search_space.py if desired
SEARCH_SPACE = {
    #'m_map': [0,1],          # Enable memory mapping when models are loaded (defaut:0)
    'flash_attn': [0,1],     #  --flash-attn <0|1>  ; Enables flash attention       
    'gpu_layers': {'low': 0, 'high': 99},           # (-ngl) Set max to model + VRAM; The max value must be found first
    'threads':    {'low': 1, 'high': max_threads},  # Adjust range to your hardware
    'ubatch_size'    : [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], #  
    'batch_size'     : [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]  # select number from list
}

# estimate max value for number of layers; ngl 
# llama-bench crash if ngl is too large. Depends on model and hardware
def estimate_max_ngl(min_ngl=0, max_ngl=SEARCH_SPACE['gpu_layers']['high']):
    low, high = min_ngl, max_ngl
    print("First, we need to estimate the maximum number of model layer (-ngl) " \
    "that can be loaded to your RAM memory.")

    while low < high:
        mid = (low + high + 1) // 2
        print(f"Testing for: -ngl = {mid}")

        cmd = [
            llama_bench_path,
            "--model", model_path,
            "-t",  str(max_threads),
            "-n", "1",     # minimal token-generation
            "-r", "1",
            "-ngl", str(mid),
            "-o", "csv"
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=80, check=True)
            low = mid  # success → try higher
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            high = mid - 1  # failure → reduce range
        
        #try:
        #    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        #    low = mid  # success → try higher
        #except subprocess.CalledProcessError:
        #    high = mid - 1  # failure → reduce range
    print(f"Estimated max ngl = {low}")
    return low



def run_llama_bench_with_csv(cmd, metric):
    # cmd should include: ... "-o", "csv"
    # use metric (a string: "tg", "pp", or "mean")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    
    # for debug 
    print(result.stdout)

    # Save stdout to a temp CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as csvfile:
        csvfile.write(result.stdout)
        csv_path = csvfile.name

    df = pd.read_csv(csv_path)
    metric_value = 0. # start metric value

    if metric == "tg":
        tg_rows = df[df["n_gen"] > 0]
        if not tg_rows.empty: # writes only if tg_row is not empty 
            metric_value = float(tg_rows["avg_ts"].iloc[0])
    elif metric == "pp":
        pp_rows = df[df["n_prompt"] > 0]
        if not pp_rows.empty: # writes only if pp_row is not empty 
            metric_value = float(pp_rows["avg_ts"].iloc[0])
    elif metric == "mean":
        tg_rows = df[df["n_gen"] > 0]
        pp_rows = df[df["n_prompt"] > 0]
        if not tg_rows.empty and not pp_rows.empty: # writes only if tg_ and pp_row are not empty 
            metric_value = 0.5 * (float(tg_rows["avg_ts"].iloc[0]) + float(pp_rows["avg_ts"].iloc[0]))
    return metric_value


def objective(trial, metric):
    # Sample params
    batch      = trial.suggest_categorical('batch', SEARCH_SPACE['batch_size'])
    flash      = trial.suggest_categorical('flash', SEARCH_SPACE['flash_attn'])
    u_batch    = trial.suggest_categorical('u_batch', SEARCH_SPACE['ubatch_size'])
    threads    = trial.suggest_int('threads', SEARCH_SPACE['threads']['low'], SEARCH_SPACE['threads']['high'])
    gpu_layers = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])


    # Build llama-bench command (can edit to add more flags)
    cmd = [
        llama_bench_path, # path to your llama-bench binary
        "-t"            , str(threads),
        "--batch-size"  , str(batch),      # (-b flag) (defaut 2024)
        "--ubatch-size" , str(u_batch),    # (-ub flag) (defaut 512) 
        "-ngl"          , str(gpu_layers), # (-ngl or --n-gpu-layers flag)
        "--flash-attn"  , str(flash),      # enables Flash Attention, a performance optimization during inference. 
        "--model"       , model_path,      # <--- change this or parametrize
        "-r"            , str(args.repeat),# number of benchmark runs/repetitions for each configuration; mean value and std calculated from it 
        "-o"            , "csv"            # save temporary .csv file with llama-bench outputs
    ]
    # note1: memory mapping is now set by defaut. Instead, need to add --no-map flag. 
    # note2: use "-r 5" for more robust results (mean value calculated over 5 llama-bench runs); Use "-r 1" for quick assessment 
    #        e.g., launch tool with: python src/optimus.py --trials 30 -r 1 

    # Add task-specific flags
    if metric in ("tg", "mean"):
        cmd += ["-n", "40"]  # tokens to generate (larger value improve final statistics, i.e. lower std in tok/s)
    if metric in ("pp", "mean"):
        cmd += ["-p", "40"]  # tokens to process (larger value improve final statistics, i.e. lower std in tok/s)

    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd, metric)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def run_optimization(n_trials=35, metric="tg"):  # (default: 35 trials, metric="tg" ; token generation speed)
    study = optuna.create_study(direction="maximize")
    # use lambda to inject metric 
    study.optimize(lambda trial: objective(trial, metric), n_trials=n_trials)
    print("Best config:", study.best_trial.params)
    print(f"Best {metric} tokens/sec:", study.best_value)

    # output: ready to use llama-server command with best llama.cpp parameters
    best = study.best_trial.params
    #command = f"{llama_bin_path}/llama-server --model {model_path} -t {best['threads']} --batch-size {best['batch']} --ubatch-size {best['u_batch']} -ngl {best['gpu_layers']} --flash-attn {best['flash']}"
    #print("\n# Copy and paste to run llama-server with these optimized settings:\n" + command)

    print("\n# You are ready to run a local llama-server:")

    # 1. llama-server (inference); listening at http://127.0.0.1:8080/ in your browser. 
    llama_server_cmd = (
        f"{llama_bin_path}/llama-server --model {model_path}"
        f" -t {best['threads']}"
        f" --batch-size {best['batch']}"
        f" --ubatch-size {best['u_batch']}"
        f" -ngl {best['gpu_layers']}"
        f" --flash-attn {best['flash']}"
    )
    print("\n# For optimal inference:")
    print(llama_server_cmd)

    # 2. llama-bench (benchmark for both tg and pp)
    llama_bench_cmd = (
        f"{llama_bench_path}"
        f" --model {model_path}"
        f" -t {best['threads']}"
        f" --batch-size {best['batch']}"
        f" --ubatch-size {best['u_batch']}"
        f" -ngl {best['gpu_layers']}"
        f" --flash-attn {best['flash']}"
        f" -n 128 -p 128 -r 3 -o csv"
    )
    print("\n# To benchmark both generation and prompt processing speeds:")
    print(llama_bench_cmd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama-optimus: Benchmark & tune llama.cpp.")
    parser.add_argument("--trials", type=int, default=35, help="Number of Optuna trials: n searches in parameter space")
    parser.add_argument("--model", type=str, help="Path to model (overrides env var)")
    parser.add_argument("--llama-bin", type=str, help="Path to llama.cpp build/bin folder (overrides env var)")
    parser.add_argument("--metric", type=str, default="tg", choices=["tg", "pp", "mean"],
        help="Which throughput metric to optimize: 'tg' (token generation, default), 'pp' (prompt processing), or 'mean' (average of both)")
    parser.add_argument("--ngl-max",type=int, 
        help="Maximum number of model layers for -ngl (skip estimation if provided; estimation runs by defaut).")
    parser.add_argument("--repeat", "-r", type=int, default=2,
        help="Number of llama-bench runs per configuration (higher = more robust, lower = faster; default: 2, for quick assessement: 1)")
    args = parser.parse_args()

    # Set paths based on CLI flags or env vars
    llama_bin_path = args.llama_bin if args.llama_bin else os.environ.get("LLAMA_BIN")
    llama_bench_path = f"{llama_bin_path}/llama-bench"
    model_path = args.model if args.model else os.environ.get("MODEL_PATH")

    if llama_bin_path is None or model_path is None:
        raise ValueError("LLAMA_BIN or MODEL_PATH not set. Set via environment variable or pass via CLI flags.")

    if not os.path.isfile(llama_bench_path):
        raise FileNotFoundError(f"llama-bench not found at {llama_bench_path}." \
        "Please, install llama.cpp and make sure to set your local LLAMA_BIN variable. "
        "(i.e. edit approprietely and run the script named 'set_local_paths.sh')" \
        "alternative: comment the 'llama_bench_path = xxxxx' line and " \
        "add a line 'llama_bench_path = \"PATH_TO_LLAMA-BENCH\" ' to define the path to llama-bench.") 

    print(f"Number of CPUs: {max_threads}.")
    print(f"Path to 'llama-bench':{llama_bench_path}")  # in llama.cpp/tools/
    print(f"Path to 'model.gguf' file:{model_path}")

    # defaut: estimate maximum number of layers before run_optimization 
    # in case the user knows ngl_max value, squipe ngl_max estimate
    if args.ngl_max is not None: 
        SEARCH_SPACE['gpu_layers']['high'] = args.ngl_max
        print(f"User-specified maximum -ngl set to {args.ngl_max}")
    else:
        SEARCH_SPACE['gpu_layers']['high'] = estimate_max_ngl()
        print(f"Setting maximum -ngl to {SEARCH_SPACE['gpu_layers']['high']}")

    run_optimization(n_trials=args.trials, metric=args.metric)  


