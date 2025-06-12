import subprocess
import re
import optuna
import os
import shutil
import pandas as pd 
import tempfile
import subprocess 
import argparse 

# count number of available cpus
max_threads = os.cpu_count()

# path to llama-bench (usualy in llama.cpp/tools)
#llama_bench_path = shutil.which("llama-bench")
llama_bin_path = os.environ.get("LLAMA_BIN")
llama_bench_path = f"{llama_bin_path}/llama-bench"

# define path to your model
model_path = os.environ.get("MODEL_PATH")

if llama_bin_path is None or model_path is None:
    raise ValueError("LLAMA_BIN or MODEL_PATH not set. Did you source set_local_paths.sh?")

if llama_bench_path is None:
    raise FileNotFoundError("llama-bench not found in PATH. " \
    "Please, install llama.cpp and make sure to set your local LLAMA_BIN variable. "
    "(i.e. edit approprietely and run the script named 'set_local_paths.sh')" \
    "alternative: comment the 'llama_bench_path = xxxxx' line and " \
    "add a line 'llama_bench_path = \"PATH_TO_LLAMA-BENCH\" ' to define the path to llama-bench.") 

print(f"Number of CPUs: {max_threads}.")
print(f"Path to 'llama-bench':{llama_bench_path}")  # in llama.cpp/tools/
print(f"Path to 'model.gguf' file:{model_path}")

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
    print("Started running estimate for the maximum number of model layer (-ngl) " \
    "that can be passed to you RAM memory")

    # avoid using max_threads in test
    if max_threads > 1:
        max_threads_test = int(max_threads/2)
    else:
        max_threads_test = max_threads 
    print(f"Max_threads in test: {max_threads_test}")

    while low < high:
        mid = (low + high + 1) // 2
        print(f"Testing for: -ngl = {mid}")

        cmd = [
            llama_bench_path,
            "--model", model_path,
            "-t",  str(max_threads_test),
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



def run_llama_bench_with_csv(cmd):
    # cmd should include: ... "-o", "csv"
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    # for debug 
    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    # Save stdout to a temp CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as csvfile:
        csvfile.write(result.stdout)
        csv_path = csvfile.name

    df = pd.read_csv(csv_path)
    tg_rows = df[df["n_gen"] > 0]  # catch the rows with results for the token generation speed 
    if tg_rows.empty:
        raise ValueError("No 'tg' (throughput) test results found in output CSV")
    return float(tg_rows["avg_ts"].iloc[0])


def objective(trial):
    # Sample params
    threads    = trial.suggest_int('threads', SEARCH_SPACE['threads']['low'], SEARCH_SPACE['threads']['high'])
    batch = trial.suggest_categorical('batch', SEARCH_SPACE['batch_size'])
    u_batch = trial.suggest_categorical('u_batch', SEARCH_SPACE['ubatch_size'])
    gpu_layers = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])
    flash      = trial.suggest_categorical('flash', SEARCH_SPACE['flash_attn'])


    # Build llama-bench command (can edit to add more flags)
    cmd = [
        llama_bench_path,        # path to your llama-bench binary
        "-t", str(threads),
        "--batch-size",  str(batch),   # (-b flag) (defaut 2024)
        "--ubatch-size", str(u_batch), # (-ub flag) (defaut 512) 
        "-ngl", str(gpu_layers),    # (-ngl or --n-gpu-layers flag)
        "--flash-attn", str(flash), # enables Flash Attention, a performance optimization during inference. 
        "--model", model_path,      # <--- change this or parametrize
        "-n", "35",             # tokens to generate (larger value improve final statistics, i.e. lower std intok/s)
        "-r", "2",              # number of benchmark runs for each configuration; mean value and std calculated from it 
        "-o", "csv"             # save temporary .csv file with llama-bench outputs
    ]
    # note: memory mapping is now set by defaut. Instead, need to add --no-map flag. 

    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def run_optimization(n_trials=35):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best config:", study.best_trial.params)
    print("Best tokens/sec:", study.best_value)

if __name__ == "__main__":
    # estimate maximum number of layers before run 
    SEARCH_SPACE['gpu_layers']['high'] = estimate_max_ngl()
    run_optimization()
