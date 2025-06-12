import subprocess
import re
import optuna
import os
import shutil
import pandas as pd 
import tempfile
import subprocess 

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
    'threads':    {'low': 1, 'high': max_threads},       # Adjust range to your hardware
    'n_batch':    {'low': 16, 'high': 8192, 'log': True},
    'gpu_layers': {'low': 0, 'high': 70},       # (-ngl) Set max to model + VRAM; The max value must be found first
    'm_map':      {'low': 0, 'high': 1}         # Enable memory mapping when models are loaded (defaut:0)
}


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
    n_batch    = trial.suggest_int('n_batch', SEARCH_SPACE['n_batch']['low'], SEARCH_SPACE['n_batch']['high'], log=SEARCH_SPACE['n_batch'].get('log', False))
    gpu_layers = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])
    m_map      = trial.suggest_int('m_map', SEARCH_SPACE['m_map']['low'], SEARCH_SPACE['m_map']['high'])

    # call tempfile to hold results from llama-bench 
    csvfile = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)

    # Build llama-bench command (can edit to add more flags)
    cmd = [
        llama_bench_path,        # path to your llama-bench binary
        "-t", str(threads),
        "--batch-size", str(n_batch), # (-b flag)
        "-ngl", str(gpu_layers),      # (-ngl or --n-gpu-layers flag)
        "--model", model_path,    # <--- change this or parametrize
        "-n", "10",             # tokens to generate (larger value improve final statistics, i.e. lower std intok/s)
        "-p", "10",             # tokens to process (larger value improve final statistics, i.e. lower std intok/s)
        "-mmp", str(m_map),     # 0; load entire model to RAM. 1; map memory and load what is needed 
        "-r", "3",              # number of benchmark runs for each configuration; mean value and std calculated from it 
        "-o", "csv"             # save temporary .csv file with llama-bench outputs
    ]

    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def run_optimization(n_trials=7):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best config:", study.best_trial.params)
    print("Best tokens/sec:", study.best_value)

if __name__ == "__main__":
    run_optimization()
