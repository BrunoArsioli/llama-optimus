import subprocess
import re
import optuna

# You can later move these to search_space.py if desired
SEARCH_SPACE = {
    'threads':    {'low': 1, 'high': 12},       # Adjust range to your hardware
    'n_batch':    {'low': 16, 'high': 8192, 'log': True},
    'gpu_layers': {'low': 0, 'high': 70}        # (-ngl) Set max to model + VRAM; The max value must be found first
    'm_map':      {'low': 0, 'high': 1}         # Enable memory mapping when models are loaded (defaut:0)
}


def extract_tokens_per_sec(output):
    # Typical line: 'eval time = 1.02 s, tok/s = 322.1'
    match = re.search(r"tok/s\s*=\s*([0-9.]+)", output)
    if match:
        return float(match.group(1))
    raise ValueError("Could not parse tokens/sec from output:\n" + output)


def objective(trial):
    # Sample params
    threads    = trial.suggest_int('threads', SEARCH_SPACE['threads']['low'], SEARCH_SPACE['threads']['high'])
    n_batch    = trial.suggest_int('n_batch', SEARCH_SPACE['n_batch']['low'], SEARCH_SPACE['n_batch']['high'], log=SEARCH_SPACE['n_batch'].get('log', False))
    gpu_layers = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])
    m_map      = trial.suggest_int('m_map', SEARCH_SPACE['m_map']['low'], SEARCH_SPACE['m_map']['high'])

    # Build llama-bench command (edit as needed)
    cmd = [
        "llama-bench",         # path to your llama-bench binary
        "-t", str(threads),
        "--n_batch", str(n_batch),
        "--gpu-layers", str(gpu_layers),    # (-ngl flag)
        "--model", "PATH_TO_MODEL.gguf",    # <--- change this or parametrize
        "-n", "80",             # tokens to generate (larger value improve final statistics, i.e. lower std intok/s)
        "-p", "80",             # tokens to process (larger value improve final statistics, i.e. lower std intok/s)
        "m_map", str(m_map),    # 0; load entire model to RAM. 1; map memory and load what is needed 
        "-r", "5"               # number of benchmark runs for each configuration; mean value and std calculated from it 
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            print(f"llama-bench failed: {output}")
            return 0.0
        tok_s = extract_tokens_per_sec(output)
        return tok_s
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def run_optimization(n_trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best config:", study.best_trial.params)
    print("Best tokens/sec:", study.best_value)

if __name__ == "__main__":
    run_optimization()





