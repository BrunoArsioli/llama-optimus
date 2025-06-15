# core.py
# handle core functions 

import subprocess
import re
import optuna
import os
import shutil
import pandas as pd 
import tempfile
from optuna.samplers import TPESampler
from .override_patterns import OVERRIDE_PATTERNS   


# count number of available cpu cores
max_threads = os.cpu_count()

# [TBD] move to search_space.py 
SEARCH_SPACE = {
    'batch_size'     : {'low': 8, 'high': 16384},   # 
    'ubatch_size'    : {'low': 4, 'high': 8192},    #  
    'threads':    {'low': 1, 'high': max_threads},  # Adjust range to your hardware
    'gpu_layers': {'low': 0, 'high': 149},          # (-ngl) Set max to model + VRAM; The max value must be found first
    'flash_attn': [0,1],                            #  --flash-attn <0|1> ; Enables flash attention       
    'override_spc'   : list(OVERRIDE_PATTERNS.keys())  # read list from src/llama_optimus/override_patterns.py
    #'m_map': [1],             # Fixed; Enable memory mapping (0 = load fully, 1 = mmap)
    #'ubatch_size'    : [4, 8, 16, 24, 32, 48, 64, 96, 128, 182, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192], #  
    #'batch_size'     : [8, 16, 24, 32, 48, 64, 96, 128, 182, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288 , 16384],  # select number from list
    #'flash_attn_type': [0, 1, 2], # not yet merged to main llama.cpp
}


# estimate max value for number of layers; ngl 
# llama-bench crash if ngl is too large. Depends on model and hardware

def estimate_max_ngl(llama_bench_path, model_path, min_ngl=0, max_ngl=SEARCH_SPACE['gpu_layers']['high']):
    """
    Estimate the maximum number of model layers (-ngl) that can be loaded into GPU/VRAM
    for the current hardware and selected model. Uses a binary search, running llama-bench
    with minimal workload for each ngl value, and returns the highest value that does not crash.

    Parameters:
        min_ngl (int): The minimum ngl value to try (default: 0).
        max_ngl (int): The maximum ngl value to try (default: 99, set by SEARCH_SPACE).

    Returns:
        int: The highest working ngl value for this model/hardware.
    """

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
        
    print(f"Estimated max ngl = {low}")
    return low



def run_llama_bench_with_csv(cmd, metric):
    """
    Run llama-bench using the specified command, saving the output as a temporary CSV,
    and extract the desired throughput metric from the CSV output.

    Parameters:
        cmd (list): The full command (as a list) to run llama-bench.
        metric (str): Which throughput metric to extract: "tg", "pp", or "mean".

    Returns:
        float: The value of the selected metric, or 0.0 if it cannot be extracted.
    """    

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    
    # for debug 
    #print(result.stdout)

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


def objective_1(trial, metric, repeat, llama_bench_path, model_path):
    """
    Objective function for Optuna optimization. Samples a set of performance parameters,
    builds the llama-bench command, runs the benchmark, and returns the throughput metric.

    Parameters:
        trial (optuna.trial.Trial): The current Optuna trial object.
        metric (str): The performance metric to optimize ("tg", "pp", or "mean").
        repeat (int): Number of llama-bench repetitions for every trial; used to calculate robust <token/s> value
    Returns:
        float: The throughput value to maximize (tokens/sec).
    """
    # Sample params
    batch        = trial.suggest_int('batch', SEARCH_SPACE['batch_size']['low'], SEARCH_SPACE['high'])
    u_batch      = trial.suggest_int('u_batch', SEARCH_SPACE['ubatch_size']['low'], SEARCH_SPACE['high'])
    threads      = trial.suggest_int('threads', SEARCH_SPACE['threads']['low'], SEARCH_SPACE['threads']['high'])
    gpu_layers   = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])

    # ----------  constraint check [under development] -------------
    # llama.cpp usually requires batch_size >= ubatch_size; 
    # most user report lower performance if constrain is violated.  Prune such trials early.
    #if batch < u_batch:
    #    raise optuna.TrialPruned()    # skip invalid trial


    # Build llama-bench command 
    cmd = [
        llama_bench_path, # path to your llama-bench binary
        "--batch-size"     , str(batch),      # (-b  flag) (default 2024)
        "--ubatch-size"    , str(u_batch),    # (-ub flag) (default 512) 
        "--threads"        , str(threads),    # (-t  flag) (default 2)  
        "-ngl"             , str(gpu_layers), # (-ngl or --n-gpu-layers flag)
        "--flash-attn"     , str(flash),      # enables Flash Attention, a performance optimization during inference. 
        "--model"          , model_path,      # <--- change this or parametrize
        "-r"               , str(repeat),     # number of benchmark runs/repetitions for each configuration; mean value and std calculated from it 
        "-o"               , "csv"            # save temporary .csv file with llama-bench outputs
    ]
    # note1: memory mapping is now set by default. Instead, need to add --no-map flag. 
    # note2: use "-r 5" for more robust results (mean value calculated over 5 llama-bench runs); Use "-r 1" for quick assessment 

    # Add task-specific flags
    if metric in ("tg", "mean"):
        cmd += ["-n", "10"]  # tokens to generate (larger value improve final statistics, i.e. lower std in tok/s)
    if metric in ("pp", "mean"):
        cmd += ["-p", "10"]  # tokens to process (larger value improve final statistics, i.e. lower std in tok/s)

    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd, metric)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0
    # return 0.0 is OK for Optuna/bench scripts; 
    # i.e. this trial will be considered a failure but not fatal.


def objective_2(trial, metric, repeat, llama_bench_path, model_path, override_mode, batch, u_batch, threads, gpu_layers):
    """
    Objective function for Optuna scan over the entire categorical parameter space

    Extra parameters:
        override-tensor;
        batch, u_batch, threads, gpu_layers: are all fixed (best parameters from initial Trials_1) 
    
    Returns:
        float: The throughput value to maximize (tokens/sec).
    """
    # Sample params
    flash        = trial.suggest_categorical('flash', SEARCH_SPACE['flash_attn'])
    #flash_type  = trial.suggest_categorical('flash_type', SEARCH_SPACE['flash_attn_type'])

    # Build llama-bench command (can edit to add more flags)
    cmd = [
        llama_bench_path, # path to your llama-bench binary
        "--batch-size"     , str(batch),      # (-b flag) (default 2024)
        "--ubatch-size"    , str(u_batch),    # (-ub flag) (default 512) 
        "-threads"         , str(threads),    # (-t  flag) (default 2)  
        "-ngl"             , str(gpu_layers), # (-ngl or --n-gpu-layers flag)
        "--flash-attn"     , str(flash),      # enables Flash Attention, a performance optimization during inference. 
        #"--flash-attn-type", str(flash_type), # Metal + CUDA now have two flash-attention kernels (0 ≈ old GEMM, 1 = FMHA, 2 = in-place FMHA). 
        "--model"          , model_path,      # <--- change this or parametrize
        "-r"               , str(repeat),     # number of benchmark runs/repetitions for each configuration; mean value and std calculated from it 
        "-o"               , "csv"            # save temporary .csv file with llama-bench outputs
    ]
    # note1: memory mapping is now set by default. Instead, need to add --no-map flag. 
    # note2: use "-r 5" for more robust results (mean value calculated over 5 llama-bench runs); Use "-r 1" for quick assessment 
    #        e.g., launch tool with: python src/optimus.py --trials 30 -r 1 

    # Add task-specific flags
    if metric in ("tg", "mean"):
        cmd += ["-n", "10"]  # tokens to generate (larger value improve final statistics, i.e. lower std in tok/s)
    if metric in ("pp", "mean"):
        cmd += ["-p", "10"]  # tokens to process (larger value improve final statistics, i.e. lower std in tok/s)

    # include trials over --override-tensor only if "scan" is passes to args.override_tensor
    if override_mode == "scan":
        override_key = trial.suggest_categorical('override_pattern', list(OVERRIDE_PATTERNS.keys()))
        cmd += ["--override-tensor", OVERRIDE_PATTERNS[override_key]] # test few configuration[TBD]maybe run after last optimization  


    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd, metric)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def objective_3(trial, metric, repeat, llama_bench_path, model_path, override_tensor, flash_attn):
    """
    Objective function for Optuna optimization. 
    After we select promising '--override-tensor' and '--flash-attn'
    estimated over favorable conditions (best par from first Trials loop)
    we now run again over the numerical parameter space

    Parameters:
        trial (optuna.trial.Trial): The current Optuna trial object.
        metric (str): The performance metric to optimize ("tg", "pp", or "mean").
        repeat (int): Number of llama-bench repetitions for every trial; used to calculate robust <token/s> value
        overrive_tensor
        flash_attn
    Returns:
        float: The throughput value to maximize (tokens/sec).
    """
    # Sample params
    batch        = trial.suggest_int('batch', SEARCH_SPACE['batch_size']['low'], SEARCH_SPACE['high'])
    u_batch      = trial.suggest_int('u_batch', SEARCH_SPACE['ubatch_size']['low'], SEARCH_SPACE['high'])
    threads      = trial.suggest_int('threads', SEARCH_SPACE['threads']['low'], SEARCH_SPACE['threads']['high'])
    gpu_layers   = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])

    # Build llama-bench command 
    cmd = [
        llama_bench_path, # path to your llama-bench binary
        "--batch-size"     , str(batch),      # (-b  flag) (default 2024)
        "--ubatch-size"    , str(u_batch),    # (-ub flag) (default 512) 
        "--threads"        , str(threads),    # (-t  flag) (default 2)  
        "-ngl"             , str(gpu_layers), # (-ngl or --n-gpu-layers flag)
        "--flash-attn"     , str(flash_attn), # enables Flash Attention, a performance optimization during inference. 
        "override-tensor"  , str(overrive_tensor) # enable loading larger tensors to the CPU (saving VRAM from GPU)
        "--model"          , model_path,      # <--- change this or parametrize
        "-r"               , str(repeat),     # number of benchmark runs/repetitions for each configuration; mean value and std calculated from it 
        "-o"               , "csv"            # save temporary .csv file with llama-bench outputs
    ]

    # Add task-specific flags
    if metric in ("tg", "mean"):
        cmd += ["-n", "10"]  # tokens to generate (larger value improve final statistics, i.e. lower std in tok/s)
    if metric in ("pp", "mean"):
        cmd += ["-p", "10"]  # tokens to process (larger value improve final statistics, i.e. lower std in tok/s)

    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd, metric)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def run_optimization(n_trials, metric, repeat, llama_bench_path, model_path, llama_bin_path, override_mode):  
    """
    Run the Optuna optimization loop for a given number of trials, using the provided metric.
    At the end, print the best configuration and ready-to-use commands for llama-server/llama-bench.

    Given the large parameter space, the optimization runs in 3 stages. 
    - Stage 1: over the numerical space: 'gpu_layers', 'threads', 'batch' and 'ubatch' 
    - Stage 2: over the categorical space: 'override_tensor' and 'flash_attn'
    - Stage 3: with the best of previous config, run again over the numerical space. 

    Parameters:
        n_trials (int): Number of Optuna trials to perform. Default: 35.
        metric (str): Which throughput metric to optimize ("tg", "pp", or "mean"). Default: tg.
        ...[TBD]

    Returns:
        None
    """

    # TRIALS : stage_1
    sampler = TPESampler(multivariate=True)  # Others: "random": RandomSampler(); "cmaes": CmaEsSampler(),
    study_1 = optuna.create_study(direction="maximize", sampler=sampler)
    # use lambda to inject metric, repeat ...  
    study_1.optimize(lambda trial: objective_1(trial, metric, repeat, llama_bench_path, model_path), n_trials=n_trials)
    print("Best config Stage_1:", study_1.best_trial.params)
    print(f"Best Stage_1 {metric} tokens/sec:", study_1.best_value)

    # output_1 best llama.cpp parameters for Trials stage_1
    best_1 = study_1.best_trial.params


    # TRIALS : stage_2
    if override_mode == "scan": 
        n_override = len(OVERRIDE_PATTERNS)
        n_trials_2 = n_override * 2  # to cover all possibilities, since flash_attn: <0|1>
    else:
        n_trials_2 = 2 # since flash_attn: <0|1> 

    sampler = TPESampler()  # Others: "random": RandomSampler(); "cmaes": CmaEsSampler(),
    study_2 = optuna.create_study(direction="maximize", sampler=sampler)
    # use lambda to inject metric, repeat ...  
    study_2.optimize(lambda trial: objective_2(trial, metric, repeat, llama_bench_path, model_path, 
                                               best_1['override_mode'], best_1['batch'], best_1['u_batch'], 
                                               best_1['threads'], best_1['gpu_layers']), n_trials=n_trials_2)
    print("Best config Stage_2:", study_2.best_trial.params)
    print(f"Best Stage_2 {metric} tokens/sec:", study_2.best_value)

    # output_1 best llama.cpp parameters for Trials stage_1
    best_2 = study_2.best_trial.params


    # TRIALS : stage_3
    sampler = TPESampler(multivariate=True)  # Others: "random": RandomSampler(); "cmaes": CmaEsSampler(),
    study_3 = optuna.create_study(direction="maximize", sampler=sampler)
    # use lambda to inject metric, repeat ...  
    study_3.optimize(lambda trial: objective_3(trial, metric, repeat, llama_bench_path, model_path, 
                                               best_2['override_tensor'], best_2['flash_attn']), n_trials=n_trials)
    print("Best config Stage_3:", study_3.best_trial.params)
    print(f"Best Stage_3 {metric} tokens/sec:", study_3.best_value)

    # output_1 best llama.cpp parameters for Trials stage_1
    best_3 = study_3.best_trial.params

    ### END OF TRIALS ###

    print("\n# You are ready to run a local llama-server:")
    print("\n# llama-server (inference): listening at http://127.0.0.1:8080/ in your browser.")

    # 1. llama-server (inference); will be listening at http://127.0.0.1:8080/ in your browser. 
    llama_server_cmd = (
        f"/path_to/llama-server --model /path_to_model.gguf"
        f" -t {best_3['threads']}"
        f" --batch-size {best_3['batch']}"
        f" --ubatch-size {best_3['u_batch']}"
        f" -ngl {best_3['gpu_layers']}"
        f" --flash-attn {best_2['flash']}"
        f" --override-tensor" {best_2['override_tensor']}
        #f" --flash-attn-type {best['flash_type']}"
    )
    print("\n# For optimal inference, run:")
    print(f"\n {llama_server_cmd}")

    # 2. llama-bench (benchmark for both tg and pp)
    llama_bench_cmd = (
        f"/path_to/llama-bench"
        f" --model path_to_model.gguf"
        f" -t {best_3['threads']}"
        f" --batch-size {best_3['batch']}"
        f" --ubatch-size {best_3['u_batch']}"
        f" -ngl {best_3['gpu_layers']}"
        f" --flash-attn {best_2['flash']}"
        f" --override-tensor" {best_2['override_tensor']}
        f" -n 128 -p 128 -r 3 -o csv"
    )
    print("\n# To benchmark both generation and prompt processing speeds:")
    print(f"\n{llama_bench_cmd}")

