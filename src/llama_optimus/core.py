# core.py
# Core functions for llama-optimus optimization

import subprocess
import re
import optuna
import os
import shutil
import pandas as pd 
import tempfile
from optuna.samplers import TPESampler
from optuna.samplers import GridSampler
from .override_patterns import OVERRIDE_PATTERNS   
from .search_space import SEARCH_SPACE, max_threads 

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
    
    # debug 
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


def objective_1(trial, n_tokens, metric, repeat, llama_bench_path, model_path):
    """
    Objective function for Optuna optimization. Samples a set of performance parameters,
    builds the llama-bench command, runs the benchmark, and returns the throughput metric.

    Parameters:
        trial (optuna.trial.Trial): The current Optuna trial object.
        n_tokens (int): the number of tokens used in pp and tg benchmark 
        metric (str): The performance metric to optimize ("tg", "pp", or "mean").
        repeat (int): Number of llama-bench repetitions for every trial; used to calculate robust <token/s> value
    Returns:
        float: The throughput value to maximize (tokens/sec).
    """
    # Sample params
    batch        = trial.suggest_int('batch', SEARCH_SPACE['batch_size']['low'], SEARCH_SPACE['batch_size']['high'])
    u_batch      = trial.suggest_int('u_batch', SEARCH_SPACE['ubatch_size']['low'], SEARCH_SPACE['ubatch_size']['high'])
    threads      = trial.suggest_int('threads', SEARCH_SPACE['threads']['low'], SEARCH_SPACE['threads']['high'])
    gpu_layers   = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])

    # ----------  constraint check [under development] -------------
    # llama.cpp usually requires batch_size >= ubatch_size; 
    # most user report lower performance if constrain is violated.  Prune such trials early.
    #if batch < u_batch:
    #    raise optuna.TrialPruned()    # skip invalid trial


    # Build llama-bench command 
    cmd_1 = [
        llama_bench_path, # path to your llama-bench binary
        #"--no-warmup"      ,                 # disable warm-up. alredy warmed-up in llama-optimus launch; [TBD in llama.cpp]
        "--batch-size"     , str(batch),      # (-b  flag) (default 2024)
        "--ubatch-size"    , str(u_batch),    # (-ub flag) (default 512) 
        "--threads"        , str(threads),    # (-t  flag) (default 2)  
        "-ngl"             , str(gpu_layers), # (-ngl or --n-gpu-layers flag)
        "--model"          , model_path,      # 
        "-r"               , str(repeat),     # number of benchmark runs/repetitions for each configuration; mean value and std calculated from it 
        "-o"               , "csv",           # save temporary .csv file with llama-bench outputs
        "--progress"
    ]
    # note1: memory mapping is now set by default. Instead, need to add --no-map flag. 
    # note2: use "-r 5" for more robust results (mean value calculated over 5 llama-bench runs); Use "-r 1" for quick assessment 

    # Add task-specific flags
    if metric in ("tg", "mean"):
        cmd_1 += ["-n", str(n_tokens)]  # tokens to generate (larger value improve final statistics, i.e. lower std in tok/s)
    if metric in ("pp", "mean"):
        cmd_1 += ["-p", str(n_tokens)]  # tokens to process (larger value improve final statistics, i.e. lower std in tok/s)

    # debug
    print("")
    print(f"cmd_1: {cmd_1}")
    print("")
    
    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd_1, metric)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0
    # return 0.0 is OK for Optuna/bench scripts; 
    # i.e. this trial will be considered a failure but not fatal.


def objective_2(trial, n_tokens, metric, repeat, llama_bench_path, model_path, override_mode, batch, u_batch, threads, gpu_layers):
    """
    Objective function for Optuna scan over the entire categorical parameter space

    Extra parameters:
        override-tensor;
        batch, u_batch, threads, gpu_layers: are all fixed (best parameters from initial Trials_1) 
    
    Returns:
        float: The throughput value to maximize (tokens/sec).
    """
    # for degug
    print(f"Running objective_2 with batch={batch}, u_batch={u_batch}, threads={threads}, gpu_layers={gpu_layers}")


    # Build llama-bench command (can edit to add more flags)
    cmd_2 = [
        llama_bench_path, # path to your llama-bench binary
        "--batch-size"     , str(batch),      # (-b flag) (default 2024)
        "--ubatch-size"    , str(u_batch),    # (-ub flag) (default 512) 
        "--threads"        , str(threads),    # (-t  flag) (default 2)  
        "-ngl"             , str(gpu_layers), # (-ngl or --n-gpu-layers flag)
        #"--flash-attn"     , str(flash_attn), # enables Flash Attention, a performance optimization during inference. 
        #"--flash-attn-type", str(flash_type),# Metal + CUDA now have two flash-attention kernels (0 ≈ old GEMM, 1 = FMHA, 2 = in-place FMHA). 
        "--model"          , model_path,      # 
        "-r"               , str(repeat),     # number of benchmark runs/repetitions for each configuration; mean value and std calculated from it 
        "-o"               , "csv"            # save temporary .csv file with llama-bench outputs
    ]
    # note1: memory mapping is now set by default. Instead, need to add --no-map flag. 
    # note2: use "-r 5" for more robust results (mean value calculated over 5 llama-bench runs); Use "-r 1" for quick assessment 
    #        e.g., launch tool with: python src/optimus.py --trials 30 -r 1 

    # Add task-specific flags
    if metric in ("tg", "mean"):
        cmd_2 += ["-n", str(n_tokens)]  # tokens to generate (larger value improve final statistics, i.e. lower std in tok/s)
    if metric in ("pp", "mean"):
        cmd_2 += ["-p", str(n_tokens)]  # tokens to process (larger value improve final statistics, i.e. lower std in tok/s)


    # remove flash-attn flag in case --flash-attn is 0 ; avoid possible misbehaviour in case `--flash-attn 0  != "" `
    flash_attn   = trial.suggest_categorical('flash_attn', SEARCH_SPACE['flash_attn'])
    if flash_attn == 1:  # in case of "0" option, do not pass the --flash-attn flag 
        cmd_2 += ["--flash-attn", str(flash_attn)] # test few configuration[TBD]maybe run after last optimization  

    # include trials over --override-tensor only if "scan" is passes to args.override_tensor
    # and, in case override_key == "none", the override-tensor flag is not inserted in cmd_2
    if override_mode == "scan":
        override_key = trial.suggest_categorical('override_tensor', list(OVERRIDE_PATTERNS.keys()))
        if override_key != "none":  # in case of "none" option, do not pass the no --override-tensor flag 
            cmd_2 += ["--override-tensor", OVERRIDE_PATTERNS[override_key]] # test few configuration[TBD]maybe run after last optimization  

    # whe we need the last two? It seems llama.cpp is treating --flash-attn 0 and --override-tensor ""
    # in different way it treats cases where those flags are simply not given.
    # [TBD] we must investigate this
    # for now, set these rules to guarantee a case with no flags is tried as well 

    # debug 
    print("")
    print(f"cmd_2: {cmd_2} ")
    print("")

    # Best config Stage_1: {'batch': 4809, 'u_batch': 1161, 'threads': 10, 'gpu_layers': 149}
    # Best Stage_1 tg tokens/sec: 159.785726

    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd_2, metric)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def objective_3(trial, n_tokens, metric, repeat, llama_bench_path, model_path, override_pattern, flash_attn, override_mode):
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
    batch        = trial.suggest_int('batch', SEARCH_SPACE['batch_size']['low'], SEARCH_SPACE['batch_size']['high'])
    u_batch      = trial.suggest_int('u_batch', SEARCH_SPACE['ubatch_size']['low'], SEARCH_SPACE['ubatch_size']['high'])
    threads      = trial.suggest_int('threads', SEARCH_SPACE['threads']['low'], SEARCH_SPACE['threads']['high'])
    gpu_layers   = trial.suggest_int('gpu_layers', SEARCH_SPACE['gpu_layers']['low'], SEARCH_SPACE['gpu_layers']['high'])

    # Build llama-bench command 
    cmd_3 = [
        llama_bench_path, # path to your llama-bench binary
        "--batch-size"     , str(batch),      # (-b  flag) (default 2024)
        "--ubatch-size"    , str(u_batch),    # (-ub flag) (default 512) 
        "--threads"        , str(threads),    # (-t  flag) (default 2)  
        "-ngl"             , str(gpu_layers), # (-ngl or --n-gpu-layers flag)
        #"--flash-attn"     , str(flash_attn), # enables Flash Attention, a performance optimization during inference. 
        #"--override-tensor", str(override_pattern),# enable loading larger tensors to the CPU (saving VRAM from GPU)
        "--model"          , model_path,      # 
        "-r"               , str(repeat),     # number of benchmark runs/repetitions for each configuration; mean value and std calculated from it 
        "-o"               , "csv"            # save temporary .csv file with llama-bench outputs
    ]

    # Add task-specific flags
    if metric in ("tg", "mean"):
        cmd_3 += ["-n", str(n_tokens)]  # tokens to generate (larger value improve final statistics, i.e. lower std in tok/s)
    if metric in ("pp", "mean"):
        cmd_3 += ["-p", str(n_tokens)]  # tokens to process (larger value improve final statistics, i.e. lower std in tok/s)

    # remove flash-attn flag in case --flash-attn is 0 ; avoid possible misbehaviour in case `--flash-attn 0  != "" `
    flash_attn   = trial.suggest_categorical('flash_attn', SEARCH_SPACE['flash_attn'])
    if flash_attn == 1:  # in case of "0" option, do not pass the --flash-attn flag 
        cmd_3 += ["--flash-attn", str(flash_attn)] # test few configuration[TBD]maybe run after last optimization  

    # include trials over --override-tensor only if "scan" is passes to args.override_tensor
    # in case override_key == "none", the override-tensor flag is not inserted in cmd_3
    if override_mode == "scan":
        override_key = trial.suggest_categorical('override_tensor', list(OVERRIDE_PATTERNS.keys()))
        if override_key != "none":  # in case of "none" option, do not pass the no --override-tensor flag 
            cmd_3 += ["--override-tensor", OVERRIDE_PATTERNS[override_key]] # test few configuration[TBD]maybe run after last optimization  



    # debug
    print("")
    print(f"cmd_3: {cmd_3}")
    print("")

    try:
        tokens_per_sec = run_llama_bench_with_csv(cmd_3, metric)
        return tokens_per_sec    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def warmup_until_stable(llama_bench_path, model_path, metric, ngl, threshold, min_runs, max_warmup,n_warmup_tokens):
    """
    Warm-up doctrine:
    - Always run at least 3 warmup cycles before checking for stability.
    - For early runs (i < 3), just collect data—do not compare yet.
    - From the 4th run onward, compare the current value to the value from 3 runs before.
      This catches slow, stepped drops caused by mid-benchmark thermal or system events.
    - If performance drops by more than `threshold` (e.g., 10%) compared to 3 runs prior,
      increment a `drops` counter.
    - If `drops` reaches `min_runs`, declare system warmed up and ready.
    - This doctrine ensures we don’t “exit early” on temporary or noisy values, and are
      able to handle gradual & small performance drops (i.e. when permance drops happen
      in the midle of the workload of a 'in loop' llama-bench runs).
    """

    history = []
    drops = 0

    # build cmd warm up 
    cmd_wup = [
        llama_bench_path,
        "-ngl", str(ngl),
        "--model", model_path,
        "-r", "2",       # benchmark repetitions
        "-n", str(n_warmup_tokens),
        "-p", str(n_warmup_tokens), 
        "-o", "csv"
    ]

    for i in range(max_warmup):
        performance = run_llama_bench_with_csv(cmd_wup, metric)
        history.append(performance)
        print(f"Warmup {i+1}: {performance:.2f} tok/s")
        
        if i >= 3:  # Only compare after the third run
            compare_perf = history[i-3]
            if performance < compare_perf * (1 - threshold):
                drops += 1
            else:
                drops = 0
            if drops >= 2: 

                print("")
                print("##########################################")
                print("# Consistent Performance drop detected.  #")
                print("# Your system warmed up.                 #") 
                print("# Ready to go!                           #")
                print("##########################################")
                print("")

                break
        # For i < 2: do not compare, just accumulate

        print("")
        print("Warmup performance history:", history)
        print("")

    return history


def run_optimization(n_trials, n_tokens, metric, repeat, llama_bench_path, model_path, llama_bin_path, override_mode):  
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

    # outpus
    print("")
    print("############################################################")
    print("# First stage: Initial exploration of parameter space      #")
    print("############################################################")
    print("")

    # TRIALS : stage_1
    sampler = TPESampler(multivariate=True)  # Others: "random": RandomSampler(); "cmaes": CmaEsSampler(),
    study_1 = optuna.create_study(direction="maximize", sampler=sampler)
    # use lambda to inject metric, repeat ...  
    study_1.optimize(lambda trial: objective_1(trial, n_tokens, metric, repeat, llama_bench_path, model_path), n_trials=n_trials)
    print("")
    print("Best config Stage_1:", study_1.best_trial.params) 
    print(f"Best Stage_1 {metric} tokens/sec:", study_1.best_value)
    print("")

    # Output: Best llama.cpp parameters from Stage 1 trials
    best_1 = study_1.best_trial.params

    # outpus
    print("")
    print("############################################################")
    print("# Second stage: Grid search over categorical parameters    #")
    print("############################################################")
    print("")


    # TRIALS : stage_2
    if override_mode == "scan": 
        n_override = len(OVERRIDE_PATTERNS)  # 
        n_trials_2 = n_override * 2  # to cover all possibilities, since flash_attn: <0|1>
        
        # define grid space
        search2 = {'flash_attn': SEARCH_SPACE['flash_attn'],
                   'override_tensor': SEARCH_SPACE['override_spc']}    
    else:
        n_trials_2 = 2 # since flash_attn: <0|1> 
        search2 = {'flash_attn': SEARCH_SPACE['flash_attn']} 

    # in this case, use grid sampler
    sampler_2 = optuna.samplers.GridSampler(search2)
    study_2 = optuna.create_study(direction="maximize", sampler=sampler_2)
    # use lambda to inject metric, repeat ...  
    study_2.optimize(lambda trial: objective_2(trial, n_tokens, metric, repeat, llama_bench_path, model_path, 
                                               override_mode, best_1['batch'], best_1['u_batch'], 
                                               best_1['threads'], best_1['gpu_layers']), n_trials=n_trials_2)
    print("")
    print("Best config Stage_2:", study_2.best_trial.params)
    print(f"Best Stage_2 {metric} tokens/sec:", study_2.best_value)
    print("")

    # Output: Best llama.cpp parameters from Stage 2 trials
    best_2 = study_2.best_trial.params

    # in case --override-tensor none, pass ""
    if 'override_tensor' not in best_2:
        best_2['override_tensor'] = "none"

    # outpus
    print("")
    print("#######################################")
    print("# Third stage: Finetune final config  #")
    print("#######################################")
    print("")

    # TRIALS : stage_3
    sampler_3 = TPESampler(multivariate=True)  # Others: "random": RandomSampler(); "cmaes": CmaEsSampler(),
    study_3 = optuna.create_study(direction="maximize", sampler=sampler_3)
    # use lambda to inject metric, repeat ...  
    study_3.optimize(lambda trial: objective_3(trial, n_tokens, metric, repeat, llama_bench_path, model_path, 
                                               best_2['override_tensor'], best_2['flash_attn'], override_mode), n_trials=n_trials)
    print("")
    print("Best config Stage_3:", study_3.best_trial.params)
    print(f"Best Stage_3 {metric} tokens/sec:", study_3.best_value)
    print("")

    # Output: Best llama.cpp parameters from Stage 3 trials
    best_3 = study_3.best_trial.params

    ### END OF TRIALS ###

    print("")
    print("You are ready to run a local llama-server:")
    print("If you launch llama-server, it will be listening at http://127.0.0.1:8080/ in your browser.")
    print("")

    # 1. llama-server (inference); will be listening at http://127.0.0.1:8080/ in your browser. 
    llama_server_cmd = (
        f"/path_to/llama-server --model /path_to_model.gguf" 
        f" -t {best_3['threads']}"
        f" --batch-size {best_3['batch']}"
        f" --ubatch-size {best_3['u_batch']}"
        f" -ngl {best_3['gpu_layers']}"
        #f" --flash-attn-type {best['flash_type']}"
    )

    if best_2['override_tensor'] != "none":
        llama_server_cmd += f'  --override-tensor "{OVERRIDE_PATTERNS[best_2["override_tensor"]]}" '  # only add if --override-tensor key is != "none" 

    # for llama-server, --flash-att is of 'action' type (i.e. do not accept <0|1> values).
    if best_2['flash_attn'] == 1:
        llama_server_cmd += f" --flash-attn "    

    print("")
    print("# For optimal inference, run:")
    print(f"{llama_server_cmd}")
    print("")

    # 2. llama-bench (benchmark for both tg and pp)
    llama_bench_cmd = (
        f"/path_to/llama-bench"
        f" --model path_to_model.gguf"
        f" -t {best_3['threads']}"
        f" --batch-size {best_3['batch']}"
        f" --ubatch-size {best_3['u_batch']}"
        f" -ngl {best_3['gpu_layers']}"
        f" --flash-attn {best_2['flash_attn']}"  # in llama-server, --flash-attn is type 'int', accepts <0|1> values.
        #f" --override-tensor {OVERRIDE_PATTERNS[best_2['override_tensor']]}"
        f" -n 128 -p 128 -r 5 --progress "
    )

    if best_2['override_tensor'] != "none":
        llama_bench_cmd += f' --override-tensor "{OVERRIDE_PATTERNS[best_2["override_tensor"]]}" ' # concatenate string if --override-tensor key is != "none" 

    print("")
    print("# To benchmark both generation and prompt processing speeds:")
    print(f"{llama_bench_cmd}")
    print("")


