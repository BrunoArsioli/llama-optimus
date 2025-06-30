# llama_optimus/cli.py
# handle parsing, validation, and env setup

import argparse, os, sys
import platform 
from pathlib import Path   
from .core import run_optimization, estimate_max_ngl, warmup_until_stable
from .override_patterns import OVERRIDE_PATTERNS   
from .search_space import SEARCH_SPACE, max_threads 

from llama_optimus import __version__


def main():
    parser = argparse.ArgumentParser(
        description="llama-optimus: Benchmark & tune llama.cpp.",
        epilog="""
        Example usage:

            llama-optimus --llama-bin my_path_to/llama.cpp/build/bin --model my_path_to/models/my-model.gguf --trials 35 --metric tg
            
        for a quick test (set a single Optuna trial and a single repetition of llama-bench):
            
            llama-optimus --llama-bin my_path_to/llama.cpp/build/bin --model my_path_to/models/my-model.gguf --trials 1 -r 1 --metric tg
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument("--trials", type=int, default=45, help="Number of Optuna/optimization trials")
    parser.add_argument("--model", type=str, help="Path to model (overrides env var)")
    parser.add_argument("--llama-bin", type=str, help="Path to llama.cpp build/bin folder (overrides env var)")

    parser.add_argument("--metric", type=str, default="tg", choices=["tg", "pp", "mean"], help="Which throughput " \
        "metric to optimize: 'tg' (token generation, default), 'pp' (prompt processing), or 'mean' (average of both)")

    parser.add_argument("--ngl-max",type=int, help="Maximum number of model layers for -ngl "
        "(skip estimation if provided; estimation runs by default).")

    parser.add_argument("--repeat", "-r", type=int, default=3, help="Number of llama-bench runs per configuration "
        "(higher = more robust, lower = faster; for quick assessment: 1)")

    parser.add_argument("--n-tokens", type=int, default=192, help="Number of tokens used in llama-bench to test " \
        "velocity of prompt processing and text generation. Keep in mind there is large variability in tok/s outputs. " \
        "If n_tokens is too low, uncertainty takes over, optimization may suffer. Still, if you need to lower it, " \
        "try to operate with n_tokens > 70 and --repeat 3. " \
        "For fast exploration/testing/debug: --n-tokens 10 --repeat 2 is fine")
    
    parser.add_argument("--n-warmup-tokens", "-nwt", type=int, default=128, help="Number of tokens passed to " \
        "llama-bench during each warmup loop. In case of large models (and you getting small tg tokens/s), "
        "if n_warmup_tokens is too large, it can happen that you warmup in the first warmup cycle, and you end " \
        "up not detecting the warmup. ")
    
    parser.add_argument("--n-warmup-runs", type=int, default=35, help="Maximum warm-up iterations before trials " \
    "begin. To skip warm-up completely, use the --no-warmup flag; Otherwise, there will be a minimum " \
    "number of warmup runs, which is set with `min_runs=3` in core function definition")

    parser.add_argument("--no-warmup", action="store_true", help="Skip the initial system warmup phase before " \
    "optimization (for debugging/testing).")

    #parser.add_argument('--version', "-v", action='version', version='llama-optimus v0.1.0')
    parser.add_argument("--version", "-v", action='version', version=f'llama-optimus v{__version__}')

    parser.add_argument("--override-mode", type=str, default="scan", choices=["none", "scan", "custom"],
    help=f"'none': do not scan this parameter; scan: 'scan' over preset override-tensor patterns; " \
    f"'custom': (future) user provides their own pattern(s). Available override patterns: {OVERRIDE_PATTERNS.keys()}" )
    
    args = parser.parse_args()

    # Set paths based on CLI flags, env vars, or prompt user to provide it
    # Resolve llama_bin_path
    llama_bin_path = (args.llama_bin or os.environ.get("LLAMA_BIN")
        or input("Please, provide the path to your 'llama.cpp/build/bin' ").strip() )

    # Check the operating system and build llama_bench_path
    if platform.system() == "Windows":
        llama_bench_path = f"{llama_bin_path}/Release/llama-bench.exe"
        # Sanity-check
        if not Path(llama_bench_path).is_file():
            sys.exit(f"ERROR: llama-bench.exe not found at {llama_bench_path}")
    else:
        llama_bench_path = f"{llama_bin_path}/llama-bench"
        # Sanity-check
        if not Path(llama_bench_path).is_file():
            sys.exit(f"ERROR: llama-bench not found at {llama_bench_path}")


    # Resolve model_path
    model_path = (args.model or os.environ.get("MODEL_PATH")
        or input("Please, provide the path to your 'ai_model.gguf' ").strip() )

    # Quick check if paths are set. ERROR msg if None or empty.
    if not llama_bin_path or not model_path:
        print("ERROR: LLAMA_BIN or MODEL_PATH not set. Set via environment variable, " \
        "pass via CLI flags, or provide paths just after launching llama-optimus. " \
        "Go to your terminal, navigate to your_path_to/llama.cpp/buil/bin and type 'pwd' to resolve the entire path. " \
        "Go to your terminal, navigate to your_path_to_AI_models/ and type 'pwd' to resolve the path. " \
        "Note: you must pass /path_to_model/model_name.gguf; e.g. your_path_model/gemma3_12B.gguf .", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(llama_bench_path):
        print(f"ERROR: llama-bench not found at {llama_bench_path}. ...", file=sys.stderr)
        sys.exit(1)

    print("")
    print("#################")
    print("# LLAMA-OPTIMUS #")
    print("#################")

    print("")
    print(f"Number of CPUs: {max_threads}.")
    print(f"Path to 'llama-bench':{llama_bench_path}")  # in llama.cpp/tools/
    print(f"Path to 'model.gguf' file:{model_path}")
    print("")

    # default: estimate maximum number of layers before run_optimization 
    # in case the user knows ngl_max value, skip ngl_max estimate
    if args.ngl_max is not None: 
        SEARCH_SPACE['gpu_layers']['high'] = args.ngl_max
        print("")
        print(f"User-specified maximum -ngl set to {args.ngl_max}")
        print("")
    else:
        print("")
        print("########################################################################")
        print("# Find maximum number of model layers that can be written to your VRAM #")
        print("########################################################################")
        print("")

        SEARCH_SPACE['gpu_layers']['high'] = estimate_max_ngl(
            llama_bench_path=llama_bench_path, model_path=model_path, 
            min_ngl=0, max_ngl=SEARCH_SPACE['gpu_layers']['high'])
        print("")
        print(f"Setting maximum -ngl to {SEARCH_SPACE['gpu_layers']['high']}")
        print("")

    # system warm-up before optimization
    max_ngl_wup=SEARCH_SPACE['gpu_layers']['high']
    
    if args.no_warmup:
        print("")
        print("#####################################################")
        print("# !!!Optimization running without system warmup!!!  #")
        print("#####################################################")
        print("")
    else: 
        print("")
        print("#######################")
        print("# Starting warmup...  #")
        print("#######################")
        print("")

        # in case n_warmup_runs is set to < 4, warn about the minimum number of warmup runs
        if args.n_warmup_runs < 4:
            print("")
            print("#########################################################################")
            print("# Setting a minimum of 4 warmup runs.                                   #")
            print('# For no warmup, pass the --no-warmup flag during llama-optimus launch  #')
            print("#########################################################################")
            print("")

        # launch warmup
        warmup_until_stable(llama_bench_path=llama_bench_path, model_path=model_path, metric=args.metric, 
                            ngl=max_ngl_wup, min_runs=4, n_warmup_runs=args.n_warmup_runs,
                            n_warmup_tokens=args.n_warmup_tokens, max_threads=max_threads)

    print("")
    print("##################################")
    print("# Starting Optimization Loop...  #")
    print("##################################")
    print("")

    run_optimization(n_trials=args.trials, n_tokens=args.n_tokens, metric=args.metric, 
                     repeat=args.repeat, llama_bench_path=llama_bench_path, 
                     model_path=model_path, llama_bin_path=llama_bin_path, override_mode=args.override_mode)  

if __name__ == "__main__":

    main()
    