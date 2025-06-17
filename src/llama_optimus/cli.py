# llama_optimus/cli.,py
# handle parsing, validation, and env setup

import argparse, os, sys
from .core import run_optimization, estimate_max_ngl, SEARCH_SPACE, warmup_until_stable
from .override_patterns import OVERRIDE_PATTERNS   
from .search_space import SEARCH_SPACE, max_threads 

from llama_optimus import __version__

# count number of available cpu cores
#max_threads = os.cpu_count()


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

    parser.add_argument("--repeat", "-r", type=int, default=2, help="Number of llama-bench runs per configuration "
        "(higher = more robust, lower = faster; default: 2, for quick assessment: 1)")

    parser.add_argument("--n-tokens", type=int, default=128, help="Number of tokens used in llama-bench to test " \
        "velocity of prompt processing and text generation. Keep in mind there is large variability in tok/s outputs. " \
        "If n_tokens is too low, uncertainty takes over, optimization may suffer. Still, if you need to lower it, " \
        "try to operate with n_tokens > 70 and --repeat 3. " \
        "For fast exploration/testing/debug: --n-tokens 10 --repeat 2 is fine")
    
    parser.add_argument("--n-warmup-tokens", "-nwt", type=int, default=128, help="Number of tokens passed to " \
        "llama-bench during each warmup loop. In case of large models (and you getting small tg tokens/s), "
        "if n_warmup_tokens is too large, it can happen that you warmup in the first warmup cycle, and you end " \
        "up not detecting the warmup. ")
    
    parser.add_argument("--no-warmup", action="store_true", help="Skip the initial system warmup phase before " \
    "optimization (for debugging/testing).")

    #parser.add_argument('--version', "-v", action='version', version='llama-optimus v0.1.0')
    parser.add_argument("--version", "-v", action='version', version=f'llama-optimus v{__version__}')

    parser.add_argument("--override-mode", type=str, default="scan", choices=["none", "scan", "custom"],
    help=f"'none': do not scan this parameter; scan: 'scan' over preset override-tensor patterns; " \
    f"'custom': (future) user provides their own pattern(s). Available override patterns: {OVERRIDE_PATTERNS.keys()}" )
    
    args = parser.parse_args()

    # Set paths based on CLI flags or env vars
    llama_bin_path = args.llama_bin if args.llama_bin else os.environ.get("LLAMA_BIN")
    llama_bench_path = f"{llama_bin_path}/llama-bench"
    model_path = args.model if args.model else os.environ.get("MODEL_PATH")

    if llama_bin_path is None or model_path is None:
        print("ERROR: LLAMA_BIN or MODEL_PATH not set. Set via environment variable or pass via CLI flags.", file=sys.stderr)
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
        print("##############################################")
        print("# !!!Optimization running without warmup!!!  #")
        print("##############################################")
        print("")
    else: 
        print("")
        print("#######################")
        print("# Starting warmup...  #")
        print("#######################")
        print("")
        warmup_until_stable(llama_bench_path=llama_bench_path, model_path=model_path, metric=args.metric, 
                            ngl=max_ngl_wup, threshold=0.06, min_runs=3, max_warmup=30,n_warmup_tokens=args.n_warmup_tokens)

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