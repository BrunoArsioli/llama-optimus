# llama_optimus/cli.,py
# handle parsing, validation, and env setup

import argparse, os, sys
from .core import run_optimization, estimate_max_ngl, SEARCH_SPACE

from llama_optimus import __version__

# count number of available cpu cores
max_threads = os.cpu_count()

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
    parser.add_argument("--metric", type=str, default="tg", choices=["tg", "pp", "mean"], help="Which throughput metric to optimize: 'tg' (token generation, default), 'pp' (prompt processing), or 'mean' (average of both)")
    parser.add_argument("--ngl-max",type=int, help="Maximum number of model layers for -ngl (skip estimation if provided; estimation runs by default).")
    parser.add_argument("--repeat", "-r", type=int, default=2, help="Number of llama-bench runs per configuration (higher = more robust, lower = faster; default: 2, for quick assessement: 1)")
    #parser.add_argument('--version', "-v", action='version', version='llama-optimus v0.1.0')
    parser.add_argument('--version', "-v", action='version', version=f'llama-optimus v{__version__}')
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

    print(f"Number of CPUs: {max_threads}.")
    print(f"Path to 'llama-bench':{llama_bench_path}")  # in llama.cpp/tools/
    print(f"Path to 'model.gguf' file:{model_path}")


    # default: estimate maximum number of layers before run_optimization 
    # in case the user knows ngl_max value, skip ngl_max estimate
    if args.ngl_max is not None: 
        SEARCH_SPACE['gpu_layers']['high'] = args.ngl_max
        print(f"User-specified maximum -ngl set to {args.ngl_max}")
    else:
        SEARCH_SPACE['gpu_layers']['high'] = estimate_max_ngl(
            llama_bench_path=llama_bench_path, model_path=model_path, 
            min_ngl=0, max_ngl=SEARCH_SPACE['gpu_layers']['high'])
        print(f"Setting maximum -ngl to {SEARCH_SPACE['gpu_layers']['high']}")

    run_optimization(n_trials=args.trials, metric=args.metric, 
                     repeat=args.repeat, llama_bench_path=llama_bench_path, 
                     model_path=model_path, llama_bin_path=llama_bin_path)  

if __name__ == "__main__":

    main()