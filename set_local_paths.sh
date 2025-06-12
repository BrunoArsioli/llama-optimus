#!/usr/bin/env bash
#set -euo pipefail  #print error 

# ---- paths --------------------------------------------------
# export, to make it visible to you python script
export LLAMA_BIN="your_path_to/llama.cpp/build/bin"


# select model dir
MODEL_DIR="your_path_to_dir_where_you_store_your_ai_models/models"

# select model to use (e.g. in case you have many models, one in each folder)
MODEL_1_NAME="model1/Llama-4-Scout-17B-16E-Instruct-UD-Q5_K_XL-00001-of-00002.gguf"
MODEL_2_NAME="model4/gemma-3-27b-it-qat-UD-Q6_K_XL.gguf"
MODEL_3_NAME="model5/Qwen2.5-VL-32B-Instruct-UD-Q8_K_XL.gguf"

# model path (e.g. to model 3):
# export, to make it visible to your python script 
export MODEL_PATH="$MODEL_DIR/$MODEL_3_NAME"

# print paths 
echo "The following paths were set in this local env:"
echo "Path to llama.cpp/bin folder: $LLAMA_BIN "
echo "Poth to directory with all your AI models: $MODEL_DIR "
echo "Path to AI model: $MODEL_PATH "


