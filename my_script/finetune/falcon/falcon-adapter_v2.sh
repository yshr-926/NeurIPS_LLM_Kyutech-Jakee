#!/bin/bash

#TODO memory不足

# config
MODEL_TYPE="falcon"
model_array=("7b" "40b" "180B")

# finetuning
for model_size in "${model_array[@]}"
do
    python3 finetune/adapter_v2.py \
        --data_dir "data/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/dolly" \
        --checkpoint_dir "checkpoints/tiiuae/${MODEL_TYPE}-${model_size}" \
        --out_dir out/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/dolly/adapter_v2/compare1 \
        --precision "bf16-true"

    python3 finetune/adapter_v2.py \
        --data_dir "data/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/lima" \
        --checkpoint_dir "checkpoints/tiiuae/${MODEL_TYPE}-${model_size}" \
        --out_dir out/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/lima/adapter_v2/compare1 \
        --precision "bf16-true"
done


# usage
# export CUDA_VISIBLE_DEVICES=
# nohup bash falcon-adapter_v2.sh &