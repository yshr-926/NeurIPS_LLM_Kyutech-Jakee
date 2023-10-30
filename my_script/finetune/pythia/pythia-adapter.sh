#!/bin/bash

# config
MODEL_TYPE="pythia"
model_array=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")

# finetuning
for model_size in "${model_array[@]}"
do
    python3 finetune/adapter.py \
        --data_dir "data/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/dolly" \
        --checkpoint_dir "checkpoints/EleutherAI/${MODEL_TYPE}-${model_size}" \
        --out_dir out/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/dolly/adapter/compare1 \
        --precision "bf16-true"

    python3 finetune/adapter.py \
        --data_dir "data/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/lima" \
        --checkpoint_dir "checkpoints/EleutherAI/${MODEL_TYPE}-${model_size}" \
        --out_dir out/${MODEL_TYPE}/${MODEL_TYPE}-${model_size}/lima/adapter/compare1 \
        --precision "bf16-true"
done

# usage
# export CUDA_VISIBLE_DEVICES=
# nohup bash pythia-adapter.sh &