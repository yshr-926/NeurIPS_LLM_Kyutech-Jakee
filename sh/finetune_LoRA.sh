#!/bin/bash

declare -A CHECKPOINT_DIR=(
    ["falcon"]="tiiuae"
    ["pythia"]="EleutherAI"
    ["llama"]="meta-llama"
    ["open-llama"]="openlm-research"
)

today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

models=('Llama-2-13b-hf')
datasets=('lima')
finetunes=('lora_swa_r8a16')
optimizers=('AdamW')
quantize='bnb.nf4-dq'

# Hyperparameters
max_iters=('50000')
batch_sizes=('128')
micro_batch_sizes=('1')
learning_rates=('0.0003')
weight_decays=('0.001')
# lr_types=('CosineAnnealingLR')
lr_types=('Fix')
warmup_steps='100'

# LoRA Hyperparameters
lora_r=8
lora_alpha=16
lora_dropout=0.05
lora_query=true
lora_key=false
lora_value=true
lora_projection=false
lora_mlp=false
lora_head=false

