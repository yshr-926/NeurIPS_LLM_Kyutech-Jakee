#!/bin/bash

declare -A CHECKPOINT_DIR=(
    ["falcon"]="tiiuae"
    ["pythia"]="EleutherAI"
    ["llama"]="meta-llama"
    ["open-llama"]="openlm-research"
)

today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

base='open-llama'
models=('open_llama_3b')
datasets=('lima')
finetunes=('lora_swa')
optimizers=('AdamW')
quantize='bnb.nf4-dq'

# Hyperparameters
max_iters=('500')
batch_sizes=('128')
micro_batch_sizes=('1')
learning_rates=('0.0003')
weight_decays=('0.001')
# lr_types=('LinearWarmupCosineAnnealingLR' 'CosineAnnealingLR')
lr_types=('Fix')
warmup_steps='100'
eta_min='0.0'

# LoRA Hyperparameters
lora_r='8'
lora_alpha='16'
lora_dropout='0.05'
lora_query=true
lora_key=false
lora_value=true
lora_projection=false
lora_mlp=false
lora_head=false

log_interval='100'
eval_interval='50'
save_interval='100'
eval_iters='100'
eval_max_new_tokens='100'

upload_flag=true
# huggingface token "write"
hf_token='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'
repo_dir='miz22'

for dataset in ${datasets[@]}
do
    for finetune in ${finetunes[@]}
    do
        for model in ${models[@]}
        do
            for optimizer in ${optimizers[@]}
            do
                for batch_size in ${batch_sizes[@]}
                do
                    for micro_batch_size in ${micro_batch_sizes[@]}
                    do
                        for learning_rate in ${learning_rates[@]}
                        do
                            for weight_decay in ${weight_decays[@]}
                            do
                                for lr_type in ${lr_types[@]}
                                do
                                    for max_iter in ${max_iters[@]}
                                    do
                                        mkdir -p logs/$base/$model/$dataset/"$finetune"_r"$lora_r"a"$lora_alpha"/$quantize/"$optimizer"/"$max_iter"_"$batch_size"_"$micro_batch_size"/"$learning_rate"_"$weight_decay"/"$lr_type"
                                        python finetune/$finetune.py \
                                            --data_dir data/$base/$model/$dataset \
                                            --checkpoint_dir checkpoints/${CHECKPOINT_DIR["${base}"]}/$model \
                                            --out_dir out/$base/$model/$dataset/"$finetune"_r"$lora_r"a"$lora_alpha"/$quantize/"$optimizer"/"$max_iter"_"$batch_size"_"$micro_batch_size"/"$learning_rate"_"$weight_decay"/"$lr_type"/"$today" \
                                            --precision "bf16-true" \
                                            --quantize $quantize \
                                            --optim_name $optimizer \
                                            --max_iters $max_iter \
                                            --log_interval $log_interval \
                                            --batch_size $batch_size \
                                            --micro_batch_size $micro_batch_size \
                                            --learning_rate $learning_rate \
                                            --weight_decay $weight_decay \
                                            --warmup_steps $warmup_steps \
                                            --eta_min $eta_min \
                                            --lr_type $lr_type \
                                            --lora_r $lora_r \
                                            --lora_alpha $lora_alpha \
                                            --lora_dropout $lora_dropout \
                                            --lora_query $lora_query \
                                            --lora_key $lora_key \
                                            --lora_value $lora_value \
                                            --lora_projection $lora_projection \
                                            --lora_mlp $lora_mlp \
                                            --lora_head $lora_head \
                                            --eval_interval $eval_interval \
                                            --save_interval $save_interval \
                                            --eval_iters $eval_iters \
                                            --eval_max_new_tokens $eval_max_new_tokens \
                                        > logs/$base/$model/$dataset/"$finetune"_r"$lora_r"a"$lora_alpha"/$quantize/"$optimizer"/"$max_iter"_"$batch_size"_"$micro_batch_size"/"$learning_rate"_"$weight_decay"/"$lr_type"/"$today"_"$time".log
                                        if [ -e out/$base/$model/$dataset/"$finetune"_r"$lora_r"a"$lora_alpha"/$quantize/"$optimizer"/"$max_iter"_"$batch_size"_"$micro_batch_size"/"$learning_rate"_"$weight_decay"/"$lr_type"/"$today"/lit_model_*.pth ] && $upload_flag; then
                                            python script/hf_upload.py \
                                                --hf_token $hf_token \
                                                --repo_dir $repo_dir \
                                                --base $base \
                                                --model $model \
                                                --dataset $dataset \
                                                --finetune $finetune \
                                                --r $lora_r \
                                                --alpha $lora_alpha \
                                                --optimizer $optimizer \
                                                --quantize $quantize \
                                                --iter $max_iter \
                                                --batch_size $batch_size \
                                                --micro_batch_size $micro_batch_size \
                                                --learning_rate $learning_rate \
                                                --weight_decay $weight_decay \
                                                --scheduler $lr_type \
                                                --date $today
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

### usage
# CUDA_VISIBLE_DEVICES=3 nohup bash sh/finetune_LoRA.sh > sh_logs/llima_swa_quantize_2023_11_02_0910.log 2> sh_logs/error_swa_r8a16_quantize_2023_11_02_0910.log &
# CUDA_VISIBLE_DEVICES=3 bash sh/finetune_LoRA.sh