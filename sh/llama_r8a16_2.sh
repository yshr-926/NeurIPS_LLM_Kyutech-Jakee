models=('Llama-2-13b-hf')
datasets=('dollylimaoasstopenbookqasciq')
finetunes=('lora_swa_r8a16_2')
optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='bnb.nf4-dq'
max_iters=('100000')

# batch_sizes=('16')
# batch_sizes=('32')
batch_sizes=('128')
micro_batch_sizes=('1')
learning_rates=('0.0008')
weight_decays=('0.01' '0.001')
lr_types=('CosineAnnealingLR')
# lr_types=('Fix')
warmup_steps='100'

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
                                        for dir in ${dirs[@]}
                                        do
                                            python scripts/merge_r8a16.py \
                                            --checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf \
                                            --lora_path $dir/*ave*.pth \
                                            --out_dir $dir &&
                                            cp checkpoints/meta-llama/Llama-2-13b-hf/*.json \
                                            $dir &&
                                            cp checkpoints/meta-llama/Llama-2-13b-hf/tokenizer.model \
                                            $dir &&
                                            if [[ $dir == *SGD* ]]; then
                                                flag='SGD'
                                            else
                                                flag='AdamW'
                                            fi
                                            mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$lr_type/$today &&
                                            python finetune/$finetune.py \
                                            --data_dir data/$dataset-$model \
                                            --checkpoint_dir checkpoints/meta-llama/$model \
                                            --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$lr_type/$today \
                                            --precision "bf16-true" \
                                            --quantize $quantize \
                                            --optim_name $optimizer \
                                            --max_iters $max_iter \
                                            --batch_size $batch_size \
                                            --micro_batch_size $micro_batch_size \
                                            --learning_rate $learning_rate \
                                            --weight_decay $weight_decay \
                                            --lr_type $lr_type \
                                            --warmup_steps $warmup_steps \
                                            --log_interval 100 \
                                            >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$lr_type/$today/"$time"_"$max_iter"_"$batch_size"_"$micro_batch_size"_"$learning_rate"_"$weight_decay"_"$lr_type".log
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
done
### 実行するとき
# CUDA_VISIBLE_DEVICES=0 nohup bash sh/llama_swa_r8a16_quantize.sh &