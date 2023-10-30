models=('Llama-2-7b-hf')
# datasets=('lima')
datasets=('lima')
finetunes=('lora_swa3')
optimizers=('AdamW')
# optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='not_quantize'
max_iters=('100')

batch_sizes=('16')
# batch_sizes=('128')
micro_batch_sizes=('1')
learning_rates=('0.0003' '0.00003')
weight_decays=('0.01')
lr_types=('CosineAnnealingLR')
# lr_types=('Fix')
# learning_rates=('3e-4')
# weight_decays=('0.01')

dirs=('out/Llama-2-7b-hf/openbookqa/lora_swa_SGD/not_quantize/CosineAnnealingLR/2023-10-23' \
'out/Llama-2-7b-hf/openbookqa/lora_swa_AdamW/not_quantize/CosineAnnealingLR/2023-10-23' \
)
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
                                    for dir in ${dirs[@]}
                                    do
                                        python scripts/merge_lora.py \
                                        --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
                                        --lora_path $dir/*ave*.pth \
                                        --out_dir $dir &&
                                        cp checkpoints/meta-llama/Llama-2-7b-hf/*.json \
                                        $dir &&
                                        cp checkpoints/meta-llama/Llama-2-7b-hf/tokenizer.model \
                                        $dir &&
                                        if [[ $dir == *SGD* ]]; then
                                            flag='SGD'
                                        else
                                            flag='AdamW'
                                        fi
                                        mkdir -p logs/$model/second_"$flag"_"$dataset"/"$finetune"_"$optimizer"/$quantize/$lr_type/$today &&
                                        python finetune/$finetune.py \
                                        --data_dir data/$dataset-$model \
                                        --checkpoint_dir $dir \
                                        --out_dir out/$model/second_"$flag"_"$dataset"/"$finetune"_"$optimizer"/$quantize/$lr_type/$today \
                                        --precision "bf16-true" \
                                        --optim_name $optimizer \
                                        --max_iters $max_iters \
                                        --batch_size $batch_size \
                                        --micro_batch_size $micro_batch_size \
                                        --learning_rate $learning_rate \
                                        --weight_decay $weight_decay \
                                        --lr_type $lr_type \
                                        >logs/$model/second_"$flag"_"$dataset"/"$finetune"_"$optimizer"/$quantize/$lr_type/$today/"$time"_"$batch_size"_"$micro_batch_size"_"$learning_rate"_"$weight_decay"_"$lr_type".log
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
# CUDA_VISIBLE_DEVICES=0 nohup bash sh/llama_swa.sh &