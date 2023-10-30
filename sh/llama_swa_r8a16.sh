models=('Llama-2-7b-hf')
# datasets=('lima')
datasets=('openbookqa' 'sciq')
finetunes=('lora_swa_r8a16')
optimizers=('AdamW')
# optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='not_quantize'
max_iters=('30000')

# batch_sizes=('16')
batch_sizes=('64')
micro_batch_sizes=('1')
learning_rates=('0.0003' '0.0001' '0.00003')
weight_decays=('0.001')
lr_types=('CosineAnnealingLR')
# lr_types=('Fix')
# learning_rates=('3e-4')
# weight_decays=('0.01')
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
                                    mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$lr_type/$today &&
                                    python finetune/$finetune.py \
                                    --data_dir data/$dataset-$model \
                                    --checkpoint_dir checkpoints/meta-llama/$model \
                                    --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$lr_type/$today \
                                    --precision "bf16-true" \
                                    --optim_name $optimizer \
                                    --max_iters $max_iters \
                                    --batch_size $batch_size \
                                    --micro_batch_size $micro_batch_size \
                                    --learning_rate $learning_rate \
                                    --weight_decay $weight_decay \
                                    --lr_type $lr_type \
                                    >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$lr_type/$today/"$time"_"$batch_size"_"$micro_batch_size"_"$learning_rate"_"$weight_decay"_"$lr_type".log
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