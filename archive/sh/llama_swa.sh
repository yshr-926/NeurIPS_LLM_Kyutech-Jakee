models=('Llama-2-13b-hf')
datasets=('lima')
finetunes=('lora_swa_r8a16')
# optimizers=('AdamW' 'SGD' 'LARS' 'LAMB' 'Lion')
optimizers=('AdamW')
# optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='bnb.nf4-dq'
max_iters=10000

batch_sizes=('128')
# batch_sizes=('4' '8' '16' '32')
micro_batch_sizes=('1')
learning_rates=('8e-4')
weight_decays=('1e-2') 
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
                                mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$max_iters"_"$batch_size"_"$learning_rate"_"$weight_decay"/$today &&
                                python finetune/$finetune.py \
                                --data_dir data/$dataset-$model \
                                --checkpoint_dir checkpoints/meta-llama/$model \
                                --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$max_iters"_"$batch_size"_"$learning_rate"_"$weight_decay"/$today \
                                --precision "bf16-true" \
                                --optim_name $optimizer \
                                --max_iters $max_iters \
                                --batch_size $batch_size \
                                --micro_batch_size $micro_batch_size \
                                --learning_rate $learning_rate \
                                --weight_decay $weight_decay \
                                >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$max_iters"_"$batch_size"_"$learning_rate"_"$weight_decay"/$today/"$time".log
                            done
                        done
                    done
                done
            done
        done
    done
done
### 実行するとき
# CUDA_VISIBLE_DEVICES=1 nohup bash sh/llima_swa.sh >sh_logs/llima4.log 2>sh_logs/error_llima4.log &