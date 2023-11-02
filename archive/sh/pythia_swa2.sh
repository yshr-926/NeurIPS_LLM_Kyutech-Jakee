models=('pythia-12b')
datasets=('lima' 'multidata')
finetunes=('lora')
# optimizers=('AdamW' 'SGD' 'LARS' 'LAMB' 'Lion')
optimizers=('SGD' 'AdamW' 'Lion')
# optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='not_quantize'
max_iters=('50000' '100000' '10000')

batch_sizes=('128' '64' '32')
# batch_sizes=('4' '8' '16' '32')
micro_batch_sizes=('1')
# learning_rates=('3e-4' '8e-4' '3e-3')
# weight_decays=('0.01' '0.005' '0.001')
learning_rates=('3e-4')
weight_decays=('0.01')

for iter in ${max_iters[@]}
do
    for batch_size in ${batch_sizes[@]}
    do
        for finetune in ${finetunes[@]}
        do
            for model in ${models[@]}
            do
                for optimizer in ${optimizers[@]}
                do
                    for micro_batch_size in ${micro_batch_sizes[@]}
                    do
                        for learning_rate in ${learning_rates[@]}
                        do
                            for weight_decay in ${weight_decays[@]}
                            do
                                for dataset in ${datasets[@]}
                                do
                                    mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$learning_rate"_"$batch_size"_"$iter"/$today &&
                                    python finetune/$finetune.py \
                                    --data_dir data/pythia/$model/$dataset \
                                    --checkpoint_dir checkpoints/EleutherAI/$model \
                                    --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$learning_rate"_"$batch_size"_"$iter"/$today \
                                    --precision "bf16-true" \
                                    --optim_name $optimizer \
                                    --max_iters $iter \
                                    --batch_size $batch_size \
                                    --micro_batch_size $micro_batch_size \
                                    --learning_rate $learning_rate \
                                    --weight_decay $weight_decay \
                                    >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$learning_rate"_"$batch_size"_"$iter"/$today/"$time"_"$batch_size"_"$micro_batch_size"_"$learning_rate"_"$weight_decay".log
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
# CUDA_VISIBLE_DEVICES=5 nohup bash sh/pythia_swa2.sh >sh_logs/pythia13.log 2>sh_logs/error_pythia13.log &