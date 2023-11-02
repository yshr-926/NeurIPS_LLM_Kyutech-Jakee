models=('pythia-12b')
datasets=('lima')
# datasets=('lima')
finetunes=('lora_swa')
# optimizers=('AdamW' 'SGD' 'LARS' 'LAMB' 'Lion')
optimizers=('AdamW')
# optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='not_quantize'
max_iters=7500

batch_sizes=('32')
# batch_sizes=('4' '8' '16' '32')
micro_batch_sizes=('1')
# learning_rates=('3e-4' '8e-4' '3e-3')
# weight_decays=('0.01' '0.005' '0.001')
# learning_rates=('6e-4')
# learning_rates=('3e-4' '8e-5')
learning_rates=('3e-5' '8e-6')
# weight_decays=('1e-3' '1e-4' '1e-5') 
weight_decays=('1e-3' '5e-4' '1e-4' '5e-5' '1e-5')

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
                            for dataset in ${datasets[@]}
                            do
                                mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$weight_decay"_"$learning_rates"/$today &&
                                python finetune/$finetune.py \
                                --data_dir data/pythia/$model/$dataset \
                                --checkpoint_dir checkpoints/EleutherAI/$model \
                                --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$weight_decay"_"$learning_rates"/$today \
                                --precision "bf16-true" \
                                --optim_name $optimizer \
                                --max_iters $max_iters \
                                --batch_size $batch_size \
                                --micro_batch_size $micro_batch_size \
                                --learning_rate $learning_rate \
                                --weight_decay $weight_decay \
                                >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$today/"$time"_"$batch_size"_"$micro_batch_size"_"$learning_rate"_"$weight_decay".log
                            done
                        done
                    done
                done
            done
        done
    done
done
### 実行するとき
# CUDA_VISIBLE_DEVICES=2 nohup bash sh/pythia_swa.sh >sh_logs/pythia12.log 2>sh_logs/error_pythia12.log &