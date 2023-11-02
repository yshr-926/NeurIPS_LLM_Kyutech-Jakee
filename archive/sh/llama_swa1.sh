models=('Llama-2-7b-hf')
datasets=('limaoasst' 'limaopenbookqasciq' 'limadolly' 'dollyoasst')
finetunes=('lora_swa')
# optimizers=('AdamW' 'SGD' 'LARS' 'LAMB' 'Lion')
optimizers=('SGD' 'AdamW')
# optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='not_quantize'
max_iters=50000

batch_sizes=('128')
# batch_sizes=('4' '8' '16' '32')
micro_batch_sizes=('1')
learning_rates=('3e-4')
# weight_decays=('1e-3' '1e-2') 
# learning_rates=('3e-4' '8e-5' '3e-5')
weight_decays=('1e-3')


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
# CUDA_VISIBLE_DEVICES=2 nohup bash sh/llima_swa1.sh >sh_logs/llima_custom.log 2>sh_logs/error_llima_custom.log &