models=('Llama-2-7b-hf')
datasets=('limadolly')
finetunes=('lora')
optimizers=('AdamW')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='not_quantize'
max_iters=1000

for dataset in ${datasets[@]}
do
    for finetune in ${finetunes[@]}
    do
        for model in ${models[@]}
        do
            for optimizer in ${optimizers[@]}
            do
                mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$today &&
                python finetune/$finetune.py \
                --data_dir data/$dataset-$model \
                --checkpoint_dir checkpoints/meta-llama/$model \
                --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$today \
                --precision "bf16-true" \
                --optim_name $optimizer \
                --max_iters $max_iters \
                >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$today/$time.log
            done
        done
    done
done
### 実行するとき
# CUDA_VISIBLE_DEVICES=0 nohup bash sh/llama.sh >sh_logs/llama_openbookqa.log 2>sh_logs/error_llama_openbookqa.log &