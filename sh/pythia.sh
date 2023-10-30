base='pythia'
models=('pythia-12b')
# 'pythia-12b' 'pythia-6.9b' 'pythia-2.8b' 'pythia-1.4b' 'pythia-1b' 'pythia-410m' 'pythia-160m' 'pythia-70m'
datasets=('lima')
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
                --data_dir data/$base/$model/$dataset \
                --checkpoint_dir checkpoints/EleutherAI/$model \
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
# CUDA_VISIBLE_DEVICES=7 nohup bash sh/pythia.sh >sh_logs/pythia.log 2>sh_logs/error_pythia.log &