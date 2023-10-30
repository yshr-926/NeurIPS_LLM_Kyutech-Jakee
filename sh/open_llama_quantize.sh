models=('open_llama_3b' 'open_llama_7b')
datasets=('dolly' 'lima')
finetunes=('lora')
optimizers=('AdamW' 'SGD' 'LARS' 'LAMB' 'Lion')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantizes=('bnb.nf4' 'bnb.nf4-dq')
max_iters=50000

for dataset in ${datasets[@]}
do
    for finetune in ${finetunes[@]}
    do
        for model in ${models[@]}
        do
            for optimizer in ${optimizers[@]}
            do
                for quantize in ${quantizes[@]}
                do
                mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$today &&
                python finetune/$fine.py \
                --data_dir data/$dataset-$model \
                --checkpoint_dir checkpoints/openlm-research/$model \
                --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$today \
                --precision "bf16-true" \
                --optim_name $optimizer \
                --max_iters $max_iters \
                >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$today/$time.log
                done
            done
        done
    done
done

### 実行するとき
# CUDA_VISIBLE_DEVICES=1 nohup bash sh/open_llama_quantize.sh 2>sh_logs/error_open_llama_quantize.log &