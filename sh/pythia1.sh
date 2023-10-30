base='pythia'
models=('pythia-12b')
# 'pythia-12b' 'pythia-6.9b' 'pythia-2.8b' 'pythia-1.4b' 'pythia-1b' 'pythia-410m' 'pythia-160m' 'pythia-70m'
datasets=('lima')
finetunes=('lora')
optimizers=('AdamW')
learning_rate=('15e-5' '75e-6')
batch_size=('1')
max_iters=('2000')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")

quantize='not_quantize'


for dataset in ${datasets[@]}
do
    for finetune in ${finetunes[@]}
    do
        for model in ${models[@]}
        do
            for optimizer in ${optimizers[@]}
            do
                for lr in ${learning_rate[@]}
                do
                    for bs in ${batch_size[@]}
                    do
                        for iter in ${max_iters[@]}
                        do
                            mkdir -p logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$lr"_"$bs"_"$iter"/$today &&
                            python finetune/$finetune.py \
                            --data_dir data/$base/$model/$dataset \
                            --checkpoint_dir checkpoints/EleutherAI/$model \
                            --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$lr"_"$bs"_"$iter"/$today \
                            --precision "bf16-true" \
                            --optim_name $optimizer \
                            --max_iters $iter \
                            --lr $lr \
                            --bs $bs \
                            >logs/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$lr"_"$bs"_"$iter"/$today/$time.log
                        done
                    done
                done
            done
        done
    done
done
### 実行するとき
# CUDA_VISIBLE_DEVICES=3 nohup bash sh/pythia2.sh >sh_logs/pythia1.log 2>sh_logs/error_pythia1.log &