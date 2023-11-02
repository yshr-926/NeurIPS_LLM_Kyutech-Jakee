models=('open_llama_3b' 'open_llama_7b')
datasets=('lima')
finetunes=('lora' 'lora_sam')
today=$(TZ=JST-9 date "+%Y-%m-%d")
time=$(TZ=JST-9 date "+%H%M")
for dataset in ${datasets[@]}
do
    for finetune in ${finetunes[@]}
    do
        for model in ${models[@]}
        do
            # python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/$model &&
            if [ $dataset = 'dolly' ]; then
                if [ ! -d data/$dataset-$model ]; then
                python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/openlm-research/$model --destination_path data/$dataset-$model 
                fi
            else
                if [ ! -d data/$dataset-$model ]; then
                python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/openlm-research/$model --destination_path data/$dataset-$model --access_token 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
                fi
            fi
            mkdir -p logs/$model/$dataset/$finetune/$today &&
            python finetune/$finetune.py \
            --data_dir data/$dataset-$model \
            --checkpoint_dir checkpoints/openlm-research/$model \
            --out_dir out/$model/$dataset/$finetune/$today \
            --precision "bf16-true" \
            >logs/$model/$dataset/$finetune/$today/$time.log
        done
    done
done

### 実行するとき
# CUDA_VISIBLE_DEVICES=0 nohup bash sh/open_llama.sh >open_llama.log 2>error_open_llama.log &