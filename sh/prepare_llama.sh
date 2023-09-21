models=('Llama-2-7b-hf' 'Llama-2-13b-hf' 'Llama-2-70b-hf')
datasets=('dolly' 'lima')

for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/$model &&
        if [ $dataset = 'dolly' ]; then
            if [ ! -d data/$dataset-$model ]; then
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/meta-llama/$model --destination_path data/$dataset-$model 
            fi
        else
            if [ ! -d data/$dataset-$model ]; then
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/meta-llama/$model --destination_path data/$dataset-$model --access_token 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
            fi
        fi
    done
done