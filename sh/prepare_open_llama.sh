models=('open_llama_3b' 'open_llama_7b' 'open_llama_13b')
datasets=('dolly' 'lima')

for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/$model &&
        if [ $dataset = 'dolly' ]; then
            if [ ! -d data/$dataset-$model ]; then
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/openlm-research/$model --destination_path data/$dataset-$model 
            fi
        else
            if [ ! -d data/$dataset-$model ]; then
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/openlm-research/$model --destination_path data/$dataset-$model --access_token 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
            fi
        fi
    done
done