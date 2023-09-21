models=('open_llama_3b' 'open_llama_7b')
datasets=('dolly' 'lima')

for model in ${models[@]}
do
    python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/$model &&
    for dataset in ${datasets[@]}
    do
        if [ $dataset = 'dolly' ]; then
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/openlm-research/$model --destination_path data/$dataset-$model --max_seq_length 2048
        else
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/openlm-research/$model --destination_path data/$dataset-$model --access_token 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 --max_seq_length 2048
        fi
    done
done