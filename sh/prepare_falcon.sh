models=('falcon-7b')
datasets=('dolly' 'lima' 'flan')

for model in ${models[@]}
do
    # python scripts/download.py --repo_id tiiuae/$model &&
    # python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/tiiuae/$model &&
    for dataset in ${datasets[@]}
    do
        if [ $dataset = 'lima' ]; then
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/tiiuae/$model --destination_path data/$dataset-$model --access_token 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 --max_seq_length 2048
        else
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/tiiuae/$model --destination_path data/$dataset-$model --max_seq_length 2048
        fi
    done
done