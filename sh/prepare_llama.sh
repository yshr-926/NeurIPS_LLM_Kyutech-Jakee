models=('Llama-2-13b-hf')
datasets=('dolly' 'lima')

for model in ${models[@]}
do
    python scripts/download.py --repo_id meta-llama/$model --access_token "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" &&
    python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/$model &&
    for dataset in ${datasets[@]}
    do
        if [ $dataset = 'dolly' ]; then
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/meta-llama/$model --destination_path data/$dataset-$model --max_seq_length 2048
        else
            python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/meta-llama/$model --destination_path data/$dataset-$model --access_token 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 --max_seq_length 2048
        fi
    done
done