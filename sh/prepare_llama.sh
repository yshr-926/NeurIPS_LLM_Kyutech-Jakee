models=('Llama-2-7b-hf')
# datasets=('dolly' 'lima')
datasets=('sciq')

for model in ${models[@]}
do
    # python scripts/download.py --repo_id meta-llama/$model --access_token "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" &&
    # python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/$model &&
    for dataset in ${datasets[@]}
    do
        python scripts/prepare_$dataset.py --checkpoint_dir checkpoints/meta-llama/$model --destination_path data/$dataset-$model --max_seq_length 2048
    done
done