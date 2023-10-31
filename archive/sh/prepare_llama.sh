models=('Llama-2-13b-hf')
datasets=('openbookqa')


for model in ${models[@]}
do
    # python script/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/$model
    for dataset in ${datasets[@]}
    do
        if [ $dataset = 'dolly' ]; then
            if [ ! -d data/$dataset-$model ]; then
            python script/prepare_$dataset.py --checkpoint_dir checkpoints/meta-llama/$model --destination_path data/$dataset-$model 
            fi
        else
            if [ ! -d data/$dataset-$model ]; then
            python script/prepare_$dataset.py --checkpoint_dir checkpoints/meta-llama/$model --destination_path data/$dataset-$model --access_token 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 --max_seq_length 2048
            fi
        fi
    done
done
