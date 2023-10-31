base='pythia'
# 'pythia-1b' 'pythia-12b' 'pythia-410m' 'pythia-6.9b' 'pythia-160m' 'pythia-2.8b' 'pythia-70m' 'pythia-1.4b'
models=('pythia-1.4b')
datasets=('multidata')

for dataset in ${datasets[@]}
do
    for model in ${models[@]}
    do
        if [ $dataset = 'dolly' ]; then
            if [ ! -d data/$base/$model/$dataset ]; then
            python script/prepare_$dataset.py --checkpoint_dir checkpoints/EleutherAI/$model --destination_path data/$base/$model/$dataset 
            fi
        else
            if [ ! -d data/$base/$model/$dataset ]; then
            python script/prepare_$dataset.py --checkpoint_dir checkpoints/EleutherAI/$model --destination_path data/$base/$model/$dataset-shuffle --access_token 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 --shuffle true
            python script/prepare_$dataset.py --checkpoint_dir checkpoints/EleutherAI/$model --destination_path data/$base/$model/$dataset --access_token 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 --shuffle false
            fi
        fi
    done
done