#!/bin/bash

declare -A CHECKPOINT_DIR=(
    ["falcon"]="tiiuae"
    ["pythia"]="EleutherAI"
    ["llama"]="meta-llama"
    ["open-llama"]="openlm-research"
)
MY_TOKEN="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"

# falcon, pythia, llama, open-llama
base="open-llama"

# falcon
# models=("falcon-7b" "falcon-40b" "falcon-180B")
# pythia
# models=("pythia-70m" "pythia-160m" "pythia-410m" "pythia-1b" "pythia-1.4b" "pythia-2.8b" "pythia-6.9b" "pythia-12b")
# llama
# models=("Llama-2-7b-hf" "Llama-2-13b-hf" "Llama-2-70b-hf")
# open-llama
# models=("open_llama_3b" "open_llama_7b" "open_llama_13b")
models=("open_llama_3b")

# datasets=("dolly" "lima" "flan" "oasst1" "openbookqa" "sciq")
datasets=("dolly" "lima")

for model in ${models[@]}
do
    # prepare model
    if [ ! -d checkpoints/${CHECKPOINT_DIR["${base}"]}/$model ]; then
        echo start download model "==>" $model
        mkdir -p checkpoints/${CHECKPOINT_DIR["${base}"]}/$model
        if [ $base = "llama" ]; then
            python script/download.py \
                --repo_id ${CHECKPOINT_DIR["${base}"]}/$model \
                --access_token $MY_TOKEN
        elif [ $model = "falcon-180B" ]; then
            python script/download.py \
                --repo_id ${CHECKPOINT_DIR["${base}"]}/$model \
                --access_token $MY_TOKEN \
                --from_safetensors true
        else
            python script/download.py \
                --repo_id ${CHECKPOINT_DIR["${base}"]}/$model
        fi
        echo start convert checkpoint "==>" $model
        python script/convert_hf_checkpoint.py \
            --checkpoint_dir checkpoints/${CHECKPOINT_DIR["${base}"]}/$model
    fi

    # prepare dataset
    for dataset in ${datasets[@]}
    do
        echo start download dataset "==>" $model/$dataset
        python script/prepare_$dataset.py \
            --checkpoint_dir checkpoints/${CHECKPOINT_DIR["${base}"]}/$model \
            --destination_path data/$base/$model/$dataset \
            --access_token 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
    done
done

### usage
# nohup bash sh/prepare_model_dataset.sh &
