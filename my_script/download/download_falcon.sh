#!/bin/bash

# falcon-7b

# model
mkdir -p checkpoints/tiiuae/falcon-7b
python3 scripts/download.py \
    --repo_id tiiuae/falcon-7b

python3 scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b

# dataset
# dolly
python3 scripts/prepare_dolly.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b \
    --destination_path data/falcon/falcon-7b/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
    --destination_path data/falcon/falcon-7b/lima \
    --access_token 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


# falcon-40b

# model
mkdir -p checkpoints/tiiuae/falcon-40b
python3 scripts/download.py \
    --repo_id tiiuae/falcon-40b
python3 scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-40b

# dataset
# dolly
python3 scripts/prepare_dolly.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-40b \
    --destination_path data/falcon/falcon-40b/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/tiiuae/falcon-40b" \
    --destination_path data/falcon/falcon-40b/lima \
    --access_token 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


# falcon-180B

# model
mkdir -p checkpoints/tiiuae/falcon-180B
# MY_TOKEN -> huggingface の token
python scripts/download.py \
    --repo_id tiiuae/falcon-180B \
    --access_token MY_TOKEN \
    --from_safetensors true
python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-180B

# dataset
# dolly
python3 scripts/prepare_dolly.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-180B \
    --destination_path data/falcon/falcon-180B/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/tiiuae/falcon-180B" \
    --destination_path data/falcon/falcon-180B/lima \
    --access_token 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
