#!/bin/bash

# # pythia-70m
# mkdir -p checkpoints/EleutherAI/pythia-70m
# # model
# python3 scripts/download.py \
#   --repo_id EleutherAI/pythia-70m
# python3 scripts/convert_hf_checkpoint.py \
#   --checkpoint_dir checkpoints/EleutherAI/pythia-70m
# # dataset
# # dolly
# python3 scripts/prepare_dolly.py \
#   --checkpoint_dir checkpoints/EleutherAI/pythia-70m \
#   --destination_path data/pythia/pythia-70m/dolly
# # lima
# python3 scripts/prepare_lima.py \
#     --checkpoint_dir "checkpoints/EleutherAI/pythia-70m" \
#     --destination_path data/pythia/pythia-70m/lima \
#     --access_token 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


# pythia-160m
mkdir -p checkpoints/EleutherAI/pythia-160m
# model
python3 scripts/download.py \
  --repo_id EleutherAI/pythia-160m
python3 scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-160m
# dataset
# dolly
python3 scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-160m \
  --destination_path data/pythia/pythia-160m/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-160m" \
    --destination_path data/pythia/pythia-160m/lima \
    --access_token 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


# pythia-410m
mkdir -p checkpoints/EleutherAI/pythia-410m
# model
python3 scripts/download.py \
  --repo_id EleutherAI/pythia-410m
python3 scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-410m
# dataset
# dolly
python3 scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-410m \
  --destination_path data/pythia/pythia-410m/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-410m" \
    --destination_path data/pythia/pythia-410m/lima \
    --access_token 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


# pythia-1b
mkdir -p checkpoints/EleutherAI/pythia-1b
# model
python3 scripts/download.py \
  --repo_id EleutherAI/pythia-1b
python3 scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-1b
# dataset
# dolly
python3 scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-1b \
  --destination_path data/pythia/pythia-1b/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-1b" \
    --destination_path data/pythia/pythia-1b/lima \
    --access_token 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


# pythia-1.4b
mkdir -p checkpoints/EleutherAI/pythia-1.4b
# model
python3 scripts/download.py \
  --repo_id EleutherAI/pythia-1.4b
python3 scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-1.4b
# dataset
# dolly
python3 scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-1.4b \
  --destination_path data/pythia/pythia-1.4b/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-1.4b" \
    --destination_path data/pythia/pythia-1.4b/lima \
    --access_token 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


# pythia-2.8b
mkdir -p checkpoints/EleutherAI/pythia-2.8b
# model
python3 scripts/download.py \
  --repo_id EleutherAI/pythia-2.8b
python3 scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-2.8b
# dataset
# dolly
python3 scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-2.8b \
  --destination_path data/pythia/pythia-2.8b/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-2.8b" \
    --destination_path data/pythia/pythia-2.8b/lima \
    --access_token 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


# pythia-6.9b
mkdir -p checkpoints/EleutherAI/pythia-6.9b
# model
python3 scripts/download.py \
  --repo_id EleutherAI/pythia-6.9b
python3 scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-6.9b
# dataset
# dolly
python3 scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-6.9b \
  --destination_path data/pythia/pythia-6.9b/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-6.9b" \
    --destination_path data/pythia/pythia-6.9b/lima \
    --access_token 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


# pythia-12b
mkdir -p checkpoints/EleutherAI/pythia-12b
# model
python3 scripts/download.py \
  --repo_id EleutherAI/pythia-12b
python3 scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-12b
# dataset
# dolly
python3 scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-12b \
  --destination_path data/pythia/pythia-12b/dolly
# lima
python3 scripts/prepare_lima.py \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-12b" \
    --destination_path data/pythia/pythia-12b/lima \
    --access_token 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
