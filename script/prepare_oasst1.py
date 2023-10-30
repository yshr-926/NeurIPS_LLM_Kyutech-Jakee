"""Implementation derived from https://github.com/LAION-AI/Open-Assistant"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import random_split
from tqdm import tqdm

import pandas as pd
from treelib import Tree

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

"""
python3 script/prepare_oasst1.py \
    --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf" \
    --destination_path data/oasst1-Llama-2-7b-hf \
    --max_seq_length 2048
"""

def prepare(
    destination_path: Path = Path("data/oasst1"),
    test_split_fraction: float = 0.1,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    mask_inputs: bool = False,  # as in alpaca-lora
    seed: int = 42,
    data_repo_id: str = "OpenAssistant/oasst1",
    ignore_index: int = -1,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the oasst1 dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    print("Loading data file...")

    from datasets import load_dataset

    dataset = load_dataset(data_repo_id, use_auth_token=access_token)
    train_data = format_dataset(dataset["train"])

    # test set is present but doesn't have any solutions, so we cannot use it here
    # but have to create our own
    # for consistency with prepare_alpaca.py and prepare_dolly.py
    test_data = format_dataset(dataset["validation"])
    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)
    
    train_set, test_set = list(train_data), list(test_data)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")

def format_dataset(dataset_partition):
    formatted_ds = []
    
    df = dataset_partition.to_pandas()
    df = df.query('lang in ["en"]')
        
    # lets grab a random message tree
    message_tree_id_set = set(df["message_tree_id"].tolist())
    message_tree_id = list(message_tree_id_set)
    
    for message_id in message_tree_id:
        df_message_tree = df.query(f"message_tree_id == '{message_id}'").sort_values("created_date")
                
        if len(df_message_tree) == 1 or message_id in {'63f26a43-8b7d-4bb2-ad36-4309034c84cd',
                                                        'fe8673b7-b875-4697-9ce4-bc063f51dc7d',
                                                        '6404bd66-8571-48eb-b290-44035d4b61f4',
                                                        '7d593566-ec33-4bb7-9216-80840feefaae',
                                                        '29a908a1-1223-435d-b96d-642aa54541b4',
                                                        '4a8477f8-2f02-4e15-b122-f55c95a3f830',
                                                        '6a4ee7a9-e549-47a0-9747-0a4006a4c6ef',
                                                        '4de8c15c-87d5-44e1-b6c3-c7160d75ae52'
                                                        }:
            continue
        
        df_message_tree = add_tree_level(df_message_tree)
    
        text_tree = Tree()
        
        s = set()
    
        for i, row in df_message_tree.iterrows():
            parent_id = row["parent_id"]
            text_short = row["text"]
            text_short = text_short.replace("\n", " ")
            parent_text = (
                df_message_tree.query(f"message_id == '{parent_id}'")["text"].values[0] if parent_id is not None else "ROOT"
            )
            parent_text = parent_text.replace("\n", " ")

            # if parent_id is None, then it is a root message so dont add parent text as is none
            if parent_id is None:
                text_tree.create_node(text_short, text_short)
                text_sh = text_short
            # else use the parent text short as the parent
            else:
                while text_short in s:
                    text_short += ' '
                text_tree.create_node(text_short, text_short, parent=parent_text)
            s.add(text_short)
        
        all_paths = []

        # すべてのパスを列挙する関数
        def enumerate_paths(tree, node_id, current_path):
            node = tree.get_node(node_id)
            current_path.append(node.tag)
            if not tree.children(node_id):
                # 葉ノードに達したら、現在のパスをリストに追加
                all_paths.append(current_path.copy())
            else:
                # 子ノードがある場合、再帰的に子ノードを探索
                children = tree.children(node_id)
                for child in children:
                    enumerate_paths(tree, child.identifier, current_path)
            current_path.pop()  # パスから現在のノードを削除

        # ルートからすべてのパスを列挙
        enumerate_paths(text_tree, text_sh, [])
        for text_list in all_paths:    
            for i in range(0, len(text_list) - 1, 2):
                formatted_ds.append({"instruction": text_list[i], "input": "", "output": text_list[i + 1]})

    return formatted_ds


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )

# oasst1 utils
def add_tree_level(df):
    """helper function to add tree level to a df"""

    ignore_list = {'a35bd422-83cd-4e98-8b25-413383c58acb',
                   '50ec40ae-576c-42ab-8f9f-85d27cba0019',
                   'd45648a2-b280-4a12-9f6d-6e07d1821dcb',
                   '799a20e9-df43-41bb-9731-9c06b478518c',
                   '83180603-8fca-48a3-9a10-55967b54d163',
                   '2538d946-d5b5-426f-abd1-fa3d55862af7',
                   'af320a94-27cd-4a46-a8d1-6821c6fbcfd2',
                   '146ce78c-ac91-4388-9ba8-271bea83de61',
                   }

    # if tree level already exists, return df
    if "tree_level" in df.columns:
        return df

    else:
        tree_level_map = {}

        # iterate over rows in df
        for i, row in df.iterrows():
            message_id = row["message_id"]
            parent_id = row["parent_id"]

            # if parent_id is None, then it is a root message
            if parent_id is None:
                tree_level_map[message_id] = 0
            # if parent_id is the same as message_tree_id, then it is a direct reply to the root message
            elif parent_id == row["message_tree_id"]:
                tree_level_map[message_id] = 1
            # else just look up the tree level of the parent_id and add 1
            else:
                if parent_id not in ignore_list:
                    tree_level_map[message_id] = tree_level_map[parent_id] + 1

        # create a df from the tree_level_map and merge it with the original df
        df_tree_level_map = (
            pd.DataFrame.from_dict(tree_level_map, orient="index", columns=["tree_level"])
            .reset_index()
            .rename(columns={"index": "message_id"})
        )

        return df.merge(df_tree_level_map, on="message_id")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
