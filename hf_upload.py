from huggingface_hub import login, HfApi, create_repo
import os

os.environ["HUGGINGFACE_TOKEN"] = "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" # 書き込み用のHFトークン
login(token=os.getenv("HUGGINGFACE_TOKEN"))

api = HfApi()

flag = True

# for model in ["Llama-2-13b-hf"]:
#     for dataset in ['limaoasst']:
#         for finetune in ["lora_swa"]:
#             for optimizer in ["AdamW"]:
#                 for quantize in ["not_quantize"]:
#                     for date in ["2023-10-23"]:
#                         for iter in ["50000"]:
#                             for lr in ["3e-4"]:
#                                 for bs in ["128"]:
#                                     for wd in ["1e-3"]:
#                                         if flag:
#                                             repo = f"miz22/{model}_{dataset}_{finetune}_{optimizer}_{quantize}_{iter}_{bs}_{lr}_{wd}_{date}"
#                                             create_repo(repo, exist_ok=False) # リポジトリ作成
#                                             out_dir = f"out/{model}/{dataset}/{finetune}_{optimizer}/{quantize}/{iter}_{bs}_{lr}_{wd}/{date}" #モデルがあるディレクトリ
#                                             api.upload_folder(
#                                                 folder_path=out_dir,
#                                                 repo_id=repo, 
#                                                 repo_type='model', 
#                                                 ignore_patterns=["lit_model.pth", "*ckpt.pth", "version*"]  # merge.pyした後はlit_model.pthが作成されるが
#                                                 )                                                           # ファイルのサイズが大きいためここでは無視
#                                         else:
#                                             print(f"miz22/{model}_{dataset}_{finetune}_{optimizer}_{quantize}_{date}")

repo = f"miz22/Llama-2-13b-hf_submission-limaoasst_lora_swa_r8a16_AdamW_bnb.nf4-dq_Fix_2023-10-27"
create_repo(repo, exist_ok=True) # リポジトリ作成
out_dir = f"out/Llama-2-13b-hf/submission-limaoasst/lora_swa_r8a16_AdamW/bnb.nf4-dq/Fix/2023-10-27" #モデルがあるディレクトリ
api.upload_folder(
folder_path=out_dir,
repo_id=repo, 
repo_type='model', 
ignore_patterns=["lit_model.pth", "*ckpt.pth", "version*"]  # merge.pyした後はlit_model.pthが作成されるが
)                                                           # ファイルのサイズが大きいためここでは無視
