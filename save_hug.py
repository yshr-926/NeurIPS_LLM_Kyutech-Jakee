from huggingface_hub import login, HfApi, create_repo, delete_repo
import os


os.environ["HUGGINGFACE_TOKEN"] = "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"  # 書き込み用のHFトークン
# repo = "yshr-926/lit-model4"
login(token=os.getenv("HUGGINGFACE_TOKEN"))
# delete_repo(os.getenv("HUGGINGFACE_REPO"))
# create_repo(repo, exist_ok=False)

api = HfApi()

for model in ["Llama-2-7b-hf"]:
    for dataset in ["flan"]:
        for finetune in ["lora"]:
            for optimizer in ["AdamW", "Lion", "SGD"]:
                for quantize in ["not_quantize"]:
                    for date in ["2023-10-09"]:
                        repo = f"yshr-926/{model}_{dataset}_{finetune}_{optimizer}_{quantize}_{date}"
                        create_repo(repo, exist_ok=True)
                        out_dir = f"out/{model}/{dataset}/{finetune}_{optimizer}/{quantize}/{date}"
                        api.upload_folder(
                            folder_path=out_dir,
                            repo_id=repo, 
                            repo_type='model', 
                            allow_patterns=[f"lit_model_lora_{optimizer}_finetuned.pth",
                                            f"lit_avemodel_lora_{optimizer}_finetuned.pth"],
                        )