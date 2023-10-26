from huggingface_hub import login, HfApi, create_repo, delete_repo
import os


os.environ["HUGGINGFACE_TOKEN"] = "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"  # 書き込み用のHFトークン
# repo = "yshr-926/lit-model4"
login(token=os.getenv("HUGGINGFACE_TOKEN"))
# delete_repo(os.getenv("HUGGINGFACE_REPO"))
# create_repo(repo, exist_ok=False)
# 'CosineAnnealingLR'
api = HfApi()

for model in ["Llama-2-7b-hf"]:
    for dataset in ["dollyoasstopenbookqa"]:
        for finetune in ["lora_swa"]:
            for optimizer in ["AdamW"]:
                for quantize in ["not_quantize"]:
                    for date in ["2023-10-24"]:
                        for batch_size in [128]:
                            for micro_batch_size in [1]:
                                for learning_rate in ['0.0003', '8e-05']:
                                    for weight_decay in ['0.01', '0.001']:
                                        for lr_type in ['CosineAnnealingLR']:
                                            repo = f"yshr-926/{model}_{dataset}_{finetune}_{optimizer}_{date}_{batch_size}_{learning_rate}_{weight_decay}_{lr_type}"
                                            create_repo(repo, exist_ok=True)
                                            out_dir = f"out/{model}/{dataset}/{finetune}_{optimizer}/{quantize}/{lr_type}/{date}"
                                            api.upload_folder(
                                                folder_path=out_dir,
                                                repo_id=repo,
                                                repo_type='model', 
                                                allow_patterns=[f"iter-025599-50000_{learning_rate}_{weight_decay}_{lr_type}-ckpt.pth"],
                                            )