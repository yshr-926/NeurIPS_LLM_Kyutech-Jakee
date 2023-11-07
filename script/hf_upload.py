from huggingface_hub import login, HfApi, create_repo
import os


def upload(
    hf_token: str = None,
    repo_dir: str = None,
    base: str = None,
    model: str = None,
    dataset: str = None,
    finetune: str = None,
    r: str = None,
    alpha: str = None,
    optimizer: str = None,
    quantize: str = None,
    iter: str = None,
    batch_size: str = None,
    micro_batch_size: str = None,
    learning_rate: str = None,
    weight_decay: str = None,
    scheduler: str = None,
    date: str = None
) -> None:
    # huggingface token "write"
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    api = HfApi()

    repo = f"{repo_dir}/{model}_{dataset}_{finetune}_{optimizer}_{quantize}_{iter}_{batch_size}_{micro_batch_size}_{learning_rate}_{weight_decay}_{date}"
    # リポジトリ作成
    create_repo(repo, exist_ok=False)
    #モデルがあるディレクトリ
    out_dir = f"out/{base}/{model}/{dataset}/{finetune}_r{r}a{alpha}/{quantize}/{optimizer}/{iter}_{batch_size}_{micro_batch_size}/{learning_rate}_{weight_decay}/{scheduler}/{date}"
    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo, 
        repo_type='model', 
        ignore_patterns=["lit_model.pth", "*ckpt.pth", "version*"]
        )
        # merge.pyした後はlit_model.pthが作成されるが,ファイルのサイズが大きいためここでは無視

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload)