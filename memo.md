# memo

## setup memo

1. docker イメージをdockerfile から作成

    - 卒論の時使ったのを流用

2. 開発版pytorch2.1 を入れる

    - cmake, lit がないよと怒られる
    
    ```
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    triton 2.0.0 requires cmake, which is not installed.
    triton 2.0.0 requires lit, which is not installed.
    ```
    
    - 黙って入れた

3. 色々pipで入れる

    - なんか怒られた

    ```
    ERROR: huggingface-hub 0.16.4 has requirement packaging>=20.9, but you'll have packaging 20.3 which is incompatible.
    ```

    - ここを参考にupdateした
    - https://stackoverflow.com/questions/68140977/huggingface-hub-0-0-12-requires-packaging-20-9-but-youll-have-packaging-20-4
    - どうやら元々入っていたみたい


## finetuning

```bash
  CUDA_VISIBLE_DEVICES=1 python3 finetune/lora.py \
  --data_dir data/dolly-stablelm3b \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b \
  --out_dir out/stablelm3b/dolly/lora/experiment1 \
  --precision "bf16-true"
```


## eval

- lora 

```bash
CUDA_VISIBLE_DEVICES=1 python3 eval/lm_eval_harness_lora.py \
    --lora_path "out/falcon7b/dolly/lora/experiment4/lit_model_lora_finetuned.pth" \
    --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath "results.json"
```

- lora 以外

```bash
CUDA_VISIBLE_DEVICES=1 python3 eval/lm_eval_harness.py \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b \
  --precision "bf16-true" \
  --eval_tasks "[truthfulqa_mc,gsm8k]" \
  --batch_size 4 \
  --save_filepath "results-stablelm-3b.json"
```

## ERROR

### scripts が import されない

- scripts -> script に変更
- error が出るたびに文を変更する
- 今の所これでどうにかなる
- 変更したファイル
    - finetune/lora.py
    - generate/lora.py
    - convert_lit_checkpoint.py

### RuntimeError: Inference tensors cannot be saved for backward. To work around you can make a clone to get a normal tensor and use it in autograd.

- @torch.inference_mode -> @torch.no_grad
- https://stackoverflow.com/questions/75517324/runtimeerror-inference-tensors-cannot-be-saved-for-backward-to-work-around-you

### CUDA_VISIBLE_DEVICES

- export CUDA_VISIBLE_DEVICES=0 はシェル内ではなく事前に設定しておく
- 事前に設定しないと上手いこといかない


  python3 script/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_lora_finetuned.pth \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b

LLMのファインチューニングには色々な種類がある
- 加算的な方法(Additive methods)
    - アダプター(Adapters)
    - ソフトプロンプト(soft Prompts)
-  他の加算的な方法(Other additive approaches Additive methods)
    - LeTS
    - LeTS
    - IA3
- 選択的なパラメータ効率の微調整(Selective PEFT)
    - ネットワークの上位数層のみを微調整する方法
    - モデルのバイアスの微調整のみを行う方法
    - 特定の行のみを調整する方法
    - スパースアップデート法
        - 実用的でない，課題あり
- 再パラメータ化ベースのパラメータ効率の微調整方法(Reparametrization-based parameter-efficient finetuning methods)
    - Fastfood変換を使用して再パラメータ化
    - Low-Rank Adaptation(LoRa)
        - 実装が簡単
    - Kronecker積の再パラメータ化

- これらの手法を組み合わせる方法
    - 例えば MAM Adapter
        - Adapers, Prompt tuning の組み合わせ
    - 他にも色々


python3 script/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_lora_finetuned.pth \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b


CUDA_VISIBLE_DEVICES=2 python3 eval/lm_eval_harness_lora.py \
    --lora_path "out/falcon7b/dolly/lora/experiment1/lit_model_lora_finetuned.pth" \
    --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc,gsm8k]" \
    --batch_size 4 \
    --save_filepath "results.json"


CUDA_VISIBLE_DEVICES=1 python3 eval/lm_eval_harness.py \
    --checkpoint_dir "checkpoints/tiiuae/falcon-7b/" \
    --precision "bf16-true" \
    --batch_size 4 \
    --save_filepath "results.json"


python eval/lm_eval_harness.py \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b \
  --precision "bf16-true" \
  --eval_tasks "[truthfulqa_mc,gsm8k]" \
  --batch_size 4 \
  --save_filepath "results-stablelm-3b.json"

  python3 script/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_lora_finetuned.pth \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b


python3 script/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_lora_finetuned.pth \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b

CUDA_VISIBLE_DEVICES=1 python3 eval/lm_eval_harness_lora.py \
    --lora_path "out/falcon7b/lima/lora/experiment1/lit_model_lora_finetuned.pth" \
    --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath "results.json"

## submission

- 使い捨て
- 外部から止めるか，CTRL+C で止めるか

```bash
docker run --name sakaguchi_submission --gpus all --rm -p 8082:80 toy_submission
```

## git

- commit して pull

```bash
git add .
git commit -m "MESSAGE"
git pull origin main
```


## 明日すること

- evalについて調べる
- 量子化の指定ができるようにshellscriptを変更する
- 評価するshellscriptを作成する
- dataset を入れてみる
- 頑張る





python3 script/convert_lit_checkpoint.py \
  --checkpoint_name lit_model_finetuned.pth \
  --out_dir out/pythia/pythia-70m/dolly/full/test \
  --model_name pythia-70m


CUDA_VISIBLE_DEVICES=2 python3 eval/lm_eval_harness.py \
    --checkpoint_dir checkpoints/EleutherAI/pythia-70m \
    --precision "bf16-true" \
    --batch_size 4 \
    --eval_tasks "[truthfulqa_mc]" \
    --save_filepath result/results.json

python3 script/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_adapter_finetuned.pth \
    --out_dir out/falcon/falcon-7b/dolly/adapter/test \
    --model_name falcon-7b

CUDA_VISIBLE_DEVICES=2 python3 eval/lm_eval_harness.py \
    --checkpoint_dir out/falcon/falcon-7b/dolly/adapter/test \
    --precision "bf16-true" \
    --batch_size 4 \
    --eval_tasks "[truthfulqa_mc]" \
    --save_filepath result/results_falcon7b1.json

python3 script/convert_hf_checkpoint.py \
    --checkpoint_name lit_model_adapter_finetuned.pth \
    --out_dir out/falcon/falcon-7b/dolly/adapter/test \
    --model_name falcon-7b




# eval/lora　(OK)
mkdir -p out/lora_merged/pythia/pythia-70m/lima/compare1

CUDA_VISIBLE_DEVICES=7 python3 script/merge_lora.py \
    --checkpoint_dir checkpoints/EleutherAI/pythia-70m \
    --lora_path out/pythia/pythia-70m/lima/lora/compare1/lit_model_lora_finetuned.pth \
    --out_dir out/lora_merged/pythia/pythia-70m/lima/compare1

cp checkpoints/EleutherAI/pythia-70m/*.json \
    out/lora_merged/pythia/pythia-70m/lima/compare1/

mkdir -p result/pythia/pythia-70m/lima/lora/compare1

CUDA_VISIBLE_DEVICES=7 python3 eval/lm_eval_harness.py \
    --checkpoint_dir out/lora_merged/pythia/pythia-70m/lima/compare1/ \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath result/pythia/pythia-70m/lima/lora/compare1/result.json

chmod 666 result/pythia/pythia-70m/dolly/lora/compare1/result.json

# eval/full
python3 script/convert_lit_checkpoint.py \
    --checkpoint_path out/pythia/pythia-70m/dolly/full/compare1/lit_model_finetuned.pth \
    --output_path out/pythia/pythia-70m/dolly/full/compare1/lit_model.pth \
    --config_path out/pythia/pythia-70m/dolly/full/compare1/lit_config.json
                        
cp checkpoints/EleutherAI/pythia-70m/*.json \
    out/pythia/pythia-70m/dolly/full/compare1/

mkdir -p result/pythia/pythia-70m/dolly/full/compare1

CUDA_VISIBLE_DEVICES=7 python3 eval/lm_eval_harness.py \
    --checkpoint_dir out/pythia/pythia-70m/dolly/full/compare1 \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath result/pythia/pythia-70m/dolly/full/compare1/result2.json

CUDA_VISIBLE_DEVICES=7 python3 eval/lm_eval_harness.py \
    --checkpoint_dir checkpoints/EleutherAI/pythia-70m \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath result/pythia/pythia-70m/dolly/full/base/result.json

mkdir -p result/pythia/pythia-70m/dolly/full/base/result.json






mkdir -p result/pythia/pythia-70m/dolly/adapter/compare1

CUDA_VISIBLE_DEVICES=7 python3 eval/lm_eval_harness.py \
    --checkpoint_dir out/pythia/pythia-70m/dolly/adapter/compare1 \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath result/pythia/pythia-70m/dolly/adapter/compare1/result.json

CUDA_VISIBLE_DEVICES=7 python3 eval/lm_eval_harness.py \
    --checkpoint_dir checkpoints/EleutherAI/pythia-70m \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath result/pythia/pythia-70m/dolly/full/base/result.json

mkdir -p result/pythia/pythia-70m/dolly/full/base/result.json


# eval/lora　(OK)
mkdir -p out/lora_merged/pythia/pythia-12b/dolly/compare1

CUDA_VISIBLE_DEVICES=7 python3 script/merge_lora.py \
    --checkpoint_dir checkpoints/EleutherAI/pythia-70m \
    --lora_path out/pythia/pythia-70m/dolly/lora/compare1/lit_model_lora_finetuned.pth \
    --out_dir out/lora_merged/pythia/pythia-70m/dolly/compare1

cp checkpoints/EleutherAI/pythia-70m/*.json \
    out/lora_merged/pythia/pythia-70m/dolly/compare1/

mkdir -p result/pythia/pythia-70m/dolly/lora/compare1

CUDA_VISIBLE_DEVICES=7 python3 eval/lm_eval_harness.py \
    --checkpoint_dir out/lora_merged/pythia/pythia-70m/dolly/compare1/ \
    --precision "bf16-true" \
    --eval_tasks "[truthfulqa_mc]" \
    --batch_size 4 \
    --save_filepath result/pythia/pythia-70m/dolly/lora/compare1/result.json

chmod 666 result/pythia/pythia-70m/dolly/lora/compare1/result.json


# model_memo

Falcon: 7B ~
LLaMA or Llama 2: 7B(6.7B) ~
OpenLLaMA: 3B ~
Red Pajama Base (not instruction tuned models): 3B ~
MPT: 7(6.7)B ~ 
OPT: 1.3B ~ 
Bloom: 1.1B ~ 
GPT Neo, J, NeoX, Pythia: 2B ~
GPT2: 1.5B ~ 
T5 (not Flan-T5): 223M ~ 
BART: 140M
DeBERTa: 184M
RoBERTa: 123M
BERT: 110M
ALBERT: 12M
DistilBERT: 66M
Electra: 14M 
UL2: 20B
Cerebras (btlm, GPT): 111M

git@github.com:yshr-926/NeurIPS_LLM_Nitanda_Lab.git

## commit

## push
git push origin sakaguchi

## pull
git fetch
git merge origin/<remote-branch-name>

git diff sakaguchi origin/<remote-branch-name>

vim
Ctrl + X で終わらせる




CUDA_VISIBLE_DEVICES=7 python3 finetune/lora.py \
    --data_dir "data/pythia/pythia-70m/flan" \
    --checkpoint_dir "checkpoints/EleutherAI/pythia-70m" \
    --out_dir "out/pythia/pythia-70m/flan/lora/test" \
    --precision "bf16-true"

# helm

- docker server を立てる

```bash
docker run --rm --name sakaguchi_submission --gpus all -p 8082:80 toy_submission_sakaguchi
```

- helm-run

```bash
helm-run --conf-paths <conf-file> --suite v1 --max-eval-instances 1000
helm-summarize --suite v1
helm-server
```


# eval task

```conf
#bigbench

# 例文の包含関係を調べる（Yes, No）
#analytic_entailment: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/analytic_entailment
{description: "big_bench:model=neurips/local,max_train_instances=3,task=analytic_entailment,subtask=", priority: 1}

# 因果関係についての質問（Yes, No）
#causal_judgment: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/causal_judgment
{description: "big_bench:model=neurips/local,max_train_instances=3,task=causal_judgment,subtask=", priority: 1}

# 絵文字から映画の予測(5択問題)
#emoji_movie: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/emoji_movie
{description: "big_bench:model=neurips/local,max_train_instances=3,task=emoji_movie,subtask=", priority: 1}

# イベントの関係の読み取り（因果，相関，中立）
#empirical_judgments: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/empirical_judgments
{description: "big_bench:model=neurips/local,max_train_instances=3,task=empirical_judgments,subtask=", priority: 1}

# わかっていることか不明なことかの判断（Ans，Unknown）
#known_unknowns: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/known_unknowns
{description: "big_bench:model=neurips/local,max_train_instances=3,task=known_unknowns,subtask=", priority: 1}

# 論理的に説明されて，順序を当てる（5択問題）
# logical_deduction: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/logical_deduction
{description: "big_bench:model=neurips/local,max_train_instances=3,task=logical_deduction,subtask=three_objects", priority: 1}

# 最後のボケがほんとか嘘かの見分け（読解問題）（Yes, No）
#strange_stories: https://github.com/google/big-bench/tree/main/bigbench/benchmark_tasks/strange_stories
{description: "big_bench:model=neurips/local,max_train_instances=3,task=strange_stories,subtask=multiple_choice", priority: 1}

# どっちが皮肉かの比較（a, b）
#snarks: https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/snarks
{description: "big_bench:model=neurips/local,max_train_instances=3,task=snarks,subtask=", priority: 1}

# 冗談かどうかの分類（Yes, No）
#dark_humor_detection: https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/dark_humor_detection
{description: "big_bench:model=neurips/local,max_train_instances=3,task=dark_humor_detection,subtask=", priority: 1}


#mmlu
# 知識問題(4択問題)
{description: "mmlu:model=neurips/local,subject=philosophy,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_biology,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_chemistry,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_computer_science,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_european_history,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_geography,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_government_and_politics,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_macroeconomics,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_mathematics,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_microeconomics,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_physics,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_psychology,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_statistics,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_us_history,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=high_school_world_history,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=moral_disputes,data_augmentation=canonical", priority: 1}
{description: "mmlu:model=neurips/local,subject=moral_scenarios,data_augmentation=canonical", priority: 1}


#truthful QA
# 知識問題(known_unknowns に似てる)（選択問題）
{description: "truthful_qa:task=mc_single,model=neurips/local", priority: 1},

#CNN/daily mail
# 抽出的要約(長文を読んで短くまとめる)
{description: "summarization_cnndm:model=neurips/local", priority: 1},
#GSM
# 簡単な計算問題
{description: "gsm:model=neurips/local", priority: 1}
#BBQ
# 読解問題
{description: "bbq:subject=all,model=neurips/local", priority: 1},
```


## eval 


--no-cache

1. hf_upload.py でアップロードする
2. create_dockerfile.py で dockerfile を作る
3. 作ったファイルをコピーする
4. docker build -f Dockerfile.skeval -t adamw_50000_128_1_0.0003_0.001 .　でイメージを作る
5. docker run --rm --name sakaguchi_eval_model --gpus all -p 8082:80 adamw_50000_128_1_0.0003_0.001 でコンテナを作成する
6. helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10
7. helm-summarize --suite v1
8. helm-server


lit_model_ave_CosineAnnealingLR_AdamW_50000_128_1_0.0008_0.01.pth
lit_model_ave_Fix_AdamW_50000_128_1_0.0003_0.001.pth




Limaでバッチサイズを小さく，ステップサイズを調整
ASGDで調整をする

1万，5万　で比較

・複数データセットを使った時の実験
・lima のバッチサイズを探す実験(64でも試してみたい，swaを使って150000も試してみたい)
・swa を使ってパラメータの調整をする実験



4. docker build -f Dockerfile.merge -t merge_3_31 .
5. docker run --rm --name sakaguchi_eval_model --gpus all -p 8082:80 merge_1_31

1_2
1_2_11
2_11
8_12
11_1
11_31
1_31


## 命名規則

logs dir
{model}/{model_size}/{dataset}/{method}_{r8}{a16}/{quantize}/{optimizer}/{iters}_{batch_size}_{micro_batch_size}/{learning_rate}_{weight_decay}/{scheduler}

log file
{date}_{time}.log

data dir 
data/{model}/{model_size}/{dataset}

out dir
{model}/{model_size}/{dataset}/{method}_{r8}{a16}/{quantize}/{optimizer}/{iters}_{batch_size}_{micro_batch_size}/{learning_rate}_{weight_decay}/{scheduler}/{date}

LoRA path 
lit_model_{ave|nonave}_lora_{optimizer}_{iters}_{batch_size}_{micro_batch_size}_{learning_rate}_{weight_decay}_{scheduler}.pth

huggingface repo
{repo_id}/{model}_{dataset}_{optimizer}_{iters}_{batch_size}_{micro_batch_size}_{learning_rate}_{weight_decay}_{scheduler}_{date}
