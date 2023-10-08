models=('open_llama_3b' 'open_llama_7b')
datasets=('dolly' 'lima')
finetunes=('lora')
optimizers=('AdamW' 'SGD' 'LARS' 'LAMB' 'Lion')
batch_size=4
eval_tasks="[truthfulqa_mc]"

ft_day="2023-09-21"
quantize='not_quantize'
# out/Llama-2-7b-hf/dolly/lora/2023-09-16/lit_model_lora_finetuned.pth
for dataset in ${datasets[@]}
do  
    for finetune in ${finetunes[@]}
    do
        for model in ${models[@]}
        do
            for optimizer in ${optimizers[@]}
            do
            mkdir -p results/$model/$dataset/"$finetune"_"$optimizer"/$quantize &&
            if [[ $finetune == *lora* ]]; then
                python scripts/merge_lora.py \
                --checkpoint_dir checkpoints/openlm-research/$model \
                --lora_path out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$ft_day/lit_model_"$finetune"_"$optimizer"_finetuned.pth \
                --out_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$ft_day &&
                cp checkpoints/openlm-research/$model/*.json \
                out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$ft_day &&
                cp checkpoints/openlm-research/$model/tokenizer.model \
                out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$ft_day
            fi
            python eval/lm_eval_harness.py \
            --checkpoint_dir out/$model/$dataset/"$finetune"_"$optimizer"/$quantize/$ft_day \
            --precision "bf16-true" \
            --batch_size  $batch_size \
            --eval_tasks $eval_tasks \
            --save_filepath results/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$ft_day"-result.json \
            >results/$model/$dataset/"$finetune"_"$optimizer"/$quantize/"$ft_day"-result.txt
            done
        done
    done
done

### 実行するとき
# CUDA_VISIBLE_DEVICES=0 nohup bash sh/eval_open_llama.sh 2>sh_logs/error_eval_open_llama.txt &