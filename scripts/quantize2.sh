#!/bin/bash
# Saliency-aware training with SW loss

SIZE=7
BIT=2
GS=64

SAVE_DIR=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym
mkdir -p $SAVE_DIR

python ./apiq/main.py \
    --model_name_or_path meta-llama/Llama-2-${SIZE}b-hf \
    --lwc --wbits ${BIT} --group_size ${GS} \
    --mixedt_epochs 20 --seqlen 2048 --nsamples 128 \
    --mixedt_peft_lr 0.0005 --mixedt_lwc_wd 0.1 --mixedt_peft_wd 0.1 --mixedt_lwc_lr 0.005 \
    --save_dir $SAVE_DIR \
    --eval_ppl \
    --aug_loss \
    --seed 42 \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --use_sw_loss \
    --scale_sw_loss \
    --sw_weight 0.02 \
    --sw_n_projections 1024 \
    --lambda_reg 1e-3 \
    --weighted_reg \
    --gradient_dir your_gradient_dir \
    # --resume /home/ubuntu/ApiQ/ApiQ/model_zoos/Llama-2-7b-hf-w2g64-fake-sym/DB-LLM/run_2025-02-26-22-40-05 \
    # --reg_method before_lora \
    # --lambda_reg 1e-3 \
    # --num_expert 16 \
    # --use_cosine_lr_scheduler \
    # --warmup_ratio 0.03 
    # --regularization_target quantization_params