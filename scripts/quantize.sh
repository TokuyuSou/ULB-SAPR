#!/bin/bash

SIZE=7
BIT=2
GS=64

SAVE_DIR=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym
mkdir -p $SAVE_DIR

python ./apiq/main.py \
    --model_name_or_path meta-llama/Llama-2-${SIZE}b-hf \
    --lwc --wbits ${BIT} --group_size ${GS} \
    --epochs 1 --seqlen 2048 --nsamples 128 \
    --peft_lr 0.0005 --peft_wd 0.1 --lwc_lr 1e-5 --lwc_wd 0.1 \
    --save_dir $SAVE_DIR \
    --eval_ppl \
    --quant_method DB-LLM \
    --aug_loss \
    --seed 2 \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    # --resume /home/ubuntu/ApiQ/ApiQ/model_zoos/Llama-2-7b-hf-w2g64-fake-sym/DB-LLM/run_2025-02-26-22-40-05 \
    # --reg_method before_lora \
    # --lambda_reg 1e-3 \
    # --num_expert 16 \
    # --use_cosine_lr_scheduler \
    # --warmup_ratio 0.03 
    # --regularization_target quantization_params