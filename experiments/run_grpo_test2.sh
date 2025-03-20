#!/bin/bash
# use DeepSpeed
# export NCCL_BLOCKING_WAIT=0
# export TOKENIZERS_PARALLELISM=false
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

export OMP_NUM_THREADS=8
GPUS="0,1,2,3"

export WANDB_RUN_NAME=Qwen-VL-3B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)

# cd /home/tiger/multimodal-open-r1
# # pip3 install vllm==0.6.6.post1
# pip3 install -e ".[dev]"
# pip3 install wandb==0.18.3
# 1 for debugging
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ../src/r1-v/src/open_r1/grpo.py \
    --deepspeed zero3_offload.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name "leonardPKU/clevr_cogen_a_train" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --save_total_limit 8 \
    --num_train_epochs 1 \
    --run_name $WANDB_RUN_NAME
