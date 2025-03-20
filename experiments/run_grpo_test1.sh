export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export OMP_NUM_THREADS="1"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ../src/r1-v/src/open_r1/grpo.py \
    --deepspeed zero2_offload.json \
    --output_dir output \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name "leonardPKU/clevr_cogen_a_train" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --tf32 True \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 8 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-2B-GRPO-CLEVR-70k
