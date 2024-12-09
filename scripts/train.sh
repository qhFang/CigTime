set -x

read -r -d '' training_commands <<EOF
train/train.py \
    --max_len 2048 \
    --dataset /media/xuzhao/Partition5/qyzheng/qihang/llm3/train.json \
    --dataset_probs 1.0 \
    --train_batch_size 512 \
    --micro_train_batch_size 16 \
    --max_samples 100000 \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --save_path /media/xuzhao/Partition5/qyzheng/qihang/etftllama2 \
    --save_steps 5000 \
    --logging_steps 1 \
    --eval_steps 10000 \
    --zero_stage 2 \
    --max_epochs 6 \
    --bf16 \
    --flash_attn \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --use_wandb dd668f1e0b6c378b65173eda10c080c101581d76 
EOF
    # --wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    deepspeed  $training_commands
fi

#\\
#    --lora_rank 16 
#    --load_in_4bit