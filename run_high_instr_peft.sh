#!/usr/bin/env bash
# 本地 peft LoRA 高层指令微调示例（不执行，仅示例）

MODEL_PATH="/path/to/llava_or_llama"    # 预训练基座
VISION_TOWER=""                          # 若需要单独指定视觉塔
DATA_ROOT="/data/yongxi/high_instr/behavior_il/stream_seed1"  # 已生成的数据目录
OUT_DIR="./outputs/high_instr_local_peft"

DATA_FILES=(
  "${DATA_ROOT}/task0" \
  "${DATA_ROOT}/task1" \
)

python train_high_instr_peft.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --data_paths "${DATA_FILES[@]}" \
  --output_dir "${OUT_DIR}" \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4 \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 2048 \
  --save_steps 500 \
  --log_steps 20 \
  --fp16

# 多卡(示例)：
# torchrun --nproc_per_node=4 train_high_instr_peft.py --model_name_or_path ${MODEL_PATH} --data_paths ... --output_dir ${OUT_DIR} --epochs 1 ...
