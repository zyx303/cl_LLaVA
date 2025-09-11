# data/behavior_il/stream_seed1/task0.json

import json 
import os

def main():
    incremental_type = 'behavior_il'
    stream_seed = 1
    model_path = 'model/llava/llava-v1.5-7b'  # 使用原始模型路径
    type = 'low'
    
    if incremental_type == 'behavior_il':
        n_task = 7
    else:
        n_task = 4
    
    # 创建输出目录
    if type == 'high':
        output_dir = f'scripts/cl_{incremental_type}_seed{stream_seed}'
    else:
        output_dir = f'scripts_low/cl_{incremental_type}_seed{stream_seed}'
    os.makedirs(output_dir, exist_ok=True)

    if type == 'high':
        checkpoints = 'checkpoints'
    else:
        checkpoints = 'checkpoint_low'
    
    # 生成持续学习脚本
    script_lines = ["#!/bin/bash", "set -x",'export CUDA_DEVICE_ORDER="PCI_BUS_ID"','export TRANSFORMERS_CACHE=/data/yongxi/.cache/huggingface']
    cuda = "7"
    for i in range(n_task):
        # 数据路径
        if type=='high':
            data_path = f'./data/{incremental_type}/stream_seed{stream_seed}/task{i}.json'
        else:
            data_path = f'./data_low/{incremental_type}/stream_seed{stream_seed}/task{i}.json'

        # 确定模型路径：第一个任务使用原始模型，后续任务使用前一个任务的输出
        if i == 0:
            current_model_path = model_path
        else:
            current_model_path = f'./{checkpoints}/{incremental_type}_seed{stream_seed}_task{i-1}'
        
        # 输出目录
        current_output_dir = f'./{checkpoints}/{incremental_type}_seed{stream_seed}_task{i}'
        
        # 生成deepspeed命令
        script_lines.extend([
            f"echo 'Starting training for Task {i}...'",
            f'deepspeed --include "localhost:{cuda}" llava/train/train_mem.py \\',
            f"    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \\",
            f"    --deepspeed ./scripts/zero3.json \\",
            f"    --model_name_or_path {current_model_path} \\",
            f"    --version v1.5 \\",
            f"    --data_path {data_path} \\",
            f"    --image_folder . \\",
            f"    --vision_tower model/clip/clip-vit-large-patch14-336 \\",
            f"    --mm_projector_type mlp2x_gelu \\",
            f"    --mm_vision_select_layer -2 \\",
            f"    --mm_use_im_start_end False \\",
            f"    --mm_use_im_patch_token False \\",
            f"    --image_aspect_ratio pad \\",
            f"    --group_by_modality_length True \\",
            f"    --bf16 True \\",
            f"    --output_dir {current_output_dir} \\",
            f"    --num_train_epochs 1 \\",
            f"    --per_device_train_batch_size 16 \\",
            f"    --per_device_eval_batch_size 4 \\",
            f"    --gradient_accumulation_steps 1 \\",
            f"    --evaluation_strategy \"no\" \\",
            f"    --save_strategy \"steps\" \\",
            f"    --save_steps 50000 \\",
            f"    --save_total_limit 1 \\",
            f"    --learning_rate 2e-4 \\",
            f"    --weight_decay 0. \\",
            f"    --warmup_ratio 0.03 \\",
            f"    --lr_scheduler_type \"cosine\" \\",
            f"    --logging_steps 1 \\",
            f"    --tf32 True \\",
            f"    --model_max_length 2048 \\",
            f"    --gradient_checkpointing True \\",
            f"    --dataloader_num_workers 4 \\",
            f"    --lazy_preprocess True \\",
            # f"    --report_to wandb",
            "",
            f"echo 'Task {i} training completed!'",
            ""
        ])
        
        # 在任务之间添加sleep（除了最后一个任务）
        if i < n_task - 1:
            script_lines.extend([
                "echo 'Waiting 5 seconds before next task...'",
                "sleep 5",
                ""
            ])
    
    # 写入脚本文件
    script_path = os.path.join(output_dir, f'finetune_cl_{incremental_type}_seed{stream_seed}.sh')
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"Generated continual learning script: {script_path}")
    print(f"This script will train {n_task} tasks sequentially with 5-second breaks between tasks.")


if __name__ == '__main__':
    main()