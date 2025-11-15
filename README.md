# LLaVA for VLA Continual Learning

这是一个将 LLaVA (Large Language and Vision Assistant) 应用到 VLA (Vision-Language-Action) 持续学习场景的项目。本项目基于 [LLaVA](https://github.com/haotian-liu/LLaVA) 代码库，并针对持续学习任务进行了修改和扩展。

实验设置基于[cl-alfred](https://github.com/snumprlab/cl-alfred)，将其与VLM结合。

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd cl_LLaVA
```

2. 安装依赖：
```bash
pip install -e .
```

3. 下载预训练模型：
   - LLaVA v1.5-7b: 放置在 `model/llava/llava-v1.5-7b/`
   - CLIP Vision Tower: 放置在 `model/clip/clip-vit-large-patch14-336/`

4. 下载完整的alfred数据集到：`data/full/json_feat_2.1.0`

## 使用流程

### 1. 生成数据集


对于high instruction:
```bash
python src/process_data.py \
    --data ./data \
    --pp_folder json_feat_2.1.0 \
    --incremental_type behavior_il \
    --stream_seed 1 \
    --output_dir ./data/behavior_il/stream_seed1
```

对于low action:
```bash
python src/process_data_low.py \
    --data ./data_low \
    --pp_folder json_feat_2.1.0 \
    --incremental_type behavior_il \
    --stream_seed 1 \
    --output_dir ./data_low/behavior_il/stream_seed1
```

### 2. 生成训练脚本

使用 `generate_cl.py` 自动生成持续学习训练脚本：

```bash
python src/generate_cl.py
```

脚本会生成：
- `scripts/cl_behavior_il_seed1/finetune_cl_behavior_il_seed1.sh` 
- `scripts_low/cl_behavior_il_seed1/finetune_cl_behavior_il_seed1.sh` 

### 3. 运行持续学习训练

执行生成的训练脚本：

```bash
bash scripts/cl_behavior_il_seed1/finetune_cl_behavior_il_seed1.sh
```

训练脚本会：
- 按任务顺序依次训练（task0, task1, ...）
- 每个任务使用前一个任务的检查点作为起点
- 使用 LoRA 进行参数高效微调
- 使用 DeepSpeed ZeRO-3 进行分布式训练

### 4. 评估模型

评估持续学习模型性能：

```bash
bash scripts/eval_cl.sh
```

或使用 Python 脚本：

```bash
python -m llava.eval.eval_cl \
    --model-path checkpoints/behavior_il_seed1_task0 \
    --model-base model/llava/llava-v1.5-7b \
    --data ./data \
    --splits valid_unseen \
    --incremental_setup behavior_il \
    --stream_seed 1 \
    --incremental_task 1 \
    --output_file eval_results/behavior_il_seed1_task0_valid_unseen.json \
    --temperature 0.2
```

## 配置说明

### 持续学习设置
见cl-alfred
- **behavior_il**: 7 个任务，按行为类型划分
- **environment_il**: 4 个任务，按环境类型划分

### PEFT 方法

本项目包含修改过的 PEFT 库，支持：
- **LoRA**
- **O-LoRA**
- **SDLoRA**
- 其他 PEFT 方法（AdaLoRA, Prefix Tuning 等）