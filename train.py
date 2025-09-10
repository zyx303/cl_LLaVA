import os
import json
import argparse
from glob import glob
from typing import List, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig  # add
from peft import SDLoraConfig  # new
from peft import L2PConfig  # new
from peft import PeftType  # new
from peft import HidePromptConfig  # new
from peft import InfLoRAConfig
from peft.utils.save_and_load import (
    set_hide_prompt_task_id,
    get_hide_prompt_task_id,
    update_hide_prompt_after_task,
)
from PIL import Image

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates
except ImportError as e:
    raise ImportError("未找到 llava 相关模块，请先安装或把其加入 PYTHONPATH") from e

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    raise ImportError("需要安装 peft: pip install peft")

from torchvision import transforms


def build_image_transform(resolution=336):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

@dataclass
class SampleItem:
    image_path: str
    user_prompt: str
    answer: str

class HighInstrDataset(Dataset):
    def __init__(self, json_files: List[str], image_root: str = ".", image_token="<image>", max_samples: int = None):
        self.items: List[SampleItem] = []
        self.image_root = image_root
        self.image_token = image_token

        for jf in json_files:
            if not os.path.isfile(jf):
                continue
            with open(jf, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[WARN] failed to parse: {jf}")
                    continue
            for entry in data:
                # entry: { id, image, conversations:[{from, value}, {from, value}] }
                if "conversations" not in entry or len(entry["conversations"]) < 2:
                    continue
                user = entry["conversations"][0]["value"]
                ans = entry["conversations"][1]["value"]
                img_rel = entry.get("image", "")
                img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
                self.items.append(SampleItem(img_path, user, ans))

        if max_samples is not None:
            self.items = self.items[:max_samples]

        print(f"[Dataset] loaded samples: {len(self.items)}")

        self.transform = build_image_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # 加载图像
        img = Image.open(it.image_path).convert("RGB")
        img_t = self.transform(img)
        return {
            "image": img_t, 
            "user": it.user_prompt,
            "answer": it.answer
        }

def collate_fn(batch, tokenizer, conv_template_name="llava_v1", image_token_index=None, image_token="<image>", max_length=2048):
    images = [b["image"] for b in batch]
    users = [b["user"] for b in batch]
    answers = [b["answer"] for b in batch]

    conv_template = conv_templates[conv_template_name]
    input_ids_list = []
    labels_list = []

    for u, a in zip(users, answers):
        conv = conv_template.copy()
        # 将原始 user prompt 直接放入(已包含 <image>）
        conv.append_message(conv.roles[0], u)
        conv.append_message(conv.roles[1], a)
        full_text = conv.get_prompt()

        # 处理图像 token
        input_ids = tokenizer_image_token(full_text, tokenizer, image_token, image_token_index, return_tensors='pt').squeeze(0)

        # 构造 labels：除去 assistant 部分之前的 token 设为 -100
        # 简单做法：找到最后一次出现 assistant 回复的文本开头
        # a 在 full_text 中应是最后段落
        # 这里粗略匹配；更精细可用 conv 内部结构
        full_tokenized = input_ids
        answer_ids = tokenizer(a, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0)
        # 在 full_tokenized 中寻找 answer_ids 的结束位置用于对齐
        # 简化处理：将 full_tokenized 末尾 len(answer_ids) 视为答案
        labels = full_tokenized.clone()
        prefix_len = full_tokenized.shape[0] - answer_ids.shape[0]
        labels[:prefix_len] = -100

        # 截断
        if full_tokenized.shape[0] > max_length:
            labels = labels[-max_length:]
            full_tokenized = full_tokenized[-max_length:]

        input_ids_list.append(full_tokenized)
        labels_list.append(labels)

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "images": torch.stack(images, dim=0)
    }

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True, help="预训练 LLaVA / LLaMA 模型目录")
    ap.add_argument("--vision_tower", type=str, default=None, help="若需要手动指定视觉塔路径")
    ap.add_argument("--data_paths", type=str, nargs='+', required=True, help="多个已转换的数据文件或包含文件的目录(支持通配符)")
    ap.add_argument("--image_root", type=str, default=".", help="图片根路径(用于拼接 relative path)")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conv_template", type=str, default="llava_v1")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=20)
    return ap.parse_args()

def expand_paths(path_patterns: List[str]) -> List[str]:
    files = []
    for p in path_patterns:
        if os.path.isdir(p):
            files.extend(sorted(glob(os.path.join(p, "*"))))
        else:
            expanded = glob(p)
            if expanded:
                files.extend(expanded)
            else:
                files.append(p)
    return files

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    data_files = expand_paths(args.data_paths)

    # 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_name_or_path, args.vision_tower, None, device_map="auto"
    )
    # LoRA 准备
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = HighInstrDataset(data_files, image_root=args.image_root, max_samples=args.max_samples)
    
    # 取得图像 token id (依赖 llava tokenizer 已包含)
    image_token = "<image>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_index = tokenizer.convert_tokens_to_ids(image_token) if image_token in tokenizer.get_vocab() else None

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b, tokenizer,
            conv_template_name=args.conv_template,
            image_token_index=image_token_index,
            image_token=image_token,
            max_length=args.max_length
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dl) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0, total_iters=warmup_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    model.train()
    global_step = 0
    device = next(model.parameters()).device

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dl):
            with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32):
                images = batch["images"].to(device)
                # 处理多模态图像特征
                image_tensor = process_images(images, image_processor, device)
                # 将图像特征写入 model 的临时缓存 (llava 方式)
                model_kwargs = {"images": image_tensor, "use_cache": False}
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    labels=batch["labels"].to(device),
                    **model_kwargs
                )
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if global_step < warmup_steps:
                    scheduler.step()

                if global_step % args.log_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch} Step {global_step} Loss {epoch_loss / (global_step + 1):.4f} LR {lr:.2e}")

                if args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"[Save] {save_path}")

                global_step += 1

        # 每个 epoch 结束保存
        save_path = os.path.join(args.output_dir, f"epoch-{epoch}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[Epoch Save] {save_path} epoch_loss={epoch_loss/len(dl):.4f}")

    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[Done] 最终模型保存在: {final_path}")

if __name__ == "__main__":
    main()