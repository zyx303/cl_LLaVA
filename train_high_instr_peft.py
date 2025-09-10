"""
data:
{
  "id": str,
  "image": "path/to/image.jpg",
  "conversations": [ {from: human, value: "<image>..."}, {from: gpt, value: 指令} ]
}
"""
import os
import json
import argparse
from glob import glob
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms

from llava.mm_utils import tokenizer_image_token, process_images
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model

from peft import LoraConfig, get_peft_model


class HighInstrDataset(Dataset):
    # data_files: json file 
    def __init__(self, data_files: List[str], image_root: str = '.', max_samples: int = None, img_res: int = 336):
        self.items = []
        self.image_root = image_root
        for fp in data_files:
            if not os.path.isfile(fp):
                continue
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            for entry in data:
                convs = entry.get('conversations', [])
                if len(convs) < 2:
                    continue
                user_msg = convs[0]['value']
                gpt_msg = convs[1]['value']
                img_rel = entry.get('image', '')
                img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
                self.items.append({'image': img_path, 'user': user_msg, 'answer': gpt_msg})
        if max_samples:
            self.items = self.items[:max_samples]
        print(f"[Dataset] 样本数: {len(self.items)}")
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = rec['image']
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            print(f"[Error] 无法打开图像: {img_path}")
            img = Image.new('RGB', (336, 336), (0, 0, 0))
        return {'image': img, 'user': rec['user'], 'answer': rec['answer']}

def build_collate(tokenizer, conv_template: str = 'llava_v1', max_length: int = 2048, image_token: str = '<image>'):
    template = conv_templates[conv_template]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def _collate(batch):
        images = [b['image'] for b in batch]
        in_list, lab_list = [], []
        for b in batch:
            conv = template.copy()
            conv.append_message(conv.roles[0], b['user'])
            conv.append_message(conv.roles[1], b['answer'])
            prompt = conv.get_prompt()
            ids = tokenizer_image_token(prompt, tokenizer, image_token, return_tensors='pt').squeeze(0)
            ans_ids = tokenizer(b['answer'], return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0)
            labels = ids.clone()
            prefix_len = ids.shape[0] - ans_ids.shape[0]
            if prefix_len < 0:
                prefix_len = 0
            labels[:prefix_len] = -100
            if ids.shape[0] > max_length:
                ids = ids[-max_length:]
                labels = labels[-max_length:]
            in_list.append(ids)
            lab_list.append(labels)
        input_ids = pad_sequence(in_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(lab_list, batch_first=True, padding_value=-100)
        return {'input_ids': input_ids, 'labels': labels, 'images': torch.stack(images, dim=0)}
    return _collate

def find_linear_module_names(model):
    vision_keywords = ['vision_tower', 'mm_projector', 'visual']
    names = set()
    for n, m in model.named_modules():
        if any(k in n for k in vision_keywords):
            continue
        if isinstance(m, torch.nn.Linear):
            leaf = n.split('.')[-1]
            names.add(leaf)
    if 'lm_head' in names:
        names.remove('lm_head')
    return list(names)

def expand_data_paths(patterns: List[str]) -> List[str]:
    files = []
    for p in patterns:
        if os.path.isdir(p):
            for f in sorted(glob(os.path.join(p, '*'))):
                if os.path.isfile(f):
                    files.append(f)
        else:
            g = glob(p)
            if g:
                files.extend([x for x in g if os.path.isfile(x)])
            elif os.path.isfile(p):
                files.append(p)
    print(f"[Data] 匹配文件: {len(files)}")
    return files

def train_loop(args):
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_name_or_path, args.vision_tower, None, device_map='auto')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_modules = find_linear_module_names(model)
    print(f"[LoRA] target_modules: {target_modules}")
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias='none')
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    data_files = expand_data_paths(args.data_paths)
    dataset = HighInstrDataset(data_files, image_root=args.image_root, max_samples=args.max_samples, img_res=args.image_res)
    collate_fn = build_collate(tokenizer, args.conv_template, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = (len(loader) * args.epochs) // args.grad_accum
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0, total_iters=warmup_steps) if warmup_steps > 0 else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        run_loss = 0.0
        for step, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=args.fp16, dtype=torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32):
                images = process_images(batch['images'], image_processor, device)
                out = model(input_ids=batch['input_ids'].to(device), labels=batch['labels'].to(device), images=images, use_cache=False)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler and global_step < warmup_steps:
                    scheduler.step()
                if global_step % args.log_steps == 0:
                    lr = optimizer.param_groups[0]['lr']
                    avg = run_loss / (global_step + 1e-9)
                    print(f"Epoch {epoch} Step {global_step} Loss {avg:.4f} LR {lr:.2e}")
                if args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f"ckpt-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"[Save] {ckpt}")
                global_step += 1
        ep_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
        os.makedirs(ep_dir, exist_ok=True)
        model.save_pretrained(ep_dir)
        tokenizer.save_pretrained(ep_dir)
        print(f"[Epoch {epoch}] 保存 -> {ep_dir}")
    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[Done] 最终: {final_dir}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', type=str, required=True,default="model/llava-v1.5-7b")
    ap.add_argument('--vision_tower', type=str, default="openai/clip-vit-large-patch14-336")
    ap.add_argument('--data_paths', type=str, default="")
    ap.add_argument('--image_root', type=str, default='.')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--max_length', type=int, default=2048)
    ap.add_argument('--max_samples', type=int, default=None)
    ap.add_argument('--image_res', type=int, default=336)
    ap.add_argument('--lora_r', type=int, default=64)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--conv_template', type=str, default='llava_v1')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--save_steps', type=int, default=1000)
    ap.add_argument('--log_steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train_loop(args)

if __name__ == '__main__':
    main()
