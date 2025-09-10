import os
import sys
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.method_manager import select_method

# Add project root to path
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'models'))

from models.model.vlm_instruction_predictor import Module
from data.preprocess import Dataset


def create_model(args):
    """Create and initialize the VLM model"""
    
    # Load or create vocabulary
    vocab_path = os.path.join(args.data, 'pp.vocab')
    vocab = torch.load(vocab_path)
    
    # Create model
    model = Module(args, vocab)
    
    return model, vocab


def prepare_data(args):
    """Prepare training and validation data"""
    
    # Load data splits
    splits_path = args.splits if hasattr(args, 'splits') else 'data/splits/oct21.json'
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    print(f"Loaded splits: {list(splits.keys())}")
    for split, data in splits.items():
        print(f"  {split}: {len(data)} samples")
    
    return splits


def create_data_loader(model, splits, split_name, batch_size=4, shuffle=True):
    """Create data loader for a specific split"""
    
    if split_name not in splits:
        print(f"Warning: Split '{split_name}' not found in data")
        return None
    
    data = splits[split_name]
    
    # Limit data for testing
    if hasattr(model.args, 'fast_epoch') and model.args.fast_epoch:
        data = data[:16]
    
    # Create simple data loader
    def collate_fn(batch):
        # batch is a list of trajectory data
        return batch
    
    # Convert to list of (traj_data, swapColor) tuples
    dataset = [(item, False) for item in data]
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    
    return loader


# def validate(model, data_loader, device):
#     """Validate the model"""
    
#     model.eval()
#     total_loss = 0
#     num_batches = 0
#     predictions = []
    
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Validating"):
#             try:
#                 # Featurize batch
#                 feat = model.featurize(batch)
                
#                 # Skip if no valid data
#                 if not feat['goal_texts'] or len(feat['target_instrs']) == 0:
#                     continue
                
#                 # Forward pass
#                 outputs = model.forward(feat)
                
#                 # Compute loss
#                 loss_dict = model.compute_loss(outputs, batch, feat)
#                 loss = loss_dict['total_loss']
                
#                 total_loss += loss.item()
#                 num_batches += 1
                
#                 # # Extract predictions
#                 # preds = model.extract_preds(outputs, batch, feat)
#                 # predictions.extend(preds['predicted_instructions'])
                
#             except Exception as e:
#                 print(f"Error in validation batch: {e}")
#                 continue
    
#     avg_loss = total_loss / max(num_batches, 1)
    
#     return avg_loss, predictions

import debugpy
def load_task_json(model, task):
    '''
    load preprocessed json from disk
    '''
    # task['task']: "look_at_obj_in_light-Watch-None-FloorLamp-203/trial_T20190908_222015_998221"}

    # data/full/json_feat_2.1.0/full_2.1.0/train/look_at_obj_in_light-Watch-None-FloorLamp-203/trial_T20190908_222015_998221/traj_data.json
    json_path = os.path.join(model.args.data, task['task'], '%s' % model.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
    # json_path = os.path.join(model.args.data, task['task'], '%s' % model.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
    full_json_path = os.path.join('data/full/json_feat_2.1.0', 'train',task['task'], 'traj_data.json')
    with open(full_json_path) as f:
        full_data = json.load(f)
    with open(json_path) as f:
        data = json.load(f)
    data['images'] = full_data['images']
    data['full_path'] = full_json_path
    return data



def main():
    if os.getenv('debug'): 
        debugpy.listen(5678)
        debugpy.wait_for_client()
    
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data/json_feat_2.1.0', help='Dataset directory')
    parser.add_argument('--data_full', type=str, default='data/full/json_feat_2.1.0/full_2.1.0', help='full Dataset directory')
    parser.add_argument('--splits', type=str, default='data/splits/oct21.json', help='Data splits file')
    
    # 添加持续学习相关参数
    parser.add_argument('--incremental_setup', type=str, 
                       choices=['behavior_il','behavior_il_test', 'environment_il', 'environment_il_nosampling'],
                       default='behavior_il', help='Incremental learning setup')
    parser.add_argument('--n_tasks', type=int, help='Number of tasks')
    parser.add_argument('--stream_seed', type=int, default=1, help='Random seed for stream')
    parser.add_argument('--memory_size', type=int, default=500, help='Memory size')
    parser.add_argument('--mode', type=str, default='base', help='Continual learning method')
    parser.add_argument('--epochs_per_task', type=int, default=3, help='Epochs per task')
    parser.add_argument('--pp_folder', type=str, default='pp', help='Preprocessed folder name')
    parser.add_argument('--data_dir', type=str, help='Data directory for CL')
    
    # Model parameters
    # parser.add_argument('--model_name', type=str, default='models.model.vlm_instruction_predictor', help='Model module')
    parser.add_argument('--llama_model_name', type=str, default='initial_model/llama', help='LLaMA model name')
    parser.add_argument('--llama_dtype', type=str, default='float16', help='LLaMA dtype')
    parser.add_argument('--llama_max_length', type=int, default=512, help='LLaMA max sequence length')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (for traditional training)')
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument('--save_every_epoch', action='store_true', help='Save model every epoch')
    
    # Architecture parameters
    parser.add_argument('--fusion_dim', type=int, default=512, help='Fusion layer dimension')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_fusion_layers', type=int, default=2, help='Number of fusion transformer layers')
    parser.add_argument('--decoder_hidden_size', type=int, default=512, help='Decoder hidden size')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--max_decode_length', type=int, default=50, help='Maximum decode length')
    
    # Dropout parameters
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Attention dropout')
    parser.add_argument('--fusion_dropout', type=float, default=0.1, help='Fusion dropout')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='Decoder dropout')
    
    # System parameters
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dout', type=str, default='exp/vlm_instr_predictor', help='Output directory')
    parser.add_argument('--fast_epoch', action='store_true', help='Use small dataset for testing')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # 添加训练模式选择
    parser.add_argument('--training_mode', type=str, choices=['traditional', 'continual'], 
                       default='continual', help='Training mode: traditional or continual learning')
    
    args = parser.parse_args()
    
    # 根据 incremental_setup 设置 n_tasks
    if args.incremental_setup in ['behavior_il','behavior_il_test']:
        args.n_tasks = 7
    elif args.incremental_setup in ['environment_il', 'environment_il_nosampling']:
        args.n_tasks = 4
    else:
        raise Exception("Invalid incremental_setup. Should be 'behavior_il', 'behavior_il_test', 'environment_il', or 'environment_il_nosampling'.")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.dout, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model, vocab = create_model(args)
    
    # Print model info
    total_params = sum(p.numel() for p in model.vlm_model.parameters())
    trainable_params = sum(p.numel() for p in model.vlm_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # dump config
    fconfig = os.path.join(args.dout, 'config.json')
    with open(fconfig, 'wt') as f:
        json.dump(vars(args), f, indent=4)

    # display dout
    print("Saving to: %s" % args.dout)
    
    samples_cnt = 0
    
    
    # 检查 embodied_split 目录是否存在
    embodied_split_dir = f'embodied_split/{args.incremental_setup}'
    if not os.path.exists(embodied_split_dir):
        raise Exception(f"Embodied split directory not found: {embodied_split_dir}")
    
    cl_method = select_method(args=args, n_classes=args.n_tasks, model=model)

    # 验证集（保持原格式）
    valid_seen_path = f'embodied_split/{args.incremental_setup}/valid_seen.json'
    valid_unseen_path = f'embodied_split/{args.incremental_setup}/valid_unseen.json'
    
    if os.path.exists(valid_seen_path):
        test_datalist_seen = json.load(open(valid_seen_path, 'r'))
        test_datalist_seen = [(s, False) for s in test_datalist_seen]
    else:
        print(f"Warning: {valid_seen_path} not found")
        test_datalist_seen = []
        
    if os.path.exists(valid_unseen_path):
        test_datalist_unseen = json.load(open(valid_unseen_path, 'r'))
        test_datalist_unseen = [(s, False) for s in test_datalist_unseen]
    else:
        print(f"Warning: {valid_unseen_path} not found")
        test_datalist_unseen = []
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(args.dout, 'logs'))
    
    
    for cur_iter in range(args.n_tasks):
        print(f"\n=== Task {cur_iter+1}/{args.n_tasks} ===")
        
        # 加载当前任务数据
        task_data_path = f'embodied_split/{args.incremental_setup}/embodied_data_disjoint_rand{args.stream_seed}_cls1_task{cur_iter}.json'
        
        if not os.path.exists(task_data_path):
            print(f"Warning: Task data file not found: {task_data_path}")
            continue
            
        cur_train_datalist = json.load(open(task_data_path, 'r'))
        print(f"Loaded {len(cur_train_datalist)} samples for task {cur_iter}")

        # 预先补齐每个样本的 frame 统计，供日志/可视化使用
        for d in cur_train_datalist:
            if 'num_frames' not in d:
                try:
                    traj_data = load_task_json(model, d['task'])
                    d['num_frames'] = len([aa for a in traj_data['num']['action_low'] for aa in a])
                except:
                    d['num_frames'] = 0  # 默认值

        # 若出现新类别，注册到方法管理器（影响 MemoryDataset 的标签映射）
        for d in cur_train_datalist:
            if d['klass'] not in cl_method.exposed_classes:
                cl_method.add_new_class(d['klass'])

        # 同步 memory 的类别映射（即使 memory 为空，也需包含当前所有已曝光类别）
        if hasattr(cl_method, 'memory'):
            mem = cl_method.memory
            # 更新类名列表与映射
            mem.cls_list = list(cl_method.exposed_classes)
            mem.cls_dict = {mem.cls_list[i]: i for i in range(len(mem.cls_list))}
            # 确保计数结构长度匹配
            needed = len(mem.cls_list) - len(mem.cls_count)
            for _ in range(max(0, needed)):
                mem.cls_count.append(0)
                mem.cls_idx.append([])
                mem.cls_train_cnt = np.append(mem.cls_train_cnt, 0)

        # 构建 DataLoader（stream 批次）
        from utils.data_loader import StreamDataset
        
        # memory 不为空时调整 batch size
        if hasattr(cl_method, 'memory') and len(cl_method.memory) > 0:
            stream_batch_size = args.batch_size // 2
        else:
            stream_batch_size = args.batch_size

        stream_ds = StreamDataset(datalist=cur_train_datalist,
                                    cls_list=cl_method.exposed_classes,
                                    data_dir=getattr(args, 'data_dir', None))
        stream_loader = DataLoader(stream_ds, batch_size=stream_batch_size, shuffle=True,
                                    drop_last=False, collate_fn=lambda x: x)

        # 为了能把 (task, swapColor) 还原为完整 sample，这里建映射
        task_to_sample = {json.dumps(s['task'], sort_keys=True): s for s in cur_train_datalist}

        # 任务前钩子
        if hasattr(cl_method, 'online_before_task'):
            cl_method.online_before_task(cur_iter)

        # training 
        for epoch in range(args.epochs_per_task):
            model.train()
            pbar = tqdm(stream_loader, desc=f"Task {cur_iter+1} Epoch {epoch+1}/{args.epochs_per_task}")
            
            for stream_batch in pbar:
                # stream_batch: List[(task_dict, 0)]
                stream_batch = list(stream_batch)
                bs_stream = len(stream_batch)

                # 组装最终 batch：stream + memory 回放
                data = []
                if bs_stream > 0:
                    data += stream_batch

                memory_batch_size = max(0, args.batch_size - bs_stream)
                if hasattr(cl_method, 'memory') and memory_batch_size > 0 and len(cl_method.memory) > 0:
                    memory_data = cl_method.memory.get_batch(memory_batch_size)
                    data += memory_data['batch']

                if len(data) == 0:
                    continue

                # 前向 & 反向
                batch = [(load_task_json(model, task), swapColor) for task, swapColor in data]
                feat = model.featurize(batch)
                
                # 跳过空批次
                if not feat.get('goal_texts') or len(feat.get('target_instrs', [])) == 0:
                    continue
                    
                out = model.forward(feat)

                cl_method.optimizer.zero_grad()
                loss_dict = model.compute_loss(out, batch, feat)
                sum_loss = sum(loss_dict.values())

                # 可选：方法级正则（如 EWC）
                if hasattr(cl_method, 'regularization_loss'):
                    reg_loss = cl_method.regularization_loss()
                    sum_loss = sum_loss + reg_loss

                # 记录旧参数/梯度（用于 EWC 的 Fisher/score 更新）
                if hasattr(cl_method, 'update_fisher_and_score'):
                    old_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad} if hasattr(cl_method, 'update_fisher_and_score') else None
                    old_grads = {n: (p.grad.clone().detach() if p.grad is not None else None) for n, p in model.named_parameters() if p.requires_grad} if hasattr(cl_method, 'update_fisher_and_score') else None

                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                cl_method.optimizer.step()

                # 更新 Fisher 与 score
                if hasattr(cl_method, 'update_fisher_and_score') and old_params is not None:
                    new_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
                    new_grads = {n: (p.grad.clone().detach() if p.grad is not None else None) for n, p in model.named_parameters() if p.requires_grad}
                    # 过滤 None 梯度
                    new_grads = {k: v for k, v in new_grads.items() if v is not None}
                    old_grads = {k: v for k, v in old_grads.items() if v is not None}
                    try:
                        cl_method.update_fisher_and_score(new_params, old_params, new_grads, old_grads)
                    except TypeError:
                        # 与具体方法签名不匹配则跳过
                        pass
                        
                if hasattr(cl_method, 'update_schedule'):
                    cl_method.update_schedule()

                samples_cnt += bs_stream

                # 训练日志（与 ER.report_training 对齐的 key）
                if hasattr(cl_method, 'report_training'):
                    cl_method.report_training(samples_cnt, {'cls_loss': float(sum_loss.detach().cpu())})

                # 更新记忆库（仅用 stream 样本）
                if hasattr(cl_method, 'update_memory'):
                    for task, _ in stream_batch:
                        key = json.dumps(task, sort_keys=True)
                        sample = task_to_sample.get(key)
                        if sample is not None:
                            cl_method.update_memory(sample)

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{float(sum_loss.detach().cpu()):.4f}",
                    'samples': samples_cnt
                })
                    
        # 任务后钩子
        if hasattr(cl_method, 'online_after_task'):
            cl_method.online_after_task(cur_iter)

        # 保存检查点
        last_klass = cur_train_datalist[-1]['klass'] if len(cur_train_datalist) else f'task{cur_iter}'
        torch.save({
            'metric': {'samples_cnt': samples_cnt},
            'model': model.state_dict(),
            'optim': cl_method.optimizer.state_dict(),
            'args': model.args,
            'vocab': model.vocab,
            'r_sum': model.r_sum if hasattr(model, 'r_sum') else 0,
        }, os.path.join(args.dout, 'net_epoch_%09d_%s.pth' % (samples_cnt, last_klass)))
        
        writer.close()
        print(f"\nContinual learning completed!")
    
    print(f"Models saved in: {args.dout}")


if __name__ == '__main__':
    main()
