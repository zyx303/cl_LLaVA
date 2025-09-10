"""
Continual Learning Method Manager
管理不同持续学习方法的选择和实例化
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List


class BaseCLMethod:
    """持续学习方法基类"""
    
    def __init__(self, args, n_classes: int, model):
        self.args = args
        self.n_classes = n_classes
        self.model = model
        self.exposed_classes = set()
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=getattr(args, 'lr', 1e-4)
        )
    
    def add_new_class(self, class_name: str):
        """添加新类别"""
        self.exposed_classes.add(class_name)
    
    def online_before_task(self, task_id: int):
        """任务开始前的钩子"""
        pass
    
    def online_after_task(self, task_id: int):
        """任务结束后的钩子"""
        pass
    
    def update_memory(self, sample):
        """更新记忆库"""
        pass


class ERMethod(BaseCLMethod):
    """Experience Replay方法"""
    
    def __init__(self, args, n_classes: int, model):
        super().__init__(args, n_classes, model)
        
        # 创建记忆库
        from .memory import MemoryDataset
        self.memory = MemoryDataset(
            dataset_size=getattr(args, 'memory_size', 500),
            train_transform=None,
            cls_list=[],
            device=torch.device('cuda' if args.gpu else 'cpu')
        )
    
    def update_memory(self, sample):
        """更新记忆库"""
        if hasattr(self.memory, 'add_new_class'):
            self.memory.add_new_class(sample['klass'])
        
        # 添加样本到记忆库
        self.memory.replace_sample(sample)
    
    def report_training(self, sample_num: int, train_loss: Dict[str, float]):
        """报告训练进度"""
        print(f"Sample {sample_num}: Loss = {train_loss.get('cls_loss', 0.0):.4f}")


class EWCMethod(BaseCLMethod):
    """Elastic Weight Consolidation方法"""
    
    def __init__(self, args, n_classes: int, model):
        super().__init__(args, n_classes, model)
        self.fisher_information = {}
        self.old_params = {}
        self.ewc_lambda = getattr(args, 'ewc_lambda', 1000)
    
    def update_fisher_and_score(self, new_params, old_params, new_grads, old_grads):
        """更新Fisher信息矩阵"""
        for name in new_grads:
            if name not in self.fisher_information:
                self.fisher_information[name] = torch.zeros_like(new_grads[name])
            
            # 累积Fisher信息
            self.fisher_information[name] += new_grads[name] ** 2
    
    def regularization_loss(self):
        """计算EWC正则化损失"""
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.old_params:
                reg_loss += (self.fisher_information[name] * 
                           (param - self.old_params[name]) ** 2).sum()
        return self.ewc_lambda * reg_loss
    
    def online_after_task(self, task_id: int):
        """任务结束后保存参数"""
        self.old_params = {name: param.clone().detach() 
                          for name, param in self.model.named_parameters()}


def select_method(args, n_classes: int, model) -> BaseCLMethod:
    """选择持续学习方法"""
    method_name = getattr(args, 'mode', 'base').lower()
    
    if method_name in ['er', 'experience_replay']:
        return ERMethod(args, n_classes, model)
    elif method_name in ['ewc', 'elastic_weight_consolidation']:
        return EWCMethod(args, n_classes, model)
    else:
        # 默认基础方法
        return BaseCLMethod(args, n_classes, model)
