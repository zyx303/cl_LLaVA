# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F

from ..utils import PeftType, PromptLearningConfig


class L2PInit(str, enum.Enum):
    RANDOM = "RANDOM"
    UNIFORM = "UNIFORM"


@dataclass
class L2PConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`L2PPromptPool`].

    Args:
        pool_size (`int`): The size of the prompt pool.
        prompt_init (Union[[`L2PInit`], `str`]): The initialization method for prompts.
        top_k (`int`): The number of top prompts to select based on similarity.
        shared_prompt_pool (`bool`): Whether to share prompt pool across tasks.
        shared_prompt_key (`bool`): Whether to share prompt keys across tasks.
        prompt_key_init (`str`): Initialization method for prompt keys.
        ortho_mu (`float`): Orthogonal regularization coefficient.
        sim_coefficient (`float`): Similarity loss coefficient.
        pull_constraint (`bool`): Whether to use pull constraint.
        pull_constraint_coeff (`float`): Pull constraint coefficient.
    """

    pool_size: int = field(
        default=20,
        metadata={"help": "The size of the prompt pool"},
    )
    prompt_init: Union[L2PInit, str] = field(
        default=L2PInit.UNIFORM,
        metadata={"help": "How to initialize the prompt pool"},
    )
    top_k: int = field(
        default=5,
        metadata={"help": "The number of top prompts to select"},
    )
    shared_prompt_pool: bool = field(
        default=True,
        metadata={"help": "Whether to share prompt pool across tasks"},
    )
    shared_prompt_key: bool = field(
        default=True,
        metadata={"help": "Whether to share prompt keys across tasks"},
    )
    prompt_key_init: str = field(
        default="uniform",
        metadata={"help": "Initialization method for prompt keys"},
    )
    ortho_mu: float = field(
        default=0.1,
        metadata={"help": "Orthogonal regularization coefficient"},
    )
    sim_coefficient: float = field(
        default=0.1,
        metadata={"help": "Similarity loss coefficient"},
    )
    pull_constraint: bool = field(
        default=True,
        metadata={"help": "Whether to use pull constraint"},
    )
    pull_constraint_coeff: float = field(
        default=0.1,
        metadata={"help": "Pull constraint coefficient"},
    )
    # Extra toggles for comparison and compatibility with HiDe-Prompt
    use_hide_prompt: bool = field(
        default=False,
        metadata={"help": "Use HiDe-Prompt wrapper instead of L2P pool when True"},
    )
    prompt_key: bool = field(
        default=True,
        metadata={"help": "Use learnable prompt keys (HiDe-compatible)"},
    )
    batchwise_prompt: bool = field(
        default=False,
        metadata={"help": "Select prompts batch-wise (HiDe option)"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.L2P


class L2PPromptPool(torch.nn.Module):
    """
    The L2P (Learning to Prompt) prompt pool module for continual learning.

    Args:
        config ([`L2PConfig`]): The configuration of the L2P prompt pool.

    Example:

    ```py
    >>> from peft import L2PPromptPool, L2PConfig

    >>> config = L2PConfig(
    ...     peft_type="L2P",
    ...     task_type="SEQ_CLS",
    ...     num_virtual_tokens=5,
    ...     token_dim=768,
    ...     pool_size=10,
    ...     top_k=5,
    ... )
    >>> l2p_pool = L2PPromptPool(config)
    ```

    **Attributes**:
        - **prompt** (`torch.nn.Parameter`) -- The prompt pool parameters.
        - **prompt_key** (`torch.nn.Parameter`) -- The prompt key parameters for selection.

    Input shape: (`batch_size`, `hidden_size`) for query

    Output shape: (`batch_size`, `num_virtual_tokens`, `token_dim`) for selected prompts
    """

    def __init__(self, config):
        super().__init__()
        self.pool_size = config.pool_size
        self.top_k = config.top_k
        self.num_virtual_tokens = config.num_virtual_tokens
        self.token_dim = config.token_dim
        self.ortho_mu = config.ortho_mu
        self.sim_coefficient = config.sim_coefficient
        self.pull_constraint = config.pull_constraint
        self.pull_constraint_coeff = config.pull_constraint_coeff

        # Initialize prompt pool (align with prompt.Prompt: uniform init over [0,1] by default)
        if str(config.prompt_init).upper().endswith("UNIFORM"):
            self.prompt = torch.nn.Parameter(
                torch.randn(self.pool_size, self.num_virtual_tokens, self.token_dim)
            )
            torch.nn.init.uniform_(self.prompt, -1, 1)
        else:  # zero/random fallback similar to original variants
            self.prompt = torch.nn.Parameter(
                torch.zeros(self.pool_size, self.num_virtual_tokens, self.token_dim)
            )

        # Initialize prompt keys for selection
        if config.prompt_key_init == "uniform":
            self.prompt_key = torch.nn.Parameter(
                torch.zeros(self.pool_size, self.token_dim)
            )
            torch.nn.init.uniform_(self.prompt_key.data, -1, 1)
        else:
            self.prompt_key = torch.nn.Parameter(
                torch.randn(self.pool_size, self.token_dim)
            )
            torch.nn.init.xavier_uniform_(self.prompt_key.data)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """L2 normalize"""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        # 保证 epsilon 与 x 的 dtype/device 一致，避免混合精度/多卡下的 dtype 问题
        eps = torch.tensor(epsilon, dtype=x.dtype, device=x.device)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, eps))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None, task_id=None, train=False):
        """
        Forward pass for L2P prompt selection and retrieval.
        
        Args:
            x_embed: Input embeddings (batch_size, seq_len, hidden_size)
            prompt_mask: Optional mask for prompt selection
            cls_features: CLS token features for prompt selection (batch_size, hidden_size)  
            task_id: Current task ID for continual learning
            train: Whether in training mode
            
        Returns:
            Dict containing:
                - selected_prompts: Selected prompt embeddings
                - reduce_sim: Similarity reduction loss
                - similarities: Similarity scores for analysis
        """
        batch_size = x_embed.shape[0]
        
        # Use CLS features for prompt selection if available, otherwise use mean pooling
        if cls_features is not None:
            query_features = cls_features
        else:
            # Use mean of input embeddings as query
            query_features = torch.mean(x_embed, dim=1)  # (batch_size, hidden_size)
        
        # Normalize query and prompt keys
        query_norm = self.l2_normalize(query_features, dim=1)  # (batch_size, hidden_size)
        prompt_key_norm = self.l2_normalize(self.prompt_key, dim=1)  # (pool_size, hidden_size)
        
        # Compute similarity between query and prompt keys
        similarity = torch.matmul(query_norm, prompt_key_norm.t())  # (batch_size, pool_size)
        # Select top-k prompts based on similarity
        if self.top_k == -1:
            top_k = self.pool_size
        else:
            top_k = min(self.top_k, self.pool_size)
        # 更稳的 mask：使用 -inf 屏蔽被禁用的 prompt，且显式对齐维度/类型
        if prompt_mask is not None:
            # 允许 mask 形状为 (pool,) 或 (B, pool)
            if prompt_mask.dim() == 1:
                prompt_mask = prompt_mask.unsqueeze(0).expand_as(similarity)
            else:
                assert prompt_mask.shape == similarity.shape, "prompt_mask shape must be (B, pool_size)"
            prompt_mask = prompt_mask.to(dtype=similarity.dtype, device=similarity.device)
            similarity = similarity.masked_fill(prompt_mask <= 0, float("-inf"))

        # Get top-k indices
        _, top_indices = torch.topk(similarity, top_k, dim=1)  # (batch_size, top_k)

        # 使用 gather 代替高级索引，减少隐式广播/复制带来的问题
        # prompts: (1, pool, V, C) -> (B, pool, V, C)
        prompts = self.prompt.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # indices: (B, top_k, 1, 1) -> (B, top_k, V, C)
        index = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_virtual_tokens, self.token_dim)
        batched_prompt_raw = torch.gather(prompts, 1, index)  # (B, top_k, V, C)
        bsz, k, V, C = batched_prompt_raw.shape
        selected_prompts = batched_prompt_raw.reshape(bsz, k * V, C)

        # Compute reduce_sim
        batched_key_norm = prompt_key_norm[top_indices]  # (B, top_k, C)
        x_embed_norm = query_norm  # (B, C)
        sim = batched_key_norm * x_embed_norm.unsqueeze(1)  # (B, top_k, C)
        reduce_sim = torch.sum(sim) / batch_size

        return {
            "selected_prompts": selected_prompts, # (B, top_k*V, C)
            "reduce_sim": reduce_sim,
            "similarities": similarity,
            "top_indices": top_indices,
            "selected_key": batched_key_norm,
            "prompt_key_norm": prompt_key_norm,
            "x_embed_norm": x_embed_norm,
        }

    def get_prompt_params(self):
        """Get prompt parameters for optimization"""
        return [self.prompt, self.prompt_key]
        
    def copy_prompts(self, src_indices, dst_indices):
        """Copy prompts from source indices to destination indices"""
        with torch.no_grad():
            self.prompt[dst_indices] = self.prompt[src_indices].clone()
            self.prompt_key[dst_indices] = self.prompt_key[src_indices].clone()
    
    def init_task_prompts(self, task_id, top_k, optimizer=None):
        """
        Initialize prompts for a new task by copying from previous task.
        This follows the PILOT implementation pattern.
        
        Args:
            task_id: Current task ID (0-indexed)
            top_k: Number of prompts per task
            optimizer: Optimizer to update after copying prompts
        """
        if task_id == 0:
            # First task, no copying needed
            return
            
        # Calculate indices for previous and current task
        prev_start = (task_id - 1) * top_k
        prev_end = task_id * top_k
        
        cur_start = prev_end  
        cur_end = (task_id + 1) * top_k
        
        # Check if indices are within pool size
        if (prev_end > self.pool_size) or (cur_end > self.pool_size):
            print(f"Warning: Task {task_id} requires {cur_end} prompts but pool size is {self.pool_size}")
            return
            
        # Copy prompts and keys from previous task
        prev_idx = slice(prev_start, prev_end)
        cur_idx = slice(cur_start, cur_end)
        
        with torch.no_grad():
            # Clear gradients first
            if self.prompt.grad is not None:
                self.prompt.grad.zero_()
            if self.prompt_key.grad is not None:
                self.prompt_key.grad.zero_()
                
            # Copy prompt parameters
            self.prompt[cur_idx] = self.prompt[prev_idx].clone()
            self.prompt_key[cur_idx] = self.prompt_key[prev_idx].clone()
            
            # Update optimizer parameters if provided
            if optimizer is not None:
                # Refresh optimizer's parameter groups
                for group in optimizer.param_groups:
                    group['params'] = [p for p in group['params'] if p.requires_grad]
    
    def apply_prompt_transfer(self, prev_task_id, curr_task_id, top_k):
        """
        Apply prompt transfer from previous task to current task.
        This is typically called when loading a model for a new task.
        
        Args:
            prev_task_id: Previous task ID
            curr_task_id: Current task ID  
            top_k: Number of prompts per task
        """
        if prev_task_id < 0 or curr_task_id <= prev_task_id:
            print(f"Invalid task IDs: prev={prev_task_id}, curr={curr_task_id}")
            return
            
        # Calculate indices
        prev_start = prev_task_id * top_k
        prev_end = (prev_task_id + 1) * top_k
        
        curr_start = curr_task_id * top_k
        curr_end = (curr_task_id + 1) * top_k
        
        # Check bounds
        if curr_end > self.pool_size:
            print(f"Warning: Task {curr_task_id} requires {curr_end} prompts but pool size is {self.pool_size}")
            return
            
        if prev_end > self.pool_size:
            print(f"Warning: Previous task {prev_task_id} range exceeds pool size")
            return
        
        # Transfer prompts
        prev_idx = slice(prev_start, prev_end)
        curr_idx = slice(curr_start, curr_end)
        
        with torch.no_grad():
            self.prompt[curr_idx] = self.prompt[prev_idx].clone()
            self.prompt_key[curr_idx] = self.prompt_key[prev_idx].clone()
            
        print(f"Transferred prompts from task {prev_task_id} to task {curr_task_id}")
        print(f"  Source range: [{prev_start}:{prev_end}]")
        print(f"  Target range: [{curr_start}:{curr_end}]")
