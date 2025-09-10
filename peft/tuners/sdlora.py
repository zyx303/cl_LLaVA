"""
State-Decoupled LoRA (SDLora) tuner for PEFT in O-LoRA.

This implementation mirrors the existing modified LoRA in this repo (which already
supports loranew_A/loranew_B for incremental tasks). We expose it as a separate
PeftType/Config/Model so it can be selected explicitly (e.g., via a flag),
without affecting the default LoRA path.

Key points:
- Keep pretrained base weights frozen.
- Keep previously learned lora_A/B (if any) frozen.
- Train only loranew_A/B for the current task and sum both contributions at forward.
- Save/Load supports loranew_* alongside lora_* (re-uses existing save_and_load logic).
"""

import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)


try:
    if is_bnb_available():
        import bitsandbytes as bnb  # type: ignore
    else:
        bnb = None  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bnb = None  # type: ignore


@dataclass
class SDLoraConfig(PeftConfig):
    """
    Configuration for SDLoraModel.

    Args:
        r (int): LoRA rank for the current task (trained in loranew_*).
        target_modules (List[str] | str): Module names (or regex) to apply LoRA.
        lora_alpha (float): Alpha for scaling.
        lora_dropout (float): Dropout for LoRA.
        fan_in_fan_out (bool): Set True for Conv1D(GPT-2) style weights.
        bias (str): 'none' | 'all' | 'lora_only'.
        modules_to_save (List[str]): Extra non-LoRA modules to keep trainable and save.
        init_lora_weights (bool): Whether to initialize LoRA weights.
        r_sum (int): Dim of concatenated previous LoRA (kept for compatibility; optional).
        save_loranew (bool): Whether to save loranew_* separately instead of concatenating into lora_*.
    """

    r: int = field(default=8, metadata={"help": "LoRA rank for current task"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List/regex of module names to replace with LoRA. e.g., ['q','v'] or regex."
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "LoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "True if target layer stores weight as (in, out) like Conv1D"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type: none | all | lora_only"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Extra modules (besides LoRA) to be trainable and saved in checkpoint."
        },
    )
    init_lora_weights: bool = field(
        default=True, metadata={"help": "Whether to initialize LoRA weights."}
    )
    r_sum: int = field(default=0, metadata={"help": "Dim of previous LoRA concat; optional."})
    save_loranew: bool = field(
        default=False,
        metadata={"help": "If True, save loranew_* separately instead of concatenating into lora_*."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.SDLORA


class SDLoraModel(torch.nn.Module):
    """
    SDLora wraps a base HF model and replaces target modules with PEFT LoRA layers that
    have both fixed (previous) lora_* and trainable loranew_*.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "SDLoraModel supports only 1 adapter with bias. For multiple adapters, set bias='none'."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use LoRA with 8-bit quantization, please install the `bitsandbytes` package."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                        lora_config.r_sum,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
                        )
                    elif isinstance(target, torch.nn.Embedding):
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = target.num_embeddings, target.embedding_dim
                        new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out=True but target is torch.nn.Linear; setting to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out=False but target is Conv1D; setting to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported (only Linear/Conv1D supported)."
                            )
                        new_module = Linear(
                            adapter_name, in_features, out_features, bias=bias, r_sum=lora_config.r_sum, **kwargs
                        )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias") and old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when merged. Unmerging first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LoRA layers")
        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge LoRA layers when model is loaded in 8-bit mode")
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        return self.model

    def consolidate_lora_directions(self):
        """
        Consolidate current task's LoRA directions into the historical directions.
        """
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                adapter_name = module.active_adapter
                if adapter_name in module.loranew_A and adapter_name in module.lora_A:
                    current_A = module.loranew_A[adapter_name].weight.detach().clone()
                    current_B = module.loranew_B[adapter_name].weight.detach().clone()
                    
                    # Add current directions as a new historical direction
                    module.add_historical_direction(adapter_name, current_A, current_B)
                    
                    # Reset current task's LoRA for next task
                    with torch.no_grad():
                        module.loranew_A[adapter_name].weight.zero_()
                        module.loranew_B[adapter_name].weight.zero_()


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(self, in_features: int, out_features: int):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        # 当前任务的LoRA参数
        self.loranew_A = nn.ModuleDict({})
        self.loranew_B = nn.ModuleDict({})
        # 历史方向存储：每个方向分开保存
        self.historical_directions = nn.ModuleDict({})  # 存储历史A和B矩阵
        self.historical_scalings = nn.ParameterDict({})  # 存储每个历史方向的可训练scaling
        # 使用 ParameterDict 来保存历史方向数量，这样可以被torch保存和加载
        self.num_historical_directions = nn.ParameterDict({})  # 记录每个adapter的历史方向数量
        # Embedding相关参数
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.loranew_embedding_A = nn.ParameterDict({})
        self.loranew_embedding_B = nn.ParameterDict({})
        # self.historical_embedding_directions = nn.ModuleDict({})
        # self.historical_embedding_scalings = nn.ParameterDict({})
        # 兼容性参数（为了向后兼容现有代码）
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, r_sum):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        
        if r > 0:
            # 当前任务的新LoRA参数 (A_t, B_t in the pseudocode)
            self.loranew_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.loranew_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            
            # 初始化历史方向存储
            self.num_historical_directions[adapter_name] = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
            self.historical_directions.update(nn.ModuleDict({adapter_name: nn.ModuleDict({})}))
            self.historical_scalings.update(nn.ParameterDict({adapter_name: nn.ParameterDict({})}))
            
            # 为了向后兼容，保留原始的lora_A和lora_B（但现在它们只是占位符）
            if r_sum > 0:
                self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r_sum, bias=False)}))
                self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r_sum, self.out_features, bias=False)}))
            else:
                # 第一个任务时，没有历史LoRA方向，创建空的占位符
                self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, 0, bias=False)}))
                self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(0, self.out_features, bias=False)}))
            
            # self.scaling[adapter_name] = lora_alpha / r if r > 0 else 1.0
            self.scaling[adapter_name] = 0.8
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def add_historical_direction(self, adapter_name, direction_A, direction_B, initial_scaling=0.8):
        """
        添加一个新的历史方向到存储中
        Args:
            adapter_name: adapter名称
            direction_A: A矩阵 [r, in_features]
            direction_B: B矩阵 [out_features, r]
            initial_scaling: 初始scaling值
        """
        if adapter_name not in self.historical_directions:
            self.historical_directions.update(nn.ModuleDict({adapter_name: nn.ModuleDict({})}))
            self.historical_scalings.update(nn.ParameterDict({adapter_name: nn.ParameterDict({})}))
            self.num_historical_directions[adapter_name] = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
        
        direction_idx = self.num_historical_directions[adapter_name].item()
        direction_name = f"dir_{direction_idx}"
        
        # 创建新的方向模块
        direction_module = nn.ModuleDict({
            'A': nn.Linear(direction_A.shape[1], direction_A.shape[0], bias=False),
            'B': nn.Linear(direction_B.shape[1], direction_B.shape[0], bias=False)
        })
        
        # 复制权重
        direction_module['A'].weight.data.copy_(direction_A)
        direction_module['B'].weight.data.copy_(direction_B)
        
        # 冻结历史方向的权重
        direction_module['A'].weight.requires_grad = False
        direction_module['B'].weight.requires_grad = False
        
        # 添加到历史方向中
        self.historical_directions[adapter_name].update(nn.ModuleDict({direction_name: direction_module}))
        
        # 添加可训练的scaling参数
        scaling_param = nn.Parameter(torch.tensor(initial_scaling, dtype=direction_A.dtype, device=direction_A.device))
        self.historical_scalings[adapter_name].update(nn.ParameterDict({direction_name: scaling_param}))
        
        # 更新历史方向数量
        self.num_historical_directions[adapter_name].data = torch.tensor(direction_idx + 1, dtype=torch.long, device=direction_A.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r if r > 0 else 1.0
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_A[adapter_name].weight)
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
        if adapter_name in self.loranew_A.keys():
            nn.init.kaiming_uniform_(self.loranew_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.loranew_B[adapter_name].weight)


class Linear(nn.Linear, LoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        r_sum: int = 0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, r_sum)
        self.active_adapter = adapter_name

    # def merge(self):
    #     if self.active_adapter not in self.lora_A.keys():
    #         return
    #     if self.merged:
    #         warnings.warn("Already merged. Nothing to do.")
    #         return
    #     if self.r[self.active_adapter] > 0:
    #         self.weight.data += (
    #             transpose(
    #                 self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
    #                 self.fan_in_fan_out,
    #             )
    #             * self.scaling[self.active_adapter]
    #         )
    #         self.merged = True

    # def unmerge(self):
    #     if self.active_adapter not in self.lora_A.keys():
    #         return
    #     if not self.merged:
    #         warnings.warn("Already unmerged. Nothing to do.")
    #         return
    #     if self.r[self.active_adapter] > 0:
    #         self.weight.data -= (
    #             transpose(
    #                 self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
    #                 self.fan_in_fan_out,
    #             )
    #             * self.scaling[self.active_adapter]
    #         )
    #         self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        # base output first: W_0 * x
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        # if no adapter branches at all, just return
        has_prev = self.active_adapter in self.lora_A.keys()
        has_new = self.active_adapter in self.loranew_A.keys()
        if not has_prev and not has_new:
            return result

        if self.disable_adapters:
            if self.merged:
                # ensure we are not in a merged state when adapters are disabled
                self.unmerge()
            return result

        if not self.merged:
            # compute in LoRA dtype
            ref_dtype = (
                self.loranew_A[self.active_adapter].weight.dtype
            )
            x_lora = x.to(ref_dtype)
            x_lora = self.lora_dropout[self.active_adapter](x_lora)

            # Historical LoRA directions: sum over previous tasks with individual trainable scalings
            # This implements the sum: α_1 A_1 B_1 + α_2 A_2 B_2 + ... + α_{t-1} A_{t-1} B_{t-1}
            if self.active_adapter in self.num_historical_directions and self.num_historical_directions[self.active_adapter].item() > 0:
                for i in range(self.num_historical_directions[self.active_adapter].item()):
                    direction_key = f"dir_{i}"
                    
                    if (self.active_adapter in self.historical_directions and 
                        direction_key in self.historical_directions[self.active_adapter] and
                        self.active_adapter in self.historical_scalings and
                        direction_key in self.historical_scalings[self.active_adapter]):
                        
                        # Get the historical direction components
                        

                         # 确保历史适配器层与输入数据类型一致
                        target_device = x_lora.device
                        target_dtype = x_lora.dtype
                        
                        historical_A = self.historical_directions[self.active_adapter][direction_key]['A']
                        historical_B = self.historical_directions[self.active_adapter][direction_key]['B']
                        
                        historical_A = historical_A.to(device=target_device, dtype=target_dtype)
                        historical_B = historical_B.to(device=target_device, dtype=target_dtype)
                        # Apply the direction with its individual trainable scaling
                        # print('-'*40)
                        # print(f"historical_A:{historical_A.weight.device}  {historical_A.weight.dtype}")
                        # print(f"historical_B:{historical_B.weight.device}  {historical_B.weight.dtype}")
                        # print(f"x_lora:{x_lora.device}  {x_lora.dtype}")
                        # print('='*40)

                        historical_output = historical_B(historical_A(x_lora))
                        result = result + historical_output * self.historical_scalings[self.active_adapter][direction_key]

            # Current task LoRA: α_t A_t B_t  
            # This implements the current task term from equation (4)
            if has_new and self.r.get(self.active_adapter, 0) > 0:
                current_output = self.loranew_B[self.active_adapter](self.loranew_A[self.active_adapter](x_lora))
                result = result + current_output * self.scaling[self.active_adapter]

        return result.to(previous_dtype)


class Embedding(nn.Embedding, LoraLayer):
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)
        self.weight.requires_grad = False
        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter]
                        @ self.lora_embedding_A[self.active_adapter],
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


# if is_bnb_available():

#     class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
#         def __init__(
#             self,
#             adapter_name,
#             in_features,
#             out_features,
#             r: int = 0,
#             lora_alpha: int = 1,
#             lora_dropout: float = 0.0,
#             **kwargs,
#         ):
#             bnb.nn.Linear8bitLt.__init__(
#                 self,
#                 in_features,
#                 out_features,
#                 bias=kwargs.get("bias", True),
#                 has_fp16_weights=kwargs.get("has_fp16_weights", True),
#                 memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
#                 threshold=kwargs.get("threshold", 0.0),
#                 index=kwargs.get("index", None),
#             )
#             LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
#             self.weight.requires_grad = False
#             init_lora_weights = kwargs.pop("init_lora_weights", True)
#             self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, r_sum=0)
#             self.active_adapter = adapter_name

#         def forward(self, x: torch.Tensor):
#             result = super().forward(x)
#             has_prev = self.active_adapter in self.lora_A.keys()
#             has_new = self.active_adapter in self.loranew_A.keys()
#             if self.disable_adapters or (not has_prev and not has_new):
#                 return result
#             elif self.r.get(self.active_adapter, 0) > 0 or has_prev:
#                 if not torch.is_autocast_enabled():
#                     expected_dtype = result.dtype
#                     if x.dtype != torch.float32:
#                         x = x.float()
#                     output = 0
#                     if has_prev and self.lora_A[self.active_adapter].weight.shape[0] > 0:
#                         output = (
#                             self.lora_B[self.active_adapter](
#                                 self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
#                             ).to(expected_dtype)
#                             * self.scaling[self.active_adapter]
#                         )
#                     else:
#                         output = torch.zeros_like(result)
#                 else:
#                     output = 0
#                     if has_prev and self.lora_A[self.active_adapter].weight.shape[0] > 0:
#                         output = (
#                             self.lora_B[self.active_adapter](
#                                 self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
#                             )
#                             * self.scaling[self.active_adapter]
#                         )
#                     else:
#                         output = torch.zeros_like(result)
#                 # add trainable new branch if exists
#                 if has_new and self.r.get(self.active_adapter, 0) > 0:
#                     output = output + (
#                         self.loranew_B[self.active_adapter](
#                             self.loranew_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
#                         )
#                         * self.scaling[self.active_adapter]
#                     )
#                 result += output
#             return result
