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

from .config import PeftType, PromptLearningConfig
import torch
from typing import Any, Dict
import numpy as np
from peft.tuners.inflora import LoraLayer as InfLoraLayer


def set_l2p_task_id(model, task_id, adapter_name="default"):
    """
    设置L2P模型的当前任务ID，用于任务迁移
    
    Args:
        model: PEFT模型
        task_id: 当前任务ID
        adapter_name: adapter名称
    """
    config = model.peft_config.get(adapter_name)
    if config and config.peft_type == PeftType.L2P:
        config.current_task_id = task_id
        model._current_task_id = task_id
        print(f"L2P: 设置任务ID为 {task_id}")
    else:
        print(f"Warning: 模型不是L2P类型或adapter {adapter_name} 不存在")


def get_l2p_task_id(model, adapter_name="default"):
    """
    获取L2P模型的当前任务ID
    
    Args:
        model: PEFT模型
        adapter_name: adapter名称
    
    Returns:
        int: 当前任务ID，如果未设置则返回0
    """
    config = model.peft_config.get(adapter_name)
    if config and config.peft_type == PeftType.L2P:
        return getattr(config, 'current_task_id', getattr(model, '_current_task_id', 0))
    return 0


# ---------------- HiDe-Prompt helpers ----------------
def set_hide_prompt_task_id(model, task_id, adapter_name="default"):
    """
    设置HiDe-Prompt模型的当前任务ID，便于推理阶段自动构造 mask/索引。
    """
    config = model.peft_config.get(adapter_name)
    if config and config.peft_type == PeftType.HIDE_PROMPT:
        config.current_task_id = task_id
        # 同步到模型实例，供运行期兜底
        model._current_task_id = task_id
        print(f"HiDe-Prompt: 设置任务ID为 {task_id}")
    else:
        print(f"Warning: 模型不是HiDe-Prompt类型或adapter {adapter_name} 不存在")


def get_hide_prompt_task_id(model, adapter_name="default"):
    """
    获取HiDe-Prompt模型的当前任务ID，未设置则返回0。
    """
    config = model.peft_config.get(adapter_name)
    if config and config.peft_type == PeftType.HIDE_PROMPT:
        return getattr(config, 'current_task_id', getattr(model, '_current_task_id', 0))
    return 0


@torch.no_grad()
def update_hide_prompt_after_task(model, task_id, prompt_momentum: float, adapter_name: str = "default"):
    if prompt_momentum <= 0 or task_id <= 0:
        return False

    try:
        eprompt = model.prompt_encoder[adapter_name]
    except Exception:
        return False

    if not hasattr(eprompt, "prompt"):
        return False

    prompt = eprompt.prompt
    try:
        # prefix : (L, 2, P, T, H, D)
        if getattr(eprompt, "use_prefix_tune_for_e_prompt", False) and prompt.dim() == 6:
            L, dual, P, T, H, D = prompt.shape
            if task_id >= P:
                print(f"Warning: task_id={task_id} 超过 prompt 池大小 P={P}，跳过动量更新")
                return False
            # 历史均值：在 pool 维度(=2)上对 [0:task_id] 求均值，保持维度便于广播
            prev_mean = prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True)  # (L,2,1,T,H,D)
            cur = prompt[:, :, task_id].detach().clone()  # (L,2,T,H,D)
            prompt[:, :, task_id].copy_((1.0 - prompt_momentum) * cur + prompt_momentum * prev_mean.squeeze(2))
            return True

        # 非 prefix : (L, P, T, C)
        if prompt.dim() == 4:
            L, P, T, C = prompt.shape
            if task_id >= P:
                print(f"Warning: task_id={task_id} 超过 prompt 池大小 P={P}，跳过动量更新")
                return False
            prev_mean = prompt[:, 0:task_id].detach().clone().mean(dim=1, keepdim=True)  # (L,1,T,C)
            cur = prompt[:, task_id].detach().clone()  # (L,T,C)
            prompt[:, task_id].copy_((1.0 - prompt_momentum) * cur + prompt_momentum * prev_mean.squeeze(1))
            return True

        print("Warning: 未识别的 HiDe-Prompt 参数形状，跳过动量更新")
        return False
    except Exception as e:
        print(f"Warning: HiDe-Prompt 动量更新失败: {e}")
        return False


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
        
        # For SDLoRA: consolidate current task's LoRA directions before saving
        if config.peft_type == PeftType.SDLORA:
            # Consolidate LoRA directions: W ← W ∪ {A_t B_t}
            if hasattr(model, 'consolidate_lora_directions'):
                model.consolidate_lora_directions()
            # Update state_dict after consolidation
            state_dict = model.state_dict()
        
        # Original concatenation logic for backward compatibility
        if not isinstance(config, PromptLearningConfig) and config.save_loranew == False and config.peft_type != PeftType.SDLORA:
            flag = 1 # this is a switch represents whether 'r_sum' is written to the config file
            for k in state_dict:
                if "lora_A" in k:
                    for k_ in state_dict:
                        if "loranew_A" in k_ and k.split("lora_A")[0] == k_.split("loranew_A")[0]:
                            state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=0) # [r_sum + r, r]
                            if flag == 1:
                                config.r_sum = state_dict[k].shape[0] 
                                flag = 0
                            break # target modules have been matched
                elif "lora_B" in k:
                    for k_ in state_dict:
                        if "loranew_B" in k_ and k.split("lora_B")[0] == k_.split("loranew_B")[0]:
                            state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=1) # [r, r_sum + r]
                            break # target modules have been matched

                
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.SDLORA,PeftType.INFLORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias

        # modified
        if bias == "none":
            if config.save_loranew and config.peft_type != PeftType.SDLORA: 
                to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "loranew_" in k} # modified
            else:
                base_keys = {k: state_dict[k] for k in state_dict if "lora_" in k}
                # For SDLoRA with separate storage, also include historical directions and scalings
                if config.peft_type == PeftType.SDLORA:
                    historical_keys = {k: state_dict[k] for k in state_dict if "historical_directions" in k or "historical_scalings" in k}
                    # Also save num_historical_directions for each layer
                    num_directions_keys = {k: state_dict[k] for k in state_dict if "num_historical_directions" in k}
                    to_return = {**base_keys, **historical_keys, **num_directions_keys}
                else:
                    to_return = base_keys
                # Update r_sum for SDLoRA based on consolidated lora_A size
                # if config.peft_type == PeftType.SDLORA:
                #     for k, v in base_keys.items():
                #         if "lora_A" in k and adapter_name in k:
                #             config.r_sum = v.shape[0]  # Update r_sum based on current consolidated size
                #             break

        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

        # modified
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k) or ("loranew_" in k) or ("historical_directions" in k) or ("historical_scalings" in k) or ("num_historical_directions" in k))}
        
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

        # ---- InfLoRA: persist continual-learning state (best-effort) ----
        if config.peft_type == PeftType.INFLORA:
            inf_state: Dict[str, Any] = {}
            # feature_list / feature_mat
            feat_list = []
            for item in getattr(model, "feature_list"):
                #store as tensor, load as nparray
                t = item
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t)
                feat_list.append(t.detach().cpu())
            inf_state["feature_list"] = feat_list
                
            feat_mat = []
            for item in getattr(model, "feature_mat"):
                t = item
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t)
                feat_mat.append(t.detach().cpu())
            inf_state["feature_mat"] = feat_mat

            inf_state["project_type"] = model.project_type

            # Per-layer matrices by module dotted path
            per_layer: Dict[str, Dict[str, Any]] = {}
            for name, module in model.named_modules():
                if isinstance(module, InfLoraLayer):
                    layer_state: Dict[str, Any] = {
                        "matrix": (module.matrix.detach().cpu() if torch.is_tensor(module.matrix) else torch.as_tensor(module.matrix)),
                        "n_matrix": int(module.n_matrix),
                        "cur_matrix": (module.cur_matrix.detach().cpu() if torch.is_tensor(module.cur_matrix) else torch.as_tensor(module.cur_matrix)),
                        "n_cur_matrix": int(module.n_cur_matrix),
                    }
                    per_layer[name] = layer_state
            inf_state["per_layer"] = per_layer
            to_return["inflora_state"] = inf_state

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.peft_type == PeftType.L2P:
            # 保存 L2P 的 prompt 池与 key，同时保存任务相关信息
            l2p = model.prompt_encoder[adapter_name]
            if not hasattr(l2p, "prompt") or not hasattr(l2p, "prompt_key"):
                raise ValueError("L2P: prompt/prompt_key 未找到，无法持久化")
            to_return["prompt_pool"] = l2p.prompt.detach().cpu()
            to_return["prompt_key"] = l2p.prompt_key.detach().cpu()
            
            # 保存L2P任务相关的元信息，用于任务迁移
            l2p_meta = {}
            if hasattr(config, 'top_k'):
                l2p_meta["top_k"] = config.top_k
            if hasattr(config, 'pool_size'):
                l2p_meta["pool_size"] = config.pool_size
            # 尝试从环境变量或其他地方获取当前任务ID
            current_task_id = getattr(config, 'current_task_id', None)
            if current_task_id is None:
                # 尝试从模型属性获取
                current_task_id = getattr(model, '_current_task_id', 0)
            l2p_meta["task_id"] = current_task_id
            l2p_meta["shared_prompt_pool"] = getattr(config, 'shared_prompt_pool', True)
            l2p_meta["shared_prompt_key"] = getattr(config, 'shared_prompt_key', True)
            to_return["l2p_meta"] = l2p_meta
        elif config.peft_type == PeftType.HIDE_PROMPT:
            # 保存 HiDe-Prompt 的池（prompt）与可选的 prompt_key，以及任务相关 meta
            eprompt = model.prompt_encoder[adapter_name]
            if not hasattr(eprompt, "prompt"):
                raise ValueError("HiDe-Prompt: prompt 未找到，无法持久化")
            to_return["hide_prompt_pool"] = eprompt.prompt.detach().cpu()
            if getattr(eprompt, "prompt_key", None) is not None:
                to_return["hide_prompt_key"] = eprompt.prompt_key.detach().cpu()

            hide_meta = {
                "pool_size": getattr(config, "pool_size", getattr(eprompt, "pool_size", None)),
                "top_k": getattr(config, "top_k", getattr(eprompt, "top_k", None)),
            }
            # 透传任务ID，便于推理时自动构造 prompt_mask
            current_task_id = getattr(config, "current_task_id", getattr(model, "_current_task_id", None))
            if current_task_id is not None:
                hide_meta["task_id"] = int(current_task_id)
            to_return["hide_prompt_meta"] = hide_meta
        else:
            # 其他 PromptLearningConfig 维持原逻辑
            if config.inference_mode:
                prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
            else:
                prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
            to_return["prompt_embeddings"] = prompt_embeddings
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    # torch.save(model.state_dict(), "full_model.pth") # for debug
    return to_return

# 加载lora
def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    # For SDLoRA: Pre-create historical direction structures before loading
    if config.peft_type == PeftType.SDLORA:
        # First, collect all historical direction data from state_dict
        historical_data = {}
        for k, v in state_dict.items():
            if "historical_directions" in k:
                # Parse the key to extract layer name and direction info
                # Example key: "base_model.model.encoder.block.0.layer.0.SelfAttention.q.historical_directions.dir_0.A.weight"
                parts = k.split("historical_directions.")
                if len(parts) >= 2:
                    layer_path = parts[0]  # e.g., "base_model.model.encoder.block.0.layer.0.SelfAttention.q."
                    remaining = parts[1]   # e.g., "dir_0.A.weight"
                    
                    remaining_parts = remaining.split(".")
                    if len(remaining_parts) >= 2:
                        direction_key = remaining_parts[0]  # "dir_0"
                        component = remaining_parts[1]  # "A" or "B"
                        
                        if layer_path not in historical_data:
                            historical_data[layer_path] = {}
                        if adapter_name not in historical_data[layer_path]:  # 使用当前的adapter_name
                            historical_data[layer_path][adapter_name] = {}
                        if direction_key not in historical_data[layer_path][adapter_name]:
                            historical_data[layer_path][adapter_name][direction_key] = {}
                        
                        historical_data[layer_path][adapter_name][direction_key][component] = v
        
        # Now create the historical direction structures in the model
        for layer_path, adapter_data in historical_data.items():
            for adapter_key, directions in adapter_data.items():
                # Find the corresponding layer in the model
                layer_parts = layer_path.strip('.').split('.')
                current_module = model
                for part in layer_parts:
                    if hasattr(current_module, part):
                        current_module = getattr(current_module, part)
                    else:
                        break
                
                # Check if this is a LoRA layer with historical directions capability
                if hasattr(current_module, 'historical_directions') and hasattr(current_module, 'historical_scalings'):
                    # Ensure the adapter key exists in historical_directions
                    if adapter_key not in current_module.historical_directions:
                        current_module.historical_directions[adapter_key] = torch.nn.ModuleDict()
                    if adapter_key not in current_module.historical_scalings:
                        current_module.historical_scalings[adapter_key] = torch.nn.ParameterDict()
                    
                    # Create each direction
                    for direction_key, components in directions.items():
                        if 'A' in components and 'B' in components:
                            # Get weight shapes to create placeholder modules
                            A_weight = components['A']
                            B_weight = components['B']
                            
                            # Create placeholder direction module structure
                            direction_module = torch.nn.ModuleDict({
                                'A': torch.nn.Linear(A_weight.shape[1], A_weight.shape[0], bias=False),
                                'B': torch.nn.Linear(B_weight.shape[1], B_weight.shape[0], bias=False)
                            })
                            
                            # Add to historical_directions
                            current_module.historical_directions[adapter_key][direction_key] = direction_module
                            
                            # Create placeholder scaling parameter
                            scaling_param = torch.nn.Parameter(torch.tensor(1.0, dtype=A_weight.dtype))
                            current_module.historical_scalings[adapter_key][direction_key] = scaling_param

                            new_num_directions = max(current_module.num_historical_directions[adapter_key], int(direction_key.split('_')[1]) + 1)
                            current_module.num_historical_directions[adapter_key] = torch.nn.Parameter(
                                torch.tensor(new_num_directions, dtype=torch.long), 
                                requires_grad=False
                            )

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.SDLORA, PeftType.INFLORA):
        # ---- InfLoRA: restore continual-learning state prior to weight loading ----
        if config.peft_type == PeftType.INFLORA and "inflora_state" in state_dict:
            inf_state = state_dict.get("inflora_state", {})
            if isinstance(inf_state, dict):
                # feature_list / feature_mat
                if "feature_list" in inf_state:
                    #store as tensor, load as nparray
                    fl = [t if not torch.is_tensor(t) else np.asarray(t.cpu()) for t in inf_state["feature_list"]]
                    model.base_model.feature_list = fl
                if "feature_mat" in inf_state:
                    fm = [t if torch.is_tensor(t) else torch.as_tensor(t) for t in inf_state["feature_mat"]]
                    model.base_model.feature_mat = fm
                if "project_type" in inf_state:
                    model.base_model.project_type = list(inf_state["project_type"])
                # Per-layer
                if "per_layer" in inf_state and isinstance(inf_state["per_layer"], dict):
                    for layer_path, lstate in inf_state["per_layer"].items():
                        obj = model
                        for part in layer_path.split("."):
                            if not part:
                                continue
                            obj = getattr(obj, part)
                        if hasattr(obj, "matrix") and "matrix" in lstate:
                            val = lstate["matrix"]
                            setattr(obj, "matrix", val if torch.is_tensor(val) else torch.as_tensor(val))
                        if hasattr(obj, "n_matrix") and "n_matrix" in lstate:
                            setattr(obj, "n_matrix", int(lstate["n_matrix"]))
                        if hasattr(obj, "cur_matrix") and "cur_matrix" in lstate:
                            val = lstate["cur_matrix"]
                            setattr(obj, "cur_matrix", val if torch.is_tensor(val) else torch.as_tensor(val))
                        if hasattr(obj, "n_cur_matrix") and "n_cur_matrix" in lstate:
                            setattr(obj, "n_cur_matrix", int(lstate["n_cur_matrix"]))
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            
            # modified
            elif "loranew_" in k: 
                suffix = k.split("loranew_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            elif k == "inflora_state":
                # Already restored above; skip passing to load_state_dict
                continue
            # For SDLoRA: handle historical_directions, historical_scalings, and num_historical_directions
            elif "historical_directions" in k or "historical_scalings" in k or "num_historical_directions" in k:
                k = k.replace("historical_directions", f"historical_directions.{adapter_name}")
                k = k.replace("historical_scalings", f"historical_scalings.{adapter_name}")
                # k = k.replace("num_historical_directions", f"num_historical_directions.{adapter_name}")
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
        elif config.peft_type == PeftType.SDLORA:
            # For SDLoRA, ensure r_sum is updated based on loaded lora_A size
            for k, v in peft_model_state_dict.items():
                if "lora_A" in k and adapter_name in k:
                    config.r_sum = v.shape[0]
                    break
    elif isinstance(config, PromptLearningConfig) or config.peft_type == PeftType.ADAPTION_PROMPT:
        if config.peft_type == PeftType.L2P:
            # 从 state_dict 恢复 L2P 的 prompt 池与 key，并实现任务迁移
            l2p = model.prompt_encoder[adapter_name]
            
            # 加载基本的prompt和key
            if "prompt_pool" in state_dict:
                with torch.no_grad():
                    l2p.prompt.data.copy_(state_dict["prompt_pool"].to(l2p.prompt.dtype))
            if "prompt_key" in state_dict and hasattr(l2p, "prompt_key"):
                with torch.no_grad():
                    l2p.prompt_key.data.copy_(state_dict["prompt_key"].to(l2p.prompt_key.dtype))
            
            if "l2p_meta" in state_dict:
                l2p_meta = state_dict["l2p_meta"]
                prev_task_id = l2p_meta.get("task_id", 0)
                top_k = l2p_meta.get("top_k", 5)
                shared_prompt_pool = l2p_meta.get("shared_prompt_pool", True)
                shared_prompt_key = l2p_meta.get("shared_prompt_key", True)
                
                # 获取当前任务ID（从config或环境变量）
                current_task_id = getattr(config, 'current_task_id', None)
                if current_task_id is None:
                    current_task_id = getattr(model, '_current_task_id', prev_task_id + 1)
                
                # 如果是新任务且启用了共享prompt池，执行任务迁移
                if (current_task_id > prev_task_id and shared_prompt_pool and 
                    hasattr(l2p, 'init_task_prompts')):
                    
                    print(f"L2P: 执行任务迁移 - 从任务{prev_task_id}到任务{current_task_id}")
                    print(f"L2P: top_k={top_k}, shared_pool={shared_prompt_pool}, shared_key={shared_prompt_key}")
                    
                    # 使用PILOT式的迁移逻辑
                    with torch.no_grad():
                        # Transfer previous learned prompt params to the new prompt
                        if shared_prompt_pool:
                            prev_start = prev_task_id * top_k
                            prev_end = (prev_task_id + 1) * top_k
                            
                            cur_start = current_task_id * top_k
                            cur_end = (current_task_id + 1) * top_k
                            
                            pool_size = l2p.prompt.shape[0]
                            if (prev_end <= pool_size) and (cur_end <= pool_size):
                                # Copy prompts from previous task to current task
                                prev_idx = slice(prev_start, prev_end)
                                cur_idx = slice(cur_start, cur_end)
                                
                                if l2p.prompt.grad is not None:
                                    l2p.prompt.grad.zero_()
                                l2p.prompt[cur_idx] = l2p.prompt[prev_idx].clone()
                                print(f"L2P: 复制prompt from [{prev_start}:{prev_end}] to [{cur_start}:{cur_end}]")
                        
                        # Transfer previous learned prompt param keys to the new prompt
                        if shared_prompt_key and hasattr(l2p, 'prompt_key'):
                            prev_start = prev_task_id * top_k
                            prev_end = (prev_task_id + 1) * top_k
                            
                            cur_start = current_task_id * top_k
                            cur_end = (current_task_id + 1) * top_k
                            
                            pool_size = l2p.prompt_key.shape[0]
                            if (prev_end <= pool_size) and (cur_end <= pool_size):
                                # Copy prompt keys from previous task to current task
                                prev_idx = slice(prev_start, prev_end)
                                cur_idx = slice(cur_start, cur_end)
                                
                                if l2p.prompt_key.grad is not None:
                                    l2p.prompt_key.grad.zero_()
                                l2p.prompt_key[cur_idx] = l2p.prompt_key[prev_idx].clone()
                                print(f"L2P: 复制prompt_key from [{prev_start}:{prev_end}] to [{cur_start}:{cur_end}]")
                    
                    # 更新模型和配置的任务ID
                    config.current_task_id = current_task_id
                    model._current_task_id = current_task_id
                    
                    print(f"L2P: 任务迁移完成，当前任务ID: {current_task_id}")
            
            # 已手动复制，避免再通过 load_state_dict 加载这些键
            peft_model_state_dict = {}
        elif config.peft_type == PeftType.HIDE_PROMPT:
            # 恢复 HiDe-Prompt 的 prompt 池与可选的 key，并传递任务ID
            eprompt = model.prompt_encoder[adapter_name]
            if "hide_prompt_pool" in state_dict:
                with torch.no_grad():
                    eprompt.prompt.data.copy_(state_dict["hide_prompt_pool"].to(eprompt.prompt.dtype))
            if "hide_prompt_key" in state_dict and getattr(eprompt, "prompt_key", None) is not None:
                with torch.no_grad():
                    eprompt.prompt_key.data.copy_(state_dict["hide_prompt_key"].to(eprompt.prompt_key.dtype))

            if "hide_prompt_meta" in state_dict:
                meta = state_dict["hide_prompt_meta"]
                if "task_id" in meta:
                    config.current_task_id = int(meta["task_id"])  # 记录到 config 与 model
                    model._current_task_id = int(meta["task_id"]) 
            # 已处理专有字段，避免重复加载
            peft_model_state_dict = {}
        else:
            peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    #保存peft_model_state_dict,debug
    with open("peft_model_state_dict_debug.log","w") as f:
        for k, v in peft_model_state_dict.items():
            f.write(f"{k}: {v}\n")
    model.load_state_dict(peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig) and config.peft_type not in [PeftType.L2P, PeftType.HIDE_PROMPT]:
         model.prompt_encoder[adapter_name].embedding.load_state_dict(
             {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
         )
