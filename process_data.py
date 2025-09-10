"""
使用 VLM 数据微调 LLaVA
基于 cl 文件夹下的数据处理逻辑，将数据转换为 LLaVA 格式进行微调
"""

import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import copy


def load_task_json(args, task):
    """加载预处理的 JSON 数据"""
    task=task['task']
    # print(task.keys())

    json_path = os.path.join(args.data, task['task'], args.pp_folder, f"ann_{task['repeat_idx']}.json")
    full_json_path = os.path.join('data/full/json_feat_2.1.0', 'train', task['task'], 'traj_data.json')
    
    with open(full_json_path) as f:
        full_data = json.load(f)
    with open(json_path) as f:
        data = json.load(f)
    
    data['images'] = full_data['images']
    data['full_path'] = full_json_path
    return data


def get_image_path(image_name, task_type, split, root):
    """获取图片完整路径"""
    root = root.replace('data/json_feat_2.1.0/', "")
    image_name = image_name.replace('.png', '.jpg')
    return os.path.join('data', 'full', split, root, 'raw_images', image_name)


def prepare_data(args, task_data_list):
    llava_data = []
    
    for task_data in tqdm(task_data_list, desc="Converting data"):
        # 加载traj_data
        traj_data = load_task_json(args, task_data)
        
        # 获取任务描述和high_level_instr
        goal_desc = traj_data['turk_annotations']['anns'][0]['task_desc']
        high_descs = traj_data['turk_annotations']['anns'][0]['high_descs']
        image_infos = traj_data['images']
        
        # 为每个高级指令和对应图片创建训练样本
        id = 0

        #find the high level instruction corresponding to the first image
        for image_info in image_infos:
            if image_info['high_idx'] == id:
                try:
                    high_desc = high_descs[id]
                except IndexError:
                    print(task_data)
                    print(f"IndexError: high_descs has length {len(high_descs)}, but trying to access index {id}. Skipping this entry.")
                    break

                # 构建图片路径
                image_path = get_image_path(
                    image_info['image_name'],
                    traj_data['task_type'],
                    traj_data['split'],
                    traj_data['root']
                )
                
                
                # 构建对话格式
                conversation = [
                    {
                        "from": "human",
                        "value": f"<image>\nGiven the goal: {goal_desc}\nWhat is the next high-level action to take?"
                    },
                    {
                        "from": "gpt", 
                        "value": high_desc
                    }
                ]
                
                # 添加到 LLaVA 数据集
                llava_data.append({
                    "id": f"{task_data['task']['task']}_{task_data['task']['repeat_idx']}_{id}",
                    "image": os.path.relpath(image_path),
                    "conversations": conversation
                })
                id +=1
                    
    return llava_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp_folder', type=str, default='pp', help='Preprocessed folder name')
    parser.add_argument('--data', type=str, default='data/json_feat_2.1.0', help='Dataset directory')
    parser.add_argument('--incremental_setup', type=str, default='behavior_il', choices=['behavior_il', 'behavior_il_test', 'environment_il'])
    parser.add_argument('--output', type=str, default='/data/yongxi/high_instr')
    args = parser.parse_args()

    if args.incremental_setup in ['behavior_il','behavior_il_test']:
        args.n_tasks = 7
    elif args.incremental_setup in ['environment_il']:
        args.n_tasks = 4
    
    for cur_iter in range(args.n_tasks):
        for stream_seed in range(1,3):
            # 加载当前任务数据
            cur_train_datalist = json.load(open(
                f'embodied_split/{args.incremental_setup}/embodied_data_disjoint_rand{stream_seed+1}_cls1_task{cur_iter}.json', 'r'
            ))
            # 转换为 LLaVA 格式
            llava_data = prepare_data(args, cur_train_datalist)

            print(f"Converted to {len(llava_data)} LLaVA training samples")
            os.makedirs(f"{args.output}/{args.incremental_setup}/stream_seed{stream_seed+1}", exist_ok=True)
            out_file = f"{args.output}/{args.incremental_setup}/stream_seed{stream_seed+1}/task{cur_iter}"
            with open(out_file, 'w') as f:
                json.dump(llava_data, f, indent=2)

            print(f"Saved LLaVA format data to {out_file}")


if __name__ == "__main__":
    main()
    