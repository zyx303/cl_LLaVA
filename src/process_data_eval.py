########################################################################################################################
# Continual Learning Settings
BEHAVIOR_TYPES = [
    [], # dummy
    [
        'look_at_obj_in_light',
        'pick_heat_then_place_in_recep',
        'pick_two_obj_and_place',
        'pick_cool_then_place_in_recep',
        'pick_and_place_simple',
        'pick_clean_then_place_in_recep',
        'pick_and_place_with_movable_recep',
    ],
    [
        'pick_and_place_simple',
        'pick_two_obj_and_place',
        'pick_clean_then_place_in_recep',
        'pick_heat_then_place_in_recep',
        'look_at_obj_in_light',
        'pick_and_place_with_movable_recep',
        'pick_cool_then_place_in_recep',
    ],
    [
        'pick_and_place_simple',
        'look_at_obj_in_light',
        'pick_and_place_with_movable_recep',
        'pick_clean_then_place_in_recep',
        'pick_two_obj_and_place',
        'pick_cool_then_place_in_recep',
        'pick_heat_then_place_in_recep',
    ],
    [
        'pick_and_place_with_movable_recep',
        'pick_two_obj_and_place',
        'look_at_obj_in_light',
        'pick_and_place_simple',
        'pick_heat_then_place_in_recep',
        'pick_cool_then_place_in_recep',
        'pick_clean_then_place_in_recep',
    ],
    [
        'pick_clean_then_place_in_recep',
        'pick_and_place_simple',
        'pick_and_place_with_movable_recep',
        'pick_heat_then_place_in_recep',
        'pick_cool_then_place_in_recep',
        'pick_two_obj_and_place',
        'look_at_obj_in_light',
    ]
]

ENVIRONMENT2NUM = {
    'Kitchen': 0,
    'Livingroom': 2,
    'Bedroom': 3,
    'Bathroom': 4,
}
ENVIRONMENT_TYPES = [
    [], # dummy
    [3, 4, 2, 0],
    [4, 3, 0, 2],
    [3, 2, 4, 0],
    [0, 4, 2, 3],
    [2, 4, 0, 3],
]

IMBALANCED_ENVIRONMENT_TYPES = [
    [], # dummy
    [3, 4, 2, 0],
    [4, 3, 0, 2],
    [3, 2, 4, 0],
    [3, 4, 0, 2],
    [4, 0, 3, 2],
]
import os 
import json
from tqdm import tqdm

def load_task_json(args, task):
    """加载预处理的 JSON 数据"""
    # task=task['task']
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
        repeat_idx = task_data['task']['repeat_idx']
        
        # 获取任务描述和high_level_instr
        goal_desc = traj_data['turk_annotations']['anns'][repeat_idx]['task_desc']
        high_descs = traj_data['turk_annotations']['anns'][repeat_idx]['high_descs']
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
#############################################################################
splits = json.load(open('data/splits/oct21.json', 'r'))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pp_folder', type=str, default='pp', help='Preprocessed folder name')
parser.add_argument('--data', type=str, default='data/json_feat_2.1.0', help='Dataset directory')
parser.add_argument('--incremental_setup', type=str, default='behavior_il', choices=['behavior_il', 'behavior_il_test', 'environment_il'])
parser.add_argument('--output', type=str, default='/data/yongxi/high_instr')
args = parser.parse_args()


for incremental_setup in ['behavior_il', 'environment_il']:
    for stream_seed in range(1,4):
        splits_now = {}
        if incremental_setup in ['behavior_il','behavior_il_test']:
            # filtering based on task types
            print(stream_seed)
            for i in range(7):
                task_types = BEHAVIOR_TYPES[stream_seed]
                seen_tasks = task_types[:(i + 1)]
                print('seen_tasks:', len(seen_tasks))
                splits_now['valid_seen'] = [t for t in splits['valid_seen'] if any(st in t['task'] for st in seen_tasks)]
                splits_now['valid_seen'] = prepare_data(args,splits_now['valid_seen'])

                splits_now['valid_unseen'] = [t for t in splits['valid_unseen'] if any(st in t['task'] for st in seen_tasks)]
                splits_now['valid_unseen'] = prepare_data(args,splits_now['valid_unseen'])


                os.makedirs(f'data/valid_seen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits_now['valid_seen'], open(f'data/valid_seen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
                os.makedirs(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits_now['valid_unseen'], open(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
        elif incremental_setup in ['environment_il', 'environment_il_nosampling']:
            # filtering based on task types
            if incremental_setup == 'environment_il':
                scene_types = ENVIRONMENT_TYPES[stream_seed]
            elif incremental_setup == 'environment_il_nosampling':
                scene_types = IMBALANCED_ENVIRONMENT_TYPES[stream_seed]
            for i in range(4):
                print('aa:',len(scene_types))
                scene_types_now = scene_types[:(i + 1)]

                def task2scene(task):
                    return int(task['task'].split('/')[0].split('-')[-1]) // 100

                print('seen_scenes:', len(scene_types_now),i+1)
                # reload eval set for environment_il
                splits = {
                    'valid_seen': json.load(open(f'embodied_split/{incremental_setup}/valid_seen.json', 'r')),
                    'valid_unseen': json.load(open(f'embodied_split/{incremental_setup}/valid_unseen.json', 'r')),
                }
                splits_now['valid_seen'] = [t for t in splits['valid_seen'] if task2scene(t) in scene_types_now]
                splits_now['valid_seen'] = prepare_data(args,splits_now['valid_seen'])
                
                splits_now['valid_unseen'] = [t for t in splits['valid_unseen'] if task2scene(t) in scene_types_now]
                splits_now['valid_unseen'] = prepare_data(args,splits_now['valid_unseen'])
                os.makedirs(f'data/valid_seen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits_now['valid_seen'], open(f'data/valid_seen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
                os.makedirs(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits_now['valid_unseen'], open(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
        else:
            print('Invalid incremental setup for evaluation:', incremental_setup)
            exit(0)

        # save splits

#############################################################################