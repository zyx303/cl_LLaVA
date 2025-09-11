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
#############################################################################
splits = json.load(open('data/splits/oct21.json', 'r'))
for incremental_setup in ['behavior_il', 'environment_il']:
    for stream_seed in range(1,4):
        if incremental_setup in ['behavior_il','behavior_il_test']:
            # filtering based on task types
            print(stream_seed)
            for i in range(7):
                task_types = BEHAVIOR_TYPES[stream_seed]
                seen_tasks = task_types[:(i + 1)]
                print('seen_tasks:', len(seen_tasks))
                splits['valid_seen'] = [t for t in splits['valid_seen'] if any(st in t['task'] for st in seen_tasks)]
                splits['valid_unseen'] = [t for t in splits['valid_unseen'] if any(st in t['task'] for st in seen_tasks)]
                os.makedirs(f'data/valid_seen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits['valid_seen'], open(f'data/valid_seen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
                os.makedirs(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits['valid_unseen'], open(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
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
                splits['valid_seen'] = [t for t in splits['valid_seen'] if task2scene(t) in scene_types_now]
                splits['valid_unseen'] = [t for t in splits['valid_unseen'] if task2scene(t) in scene_types_now]

                os.makedirs(f'data/valid_seen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits['valid_seen'], open(f'data/valid_seen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
                os.makedirs(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}', exist_ok=True)
                json.dump(splits['valid_unseen'], open(f'data/valid_unseen/{incremental_setup}_stream{stream_seed}/step{i+1}.json', 'w'))
        else:
            print('Invalid incremental setup for evaluation:', incremental_setup)
            exit(0)

        # save splits

#############################################################################