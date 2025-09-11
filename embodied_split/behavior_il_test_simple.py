import json 
import os 
import sys 

for i in range(7):
    # /home/yongxi/work/cl-alfred/embodied_split/behavior_il/embodied_data_disjoint_rand1_cls1_task0.json
    file = json.load(open(f'/home/yongxi/work/cl-alfred/embodied_split/behavior_il/embodied_data_disjoint_rand1_cls1_task{i}.json'))
    new_file_path = f'/home/yongxi/work/cl-alfred/embodied_split/behavior_il_test/embodied_data_disjoint_rand1_cls1_task{i}.json'
    new_file = []
    for j in range(20):
        new_file.append(file[j])
    # 创建文件
    # os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    json.dump(new_file, open(new_file_path, 'w'))