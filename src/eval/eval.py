import json
import pprint
import random
import time
import torch
import torch.multiprocessing as mp
import numpy as np
# from models.nn.resnet import Resnet
from data.preprocess import Dataset
from importlib import import_module
from llava.utils import disable_torch_init
from gen import constants
from llava.mm_utils import process_images
from PIL import Image
import os 
from llava.model import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


class Eval(object):

    # tokens
    STOP_TOKEN = "<<stop>>"
    SEQ_TOKEN = "<<seg>>"
    TERMINAL_TOKENS = [STOP_TOKEN, SEQ_TOKEN]

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        self.splits[self.args.eval_split] = json.load(open(self.args.split_json, 'r'))

        # load high-level model with lora 
        print("Loading high-level model: ", self.args.model_path_high)
        disable_torch_init()
        # this may be mm projector only
        print('Loading LLaVA from base model...')
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_base_high, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(self.args.model_path_high)
        # load high model
        self.high_model = LlavaLlamaForCausalLM.from_pretrained(self.args.model_base_high, low_cpu_mem_usage=True, config=cfg_pretrained)
        mm_projector_weights = torch.load(os.path.join(self.args.model_path_high, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        self.high_model.load_state_dict(mm_projector_weights, strict=False)

        
        # load low-level model with lora
        print("Loading low-level model: ", self.args.model_path_low)
        self.low_model = LlavaLlamaForCausalLM.from_pretrained(self.args.model_base_low, low_cpu_mem_usage=True, config=cfg_pretrained)
        mm_projector_weights = torch.load(os.path.join(self.args.model_path_low, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        self.low_model.load_state_dict(mm_projector_weights, strict=False)

        # updated args
        # self.model.args.dout = self.args.model_path.replace(self.args.model_path.split('/')[-1], '')
        # self.model.args.data = self.args.data if self.args.data else self.model.args.data

        # self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        #load vision encoder
        self.vision_tower = self.llava_model_high.get_vision_tower()
        self.image_processor = self.vision_tower.image_processor


        # gpu
        if self.args.gpu:
            self.llava_model_high = self.llava_model_high.to(torch.device('cuda'))
            self.llava_model_low = self.llava_model_low.to(torch.device('cuda'))

        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))


    def extract_vision_features(self, image, model_type='high'):
        """使用 LLaVA vision tower 提取视觉特征"""
            
        if self.vision_tower is None:
            raise ValueError(f"Vision tower for {model_type} model not initialized.")
        
        # 将 PIL Image 转换为适合 LLaVA 的格式
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = [Image.fromarray(np.uint8(image))]
        
        # 使用 LLaVA 的图像处理器
        images_tensor = process_images(
            images,
            self.image_processor,
            self.high_model.config
        )
        
        # 移动到正确的设备和数据类型
        images_tensor = images_tensor.to(self.vision_tower.device, dtype=torch.float16)

        # 提取视觉特征
        with torch.no_grad():
            vision_features = self.vision_tower(images_tensor)
        
        return vision_features

    # generate high instruction or low action
    def generate_response(self, image, goal_text, vis_feature=None, high_instr=None, model_type='high'):
        """使用指定模型生成回答"""
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        
        if model_type == 'high':
            model = self.high_model
            prompt_raw = f"<image>\nGiven the goal: {goal_text}\nWhat is the next high-level action to take?"
        elif model_type == 'low':
            model = self.low_model
            prompt_raw = f'<image>\nAccording to the image, Given the goal: "{goal_text}"\nThe high-level instruction is "{high_instr}", what is the next low-level action to take?\nAvailable low-level actions: ["MoveAhead","RotateLeft","RotateRight","LookUp","LookDown","PickupObject","HeatObject","PutObject","OpenObject","CloseObject","ToggleObjectOn","ToggleObjectOff"]. Please exactly output one of them as your answer. If the action is in ["PickupObject","PutObject","OpenObject","CloseObject","ToggleObjectOn"] please also give the label right after the action.'
        else:
            raise ValueError("model_type must be 'high' or 'low'")
        
        # 处理图像token
        # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # qs = prompt
        # if IMAGE_PLACEHOLDER in qs:
        #     if model.config.mm_use_im_start_end:
        #         qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
        #     else:
        #         qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
        # else:
        #     if model.config.mm_use_im_start_end:
        #         qs = image_token_se + "\n" + qs
        #     else:

        

        conv_mode = "v1"
        qs = prompt_raw
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # process image
        
        images_tensor = process_images([image], self.image_processor, model.config).to(model.device, dtype=torch.float16)
        image_sizes = [image.size]

        # tokenize and generate
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if self.args.temperature > 0 else False,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            num_beams=self.args.num_beams,
            max_new_tokens=self.args.max_new_tokens,
            use_cache=True,
        )
        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if model_type == 'high':
            # 返回高层指令
            if os.getenv('DEBUG', '0') == '1':
                print(f"High-level model prompt:\n{prompt}\n")
                print(f"High-level model response:\n{pred}\n")
        else:
            words=pred.split(' ')
            if len(words)>1:
                action=words[0]
                obj=' '.join(words[1:])
                pred={'action':action,'object':obj}
            else:
                pred={'action':words[0],'object':None}  

        return pred,images_tensor
    
    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        files = self.splits[self.args.eval_split]

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:16]

        if self.args.shuffle:
            random.shuffle(files)
        for traj in files:
            task_queue.put(traj)
        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, task_queue, self.args, lock,
                                                       self.successes, self.failures, self.results))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # save
        self.save_results()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()
