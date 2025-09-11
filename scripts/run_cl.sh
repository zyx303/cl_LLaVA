export TRANSFORMERS_CACHE=/data/yongxi/.cache/huggingface
export CUDA_VISIBLE_DEVICES=7
# python -m llava.eval.run_llava \
#   --model-path checkpoints/behavior_il_seed1_task0   \
#   --model-base model/llava/llava-v1.5-7b \
#   --image-file data/full/train/look_at_obj_in_light-Box-None-FloorLamp-212/trial_T20190908_193427_340509/raw_images/000000000.jpg \
#   --query "Given the goal: Examine a box by the light of a floor lamp\nWhat is the next high-level action to take?"

# python -m llava.eval.run_llava \
#   --model-path checkpoints/behavior_il_seed1_task0   \
#   --model-base model/llava/llava-v1.5-7b \
#   --image-file data/full/train/look_at_obj_in_light-Box-None-FloorLamp-212/trial_T20190908_193427_340509/raw_images/000000224.jpg \
#   --query "Given the goal: Examine a box by the light of a floor lamp\nWhat is the next high-level action to take?"

img="data/full/train/look_at_obj_in_light-Pillow-None-DeskLamp-324/trial_T20190908_155703_086493/raw_images/000000030.jpg"
query="Given the goal: Examine a pillow.\nWhat is the next high-level action to take?"
python -m llava.eval.run_llava \
  --model-path checkpoints/behavior_il_seed1_task0   \
  --model-base model/llava/llava-v1.5-7b \
  --image-file "$img" \
  --query "$query"