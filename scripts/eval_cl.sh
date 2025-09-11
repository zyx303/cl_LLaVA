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

python -m llava.eval.eval_cl \
  --model-path checkpoints/behavior_il_seed1_task0   \
  --model-base model/llava/llava-v1.5-7b \
  --data ./data \
  --splits valid_unseen \
  --incremental_setup behavior_il \
  --stream_seed 1 \
  --incremental_task 1 \
  --output_file eval_results/behavior_il_seed1_task0_valid_unseen.json \
  --temperature 0.2