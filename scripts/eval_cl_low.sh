export TRANSFORMERS_CACHE=/data/yongxi/.cache/huggingface
export CUDA_VISIBLE_DEVICES=7
python -m llava.eval.eval_cl \
  --model-path checkpoint_low/behavior_il_seed1_task0   \
  --model-base model/llava/llava-v1.5-7b \
  --data ./data_low \
  --splits valid_unseen \
  --incremental_setup behavior_il \
  --stream_seed 1 \
  --incremental_task 1 \
  --output_file eval_results_low/behavior_il_seed1_task0_valid_unseen.json \
  --temperature 0