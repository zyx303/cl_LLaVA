import argparse
import json
import os
import re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)




def _pick_human_and_ref(conv: List[Dict[str, str]]) -> Tuple[str, str]:
    """Pick the last human->gpt pair as (prompt, reference)."""
    if not conv:
        return '', ''
    prompt = ''
    ref = ''
    for i in range(len(conv) - 1):
        a, b = conv[i], conv[i + 1]
        if a.get('from') == 'human' and b.get('from') == 'gpt':
            prompt, ref = a.get('value', ''), b.get('value', '')
    return prompt, ref


# ---------------------------
# Image helpers
# ---------------------------
def _load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image


# ---------------------------
# ROUGE (simple implementation)
# ---------------------------
_ALNUM_CJK = re.compile(r"[^\w\u4e00-\u9fff]+", flags=re.UNICODE)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = _ALNUM_CJK.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    return _normalize(text).split()


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(pred: str, ref: str, n: int = 1) -> Dict[str, float]:
    pt = _tokens(pred)
    rt = _tokens(ref)
    p_ngrams = _ngrams(pt, n)
    r_ngrams = _ngrams(rt, n)
    if not p_ngrams and not r_ngrams:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not p_ngrams or not r_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    from collections import Counter
    pc = Counter(p_ngrams)
    rc = Counter(r_ngrams)
    overlap = sum((pc & rc).values())
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(rc.values()))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_len(xs: List[str], ys: List[str]) -> int:
    # classic DP LCS length
    m, n = len(xs), len(ys)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        xi = xs[i - 1]
        row = dp[i]
        prow = dp[i - 1]
        for j in range(1, n + 1):
            if xi == ys[j - 1]:
                row[j] = prow[j - 1] + 1
            else:
                row[j] = row[j - 1] if row[j - 1] >= prow[j] else prow[j]
    return dp[m][n]


def rouge_l(pred: str, ref: str) -> Dict[str, float]:
    pt = _tokens(pred)
    rt = _tokens(ref)
    if not pt and not rt:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pt or not rt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_len(pt, rt)
    precision = lcs / len(pt)
    recall = lcs / len(rt)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------
# Evaluation
# ---------------------------
@torch.inference_mode()
def evaluate(args):
    # model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, use_peft=True
    )

    # conv mode selection (copy from run_llava.py with small tweaks)
    if args.conv_mode is None:
        name = model_name.lower()
        if "llama-2" in name:
            args.conv_mode = "llava_llama_2"
        elif "mistral" in name:
            args.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in name:
            args.conv_mode = "chatml_direct"
        elif "v1" in name:
            args.conv_mode = "llava_v1"
        elif "mpt" in name:
            args.conv_mode = "mpt"
        else:
            args.conv_mode = "llava_v0"

    data_path = os.path.join(args.data, args.splits, f"{args.incremental_setup}_stream{args.stream_seed}", f"step{args.incremental_task}.json")
    print(f"Loading data from {data_path} ...")
    data = json.load(open(data_path, 'r'))
    print(f"Loaded {len(data)} examples.")


    # results
    total = 0
    sum_r1, sum_r2, sum_rl = 0.0, 0.0, 0.0

    writer = open(os.path.expanduser(args.output_file), 'w', encoding='utf-8') if args.output_file else None

    for ex in tqdm(data):
        img_path = ex['image']

        try:
            image = _load_image(img_path)
        except Exception as e:
            print(f"[WARN] skip example {ex.get('id')} due to image load error: {e}")
            continue

        prompt_raw, ref = _pick_human_and_ref(ex.get('conversations', []))
        if not prompt_raw or not ref:
            print(f"[WARN] skip example {ex.get('id')} due to empty prompt/ref")
            continue

        # # insert image tokens similar to run_llava
        # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # qs = prompt_raw
        # if IMAGE_PLACEHOLDER in qs:
        #     if model.config.mm_use_im_start_end:
        #         qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
        #     else:
        #         qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
        # else:
        #     if model.config.mm_use_im_start_end:
        #         qs = image_token_se + "\n" + qs
        #     else:
        #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv_mode = "llava_v1"
        qs = prompt_raw
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # process image
        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
        image_sizes = [image.size]

        # tokenize and generate
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # compute ROUGE
        r1 = rouge_n(pred, ref, 1)["f1"]
        r2 = rouge_n(pred, ref, 2)["f1"]
        rl = rouge_l(pred, ref)["f1"]

        total += 1
        sum_r1 += r1
        sum_r2 += r2
        sum_rl += rl

        if writer:
            out_js = {
                "id": ex.get('id'),
                "image": ex.get('image'),
                "pred": pred,
                "ref": ref,
                "rouge1_f1": r1,
                "rouge2_f1": r2,
                "rougeL_f1": rl,
            }
            writer.write(json.dumps(out_js, ensure_ascii=False) + "\n")
            writer.flush()


    if total == 0:
        print(json.dumps({"count": 0, "rouge1_f1": 0, "rouge2_f1": 0, "rougeL_f1": 0}))
        return

    avg = {
        "count": total,
        "rouge1_f1": round(sum_r1 / total * 100, 2),
        "rouge2_f1": round(sum_r2 / total * 100, 2),
        "rougeL_f1": round(sum_rl / total * 100, 2),
    }
    writer.write(json.dumps(avg, ensure_ascii=False))
    writer.flush()
    if writer:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data", default='./data',type=str, required=True, help="JSONL or JSON; each item has id, image, conversations")
    parser.add_argument('--splits',default='valid_unseen', type=str, help='which split to eval')
    parser.add_argument('--incremental_setup', type=str, required=True, help='which incremental learning setup to eval')
    parser.add_argument("--stream_seed", type=int, required=True, help='which stream seed to eval')
    parser.add_argument("--incremental_task", type=int, required=True, help='which incremental task to eval')
    parser.add_argument("--output_file", type=str, default="", help="Optional JSONL to save per-sample results")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
