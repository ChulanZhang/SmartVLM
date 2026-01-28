"""
Single-image QA demo for the vision token scheduler (Exp1: adaptive vision token prune).
Usage: python -m src.adallava.eval.run_vision_token_scheduler --model-path <ckpt> --token_budget 0.5 --image-file <path> --query "Describe this image."
"""

import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token

from ..model.builder import load_pretrained_model

from PIL import Image
import requests
from io import BytesIO
import re


def image_parser(args):
    return args.image_file.split(args.sep)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    return [load_image(f) for f in image_files]


def eval_model(args):
    disable_torch_init()

    model_name = args.model_name or "llava_v1"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path)

    if getattr(model.config, "token_selecting", "none") != "adaptive":
        raise RuntimeError(
            "run_vision_token_scheduler expects a checkpoint with token_selecting='adaptive'. "
            "Use run_ada_llava.py for standard AdaLLaVA."
        )

    model.current_token_budget = args.token_budget

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = getattr(args, "conv_mode", "llava_v1")

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            latency=args.latency,
            return_dict_in_generate=True,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    output_ids = outputs.sequences
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
    print(f"[token_budget={args.token_budget}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision token scheduler (adaptive) demo.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--latency", type=float, default=1.0, help="LLM latency budget (1.0 = full)")
    parser.add_argument(
        "--token_budget",
        type=float,
        default=1.0,
        help="Vision token budget ratio in [0,1]. 1.0 = use all tokens.",
    )
    args = parser.parse_args()
    eval_model(args)
