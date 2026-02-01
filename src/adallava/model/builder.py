#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import json
import os
import warnings
import shutil

from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from .ada_llava_llama import AdaLlavaLlamaForCausalLM


def _load_vision_tower_from_checkpoint(model, model_path: str) -> None:
    """Load vision tower weights from a local checkpoint into model.model.vision_tower.

    LLaVA's vision_tower is lazy-loaded via load_model() from mm_vision_tower (HF URL), so
    checkpoint weights for model.vision_tower.* are not used during from_pretrained. For
    checkpoints that were trained with the vision tower (e.g. Exp1), we overwrite with
    checkpoint weights when present.
    """
    prefix = "model.vision_tower."
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return
    with open(index_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    vision_keys = [k for k in weight_map if k.startswith(prefix)]
    if not vision_keys:
        return
    shard_to_keys = {}
    for k in vision_keys:
        shard = weight_map[k]
        shard_to_keys.setdefault(shard, []).append(k)
    state = {}
    for shard_file, keys in shard_to_keys.items():
        path = os.path.join(model_path, shard_file)
        if not os.path.isfile(path):
            return
        try:
            from safetensors.torch import load_file
            shard_state = load_file(path, device="cpu")
        except Exception:
            return
        for k in keys:
            if k in shard_state:
                new_k = k[len(prefix):]
                state[new_k] = shard_state[k]
    if not state:
        return
    vision_tower = model.get_model().vision_tower
    missing, unexpected = vision_tower.load_state_dict(state, strict=False)
    if missing:
        warnings.warn(
            f"Vision tower: {len(missing)} keys in module not in checkpoint (expected if structure differs)."
        )
    # Log so user knows trained ViT weights were applied (no "not used" from from_pretrained).
    import logging
    logging.getLogger(__name__).info(
        f"Loaded {len(state)} vision tower weights from checkpoint (local path)."
    )


def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    elif not load_8bit and not load_4bit and device_map == "auto":
        # Use explicit single-device map to avoid accelerate's infer_auto_device_map tied-weights
        # bug (IndexError when tied_param names don't match modules_to_treat).
        kwargs["device_map"] = {"": "cuda:0"}
        device_map = "cuda:0"  # so vision_tower.load_model(device_map=...) uses same device

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AdaLlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    # Overwrite vision tower with checkpoint weights when loading from a local path (e.g. Exp1).
    if os.path.isdir(model_path):
        _load_vision_tower_from_checkpoint(model, model_path)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
