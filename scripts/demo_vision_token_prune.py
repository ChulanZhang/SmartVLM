#!/usr/bin/env python3
"""
Demo: run one image through the vision token scheduler (Exp1) and print intermediate
variables to verify tokens are pruned as expected.

Usage (from repo root):
  PYTHONPATH=src python scripts/demo_vision_token_prune.py \\
    --model-path checkpoints/ada-llava-vision-token-scheduler-v1.5-7b \\
    --image-file docs/snowman.jpg \\
    --token_budget 0.5

Prints: token_budget, N (total patches), K (kept per sample), keep_mask sums, logits stats,
and optionally runs one generate step to show the model output.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root: PYTHONPATH=src or add src to path
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images
from llava.utils import disable_torch_init

from adallava.model.builder import load_pretrained_model


def main():
    parser = argparse.ArgumentParser(description="Vision token prune demo: print intermediate vars.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--token_budget", type=float, default=0.5, help="Vision token ratio in [0,1]")
    parser.add_argument("--no-generate", action="store_true", help="Only print vision path vars, do not run generate")
    parser.add_argument("--query", type=str, default="What is in this image?")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (vision token selection is deterministic at eval)")
    parser.add_argument("--brief", action="store_true", help="Only print token_budget and model reply (for sweep scripts)")
    parser.add_argument("--debug", action="store_true", help="Print prompt_len, gen_len, and first generated token ids (for diagnosing empty output)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path)

    if getattr(model.config, "token_selecting", "none") != "adaptive":
        raise RuntimeError("This demo expects token_selecting='adaptive'. Checkpoint may be standard AdaLLaVA.")

    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Load and process one image
    from PIL import Image
    image = Image.open(args.image_file).convert("RGB")
    images_tensor = process_images([image], image_processor, model.config).to(device=device, dtype=dtype)
    batch_size = images_tensor.size(0)

    N = getattr(model.config, "num_vision_patches", 576)
    token_budget_t = torch.full((batch_size,), args.token_budget, device=device, dtype=dtype)

    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device)  # same device as main model to avoid cuda:0 vs cuda:1
    vision_tower.to(device=device, dtype=dtype)

    # 1) Vision encoder with budget token -> [B, N+2, C]
    with torch.inference_mode():
        vision_output = model.vision_encoder_with_budget(
            images_tensor, token_budget_t, vision_tower=vision_tower
        ).to(dtype)
    B, seq_len, C = vision_output.shape
    assert seq_len == N + 2, f"Expected N+2={N+2}, got {seq_len}"

    # 2) Controller: budget repr -> logits; then Gumbel top-K -> mask
    # Clone so inference-mode tensors can be used in Linear (avoids "inference tensors cannot be saved for backward").
    budget_repr = vision_output[:, -1, :].clone()
    logits = model.vision_token_controller.logit_head(budget_repr)
    selected, keep_mask = model.vision_token_controller(vision_output.clone(), token_budget_t)

    # K per sample (what we requested)
    if token_budget_t.dtype in (torch.float32, torch.float16, torch.bfloat16):
        K_per_batch = (token_budget_t * N).long().clamp(1, N)
    else:
        K_per_batch = token_budget_t.long().clamp(1, N)

    if not args.brief:
        print("=" * 60)
        print("Vision token prune demo — intermediate variables")
        print("=" * 60)
        print(f"  token_budget (ratio)     = {args.token_budget}")
        print(f"  N (total patches)         = {N}")
        print(f"  K (requested per sample)  = {K_per_batch.tolist()}")
        print(f"  keep_mask shape           = {keep_mask.shape}")
        print(f"  keep_mask sum per sample  = {keep_mask.sum(dim=1).long().tolist()}  (should equal K)")
        print(f"  selected (masked) shape   = {selected.shape}")
        print(f"  nonzero patches per sample (after mask) = {(selected.abs().sum(dim=-1) > 0).sum(dim=1).tolist()}")
        print(f"  logits shape              = {logits.shape}")
        print(f"  logits min/max/mean       = {logits.min().item():.4f} / {logits.max().item():.4f} / {logits.mean().item():.4f}")
        print("=" * 60)
        mask_sums = keep_mask.sum(dim=1).long()
        for b in range(B):
            k_b = int(K_per_batch[b].item())
            m_b = int(mask_sums[b].item())
            if m_b != k_b:
                print(f"  [WARN] sample {b}: mask sum={m_b} != K={k_b}")
            else:
                print(f"  [OK] sample {b}: mask sum={m_b} == K={k_b}")
        print("=" * 60)

    if args.no_generate:
        if args.brief:
            print(f"token_budget={args.token_budget} (no generate)")
        else:
            print("Skipping generate (--no-generate). Done.")
        return

    # Optional: one generate step to show model output
    from llava.mm_utils import tokenizer_image_token
    from llava.conversation import conv_templates

    model.current_token_budget = args.token_budget
    qs = DEFAULT_IMAGE_TOKEN + "\n" + args.query
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    image_sizes = [image.size]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            latency=1.0,
            max_new_tokens=64,
            do_sample=False,
            return_dict_in_generate=True,
        )
    out = outputs.sequences
    prompt_len = getattr(outputs, "prompt_len", None)
    gen_len = getattr(outputs, "gen_len", None)
    if prompt_len is None:
        prompt_len = input_ids.shape[1] + 1  # fallback: original prompt + latency token
    if gen_len is None:
        gen_len = out.shape[1] - prompt_len
    # Handle framework returning only generated part (out shorter than prompt_len).
    if prompt_len > out.shape[1]:
        prompt_len = 0
        gen_len = out.shape[1]
    gen_len = max(0, gen_len)

    # Decode only the *generated* part so reply is the model answer, not full prompt.
    gen_ids = out[0, prompt_len:].tolist() if prompt_len < out.shape[1] else []
    vocab_size = len(tokenizer)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    pad_id = getattr(tokenizer, "pad_token_id", None)

    def _safe_id(i: int) -> int:
        if not (0 <= i < vocab_size):
            return eos_id
        if i == 0 and pad_id is None:
            return eos_id  # LLaMA sp often has no piece for id 0
        return i

    safe_gen_ids = [_safe_id(i) for i in gen_ids]
    reply = tokenizer.decode(safe_gen_ids, skip_special_tokens=True).strip()

    if args.debug:
        print(f"[DEBUG] prompt_len={prompt_len} gen_len={gen_len} seq_len={out.shape[1]}")
        if gen_len > 0:
            head = gen_ids[: min(20, len(gen_ids))]
            eos_id_val = tokenizer.eos_token_id
            print(f"[DEBUG] first generated token ids: {head} (eos_token_id={eos_id_val})")
            print(f"[DEBUG] all_eos: {all(t == eos_id_val for t in gen_ids)}")

    if args.brief:
        print(f"token_budget={args.token_budget}")
        print(f"reply: {reply[:200]}{'...' if len(reply) > 200 else ''}")
    else:
        print("Model reply (first 64 new tokens):")
        print(reply)
        print("Done.")


if __name__ == "__main__":
    main()
