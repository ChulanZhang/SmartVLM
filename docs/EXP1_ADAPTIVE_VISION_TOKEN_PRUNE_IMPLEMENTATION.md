# Experiment 1: Adaptive Vision Token Prune — Implementation Plan

This document is the full implementation plan for **Experiment 1: Adaptive Vision Token Pruning vs. Static Token Reduction Methods**. It defines the goal, the design (including how AdaLLaVA’s two-path budget–content influence is mirrored on the vision side), the technical choices, the implementation phases, and a **code-level implementation plan** with file paths, interfaces, and step-by-step changes.

---

## 1. Goal and Scope

**Goal:** Compare the proposed adaptive vision token pruning method to training-free baselines (PruMerge, FastV) in efficiency and accuracy.

**In scope:**
- A **vision-side token-level controller** that outputs per-token keep probabilities conditioned on a token budget and on content.
- Reusing AdaLLaVA’s pattern: a **budget token** fuses with vision tokens (inside the vision encoder or right after it), and the controller uses that representation plus a scalar budget to decide **which** and **how many** tokens to keep.
- End-to-end training with language modeling loss (no extra FLOPs loss in the baseline plan).
- Evaluation at multiple token budgets; reporting accuracy and FLOPs; comparison with LLaVA (full), PruMerge, and FastV.

**Out of scope for Exp1:** LLM-side adaptive depth/width. The LLM runs at full width and depth; only the number of vision tokens is adapted.

---

## 2. Design Overview

### 2.1 How AdaLLaVA’s scheduler uses budget and content (two paths)

In AdaLLaVA, the scheduler’s decision is influenced in two ways:

**Path 1 — Content and budget jointly shape “which” to select:**  
The latency scalar is encoded to `latency_embedding` and inserted as a **latency token** into the LLM’s `inputs_embeds` (e.g. after image tokens, near the end of the prompt). That token passes through the first `num_prefix_layers` LLM layers. In each layer it does **self-attention** with image and text tokens. When the scheduler is called, it reads `x = hidden_states[:, latency_token_position, :]`, i.e. the hidden state at that position **after** those prefix layers. So `x` is a fusion of “budget” (from the initial latency embedding) and “content” (from attention to image/text). The scheduler then computes `logits = mlp_head(x)`. Thus, **which** layers/heads get high logits is indirectly determined by both budget and content via this fused `x`.

**Path 2 — Budget directly fixes “how many” to select:**  
The scalar `latency` is quantized to an integer `n` (e.g. number of layers to use). The scheduler runs `n_times_gumbel_softmax(logits, n=n, ...)` so that **exactly n** items are selected. So **how many** is set directly by the latency scalar, not by the logits’ values.

In code: latency is inserted in `ada_llava_llama.py` (L136–141, L186–188); the scheduler reads `hidden_states` at `latency_token_position` in `modeling_ada_llama.py` when `idx == num_prefix_layers` (L337–340), at which point `hidden_states` has been updated by layers 0..num_prefix_layers-1; and `simple_scheduler.py` uses `latency_quantizing` and `n_times_gumbel_softmax(..., n=latency_.item(), ...)` (L56–62).

### 2.2 Vision-side analog: budget token + token controller

We mirror this on the vision side:

1. **Budget token and fusion**  
   A **budget token** is added to the vision encoder’s **input** sequence (e.g. appended after CLS and patch tokens). “Appended at the end” here means: it is the **last position** of the sequence fed into the encoder. That sequence is then passed through the ViT’s transformer layers, so the budget token attends to all vision tokens and vice versa in every layer. Putting it at the beginning `[BUDGET, CLS, P_1, ..., P_N]` would also allow full fusion; we default to append.  
   The controller is fed the vision encoder output (including the budget token’s representation). So:
   - **Path 1:** The representation used to compute keep-logits is budget- and content-aware (budget token is inside the ViT and fuses with all patches via self-attention).
   - **Path 2:** A scalar token budget K (or ratio r) is used to choose **how many** vision tokens to keep via Gumbel top-K, in the same way latency is used in AdaLLaVA.

2. **Token-level controller**  
   The controller maps vision encoder output (and, in some designs, an explicit budget embedding) to **per-patch keep logits**, then applies Gumbel-Softmax top-K with the current token budget K so that exactly K patches are kept. CLS is always kept. Only the selected tokens (CLS + K patches) are sent to the mm_projector and then to the LLM.

3. **Data flow**  
   `images → vision encoder (with optional budget token) → controller(vision_output, token_budget) → selected [B, 1+K, C_v] → mm_projector → LLM`.  
   For compatibility with LLaVA’s multimodal path, we can fix the number of output tokens per run (e.g. always 1+K for a chosen K), or pad/truncate to a fixed length in a first version.

### 2.3 Budget inside the encoder

The vision encoder (e.g. CLIP ViT) does: **patch embed → position embed → transformer layers → output**. We append one budget embedding after patch (and CLS) so the sequence is `[CLS, P_1, ..., P_N, BUDGET]` with shape `[B, N+2, C_v]`. This is fed through the **same** ViT transformer stack. The budget token’s output is then a representation that has already interacted with all patches over all layers. Fusion happens **inside** the ViT via self-attention.

The controller takes **only** this budget token output (last position) and one linear `Linear(C_v, N)` → per-patch logits `[B, N]`, same design as AdaLLaVA’s scheduler: one fused vector → `mlp_head` → per-item logits. Code: `vision_encoder_with_budget` returns `[B, N+2, C]`; `vision_token_controller` uses `vision_output[:, -1, :]` and `logit_head(budget_repr)`.

**AdaLLaVA scheduler (for reference):** The LLM-side scheduler takes **only** the latency token’s hidden state `x` after prefix layers; `logits = self.mlp_head(x)` with `mlp_head = nn.Linear(hidden_size, num_sub_layer)` — a single linear layer. We mirror this: budget token output → `Linear(C_v, N)` → per-patch logits.

---

## 3. Technical Design

### 3.1 Vision encoder and budget token

- **C_v:** Vision encoder hidden dimension (e.g. 1024 for ViT-L; same as `mm_hidden_size`).
- **Where to inject:** One extra token after patch+CLS embedding, i.e. last position of the encoder input. Sequence length becomes N+2; we extend position embeddings by one (e.g. one learned vector for that position).
- **Budget embedding:** Scalar budget b ∈ [0,1] (or discrete K). Encode like AdaLLaVA’s scheduler: sinusoidal over 256 dims then MLP to `C_v`, so we get `[B, 1, C_v]`; this is appended to the patch sequence before the ViT.
- **Shapes:** Patch+CLS → `[B, N+1, C_v]`; with budget → `[B, N+2, C_v]`; after ViT → `[B, N+2, C_v]`. The controller predicts keep/drop only for the N patches; CLS and budget token are not sent to the projector.
- **Pipeline unchanged:** Controller output is always `[B, N, C_v]` (only N patch tokens) so mm_projector and LLM receive the same token count as LLaVA; projector input size stays fixed.

### 3.2 Token-level controller

- **Inputs:** `vision_output` `[B, N+2, C_v]` (last position = budget token output); and `token_budget` (scalar K or ratio r, or per-batch).
- **Logits:** Only the budget token output (last position) → one linear `Linear(C_v, N)` → per-patch logits `[B, N]`. No concat with patch tokens (AdaLLaVA-style).
- **Discrete selection:** Reuse AdaLLaVA’s Gumbel top-k: `masked_gumbel_softmax_topk` or a loop of `n_times_gumbel_softmax` to select exactly K patches. Training: hard Gumbel with straight-through; inference: hard top-K.
- **Output:** Mask or indices for K patches. We output fixed `[B, N, C_v]` (N patch positions, unselected masked to 0) so mm_projector input size is unchanged; no CLS token is passed to the projector.

### 3.3 Training and evaluation

- **Training:** Same as AdaLLaVA: **only LM loss**. Each batch samples a random token budget K (or r mapped to K), e.g. from a uniform over a range. The controller receives that K and must choose exactly K patches via Gumbel; no extra FLOPs or budget regularizer in the baseline. The comparison with “no token prune” is done **at evaluation time** by plotting accuracy vs. FLOPs and comparing to full LLaVA, PruMerge, and FastV, not by an extra loss term.
- **Evaluation:** Run at several fixed K (e.g. 64, 128, 256, 384, 576). For each K, compute accuracy and FLOPs. FLOPs in Exp1 are dominated by **sequence length** (vision token count + text length) because we keep LLM width and depth fixed; we use vision_encoder (fixed) + mm_projector(∝ K) + LLM(∝ seq_len) to form the accuracy–FLOPs curve.

### 3.4 Budget → compute: AdaLLaVA vs. vision token scheduler

- **AdaLLaVA (LLM side):** The scalar `latency` ∈ [0,1] is **not** converted to FLOPs in the training loop. It is quantized (e.g. via `latency_quantizing`) to an integer that determines **how many LLM layers** (or sub-layers) to run (e.g. latency 1.0 → run all non-prefix layers). So the “budget” is a **compute ratio** mapped to discrete units (number of layers); there is no explicit FLOPs formula in the scheduler forward.
- **Vision token scheduler (Exp1):** Similarly, `token_budget` ∈ [0,1] is mapped to **how many** vision tokens to keep: K = round(token_budget × N). Again, this is a ratio → discrete count (tokens), not a FLOPs value. FLOPs are used only **at evaluation** (e.g. accuracy–FLOPs curves) via an analytic formula that combines vision encoder, mm_projector, and LLM cost given K and sequence length.
- **Alignment:** Both sides use the same *pattern*: a normalized budget (ratio) → quantized count of units (layers vs. tokens). Neither converts budget to FLOPs inside the forward pass. A single **unified** “whole-model FLOPs budget” would require a separate FLOPs model (e.g. FLOPs ≈ f(K_tokens, N_layers)), either at eval time (as in scripts that estimate FLOPs from budget) or in a joint controller; that is out of scope for the baseline Exp1 design.

### 3.6 FLOPs and masked tokens (current implementation)

**Training vs evaluation (aligned with AdaLLaVA):**

- **Training:** The controller outputs **fixed shape** `[B, N, C]` with **unselected positions zeroed** (masked to 0). This tensor is passed to `mm_projector` and then to the LLM. Sequence length remains N so that backward and straight-through work.
- **Evaluation:** In `encode_images`, when **`not self.training`**, we **gather only the K selected tokens** into `[B, K, C]` and pass that to mm_projector. So mm_projector and the LLM see sequence length **K**; **FLOPs truly decrease**. LLaVA merge uses `image_features.size(1)` for the number of image token positions, so no change to `prepare_inputs_labels_for_multimodal` is required.
- **Plot / estimate:** `scripts/plotting/plot_accuracy_flops.py` uses `estimate_flops_from_budget(token_budget)`, which assumes effective sequence length **(1+K)+text**. With the eval-time gather, this now matches the **actual** forward at evaluation.

### 3.5 Edge cases

- K=0: require at least 1 token (e.g. CLS only).
- K≥N: no pruning; use all patches.
- Batch with different K per sample: support per-sample token_budget like AdaLLaVA’s per-sample latency.

---

## 4. Implementation

The adaptive path uses: (1) `VisionEncoderWithBudgetToken` to append budget to patch sequence and run ViT → `[B, N+2, C_v]`; (2) `VisionTokenController` with only the budget token output → `Linear(C_v, N)` → Gumbel top-K. FLOPs counting (vision + mm_projector + LLM), accuracy–FLOPs curves, and comparison with LLaVA, PruMerge, and FastV are done at evaluation.

---

## 5. Code-Level Implementation Plan

### 5.1 New files and modules

| Path | Purpose |
|------|--------|
| `src/adallava/model/multimodal_encoder/budget_embedding.py` | Encode scalar budget → vector (sinusoidal + MLP). |
| `src/adallava/model/multimodal_encoder/vision_token_controller.py` | Per-patch logits + Gumbel top-K; inputs vision features and token_budget. |
| `src/adallava/model/multimodal_encoder/vision_with_budget_token.py` | Wrapper that appends budget token, runs ViT, returns `[B, N+2, C_v]`. |

### 5.2 Budget embedding (`budget_embedding.py`)

**Interface:**

- `BudgetEmbedding(dim_out: int, num_freqs: int = 128, hidden_dim: int = 256)`  
  - `forward(budget: Tensor) -> Tensor`  
  - `budget`: `[B]` or `[B, 1]`, values in [0, 1].  
  - Output: `[B, dim_out]`.

**Implementation sketch:**

- Reuse logic from `simple_scheduler.py` `latency_encoding` (L37–54): scale budget to [0,2π], sinusoidal over `num_freqs` to get 256-d, then a small MLP (e.g. `FeedForward(256, hidden_dim, dim_out)`) to `dim_out`. For vision, `dim_out = C_v` (e.g. 1024 for ViT-L).
- No quantization inside this module; quantization of “how many tokens” is done in the controller when calling Gumbel (e.g. K = round(r * N) or K from a discrete set).

**References:** `src/adallava/model/scheduler/simple_scheduler.py` (`latency_encoding`, `scheduler_up_proj`), `scheduler_utils.latency_quantizing` if we want discrete K levels.

### 5.3 Vision token controller (`vision_token_controller.py`)

**Interface:**

- `VisionTokenController(vision_dim: int, num_patches: int, tau: float = 5.0)`  
  - `forward(vision_output: Tensor, token_budget: Tensor) -> Tuple[Tensor, Tensor]`  
  - `vision_output`: `[B, N+2, C]` (last position = budget token output from ViT).  
  - `token_budget`: `[B]` with values in [0,1] (ratio) or integer K per batch; or a single scalar.  
  - Returns: `(selected_features, mask)` where `selected_features` is `[B, N, C]` (patch tokens only, unselected masked to 0) and `mask` is `[B, N]`.

**Implementation:** Extract `budget_repr = vision_output[:, -1, :]` and `patch_tokens = vision_output[:, 1:-1, :]`. Logits: `logits = self.logit_head(budget_repr)` with `logit_head = nn.Linear(vision_dim, num_patches)`. Map `token_budget` to K per batch; select K via `n_times_gumbel_softmax`; apply mask to patch tokens. Straight-through in training.

### 5.4 Vision encoder with budget token (`vision_with_budget_token.py`)

**Interface:**

- `VisionEncoderWithBudgetToken(vision_tower: nn.Module, budget_embedding_module: nn.Module, vision_hidden_size: int)`  
  - `forward(images: Tensor, budget: Tensor) -> Tensor`  
  - Returns `[B, N+2, C_v]` where the last position is the budget token’s output.

**Implementation sketch:**

1. Obtain the vision tower’s patch (and CLS) embedding logic. In LLaVA this is usually inside the same “vision_tower” object that has `vision_model.embeddings` and `vision_model.encoder`. We need to:
   - Run patch+CLS embedding on `images` → `[B, N+1, C_v]`.
   - Compute `budget_emb = budget_embedding_module(budget)` → `[B, C_v]`, then `[B, 1, C_v]`.
   - Concatenate: `x = cat([patch_cls, budget_emb], dim=1)` → `[B, N+2, C_v]`.
2. Handle positions: the standard ViT uses a fixed number of position embeddings. Add one more learned parameter `self.budget_pos_embed` of shape `[1, 1, C_v]` and add it to the last position of `x` (or use the last patch position for the budget token as an ablation).
3. Run the **same** transformer stack as the original ViT on `x`. This may require calling `vision_tower.vision_model.encoder(x)` with an encoder that accepts variable length, or wrapping the encoder forward so the input is `[B, N+2, C_v]`. If the official ViT has a fixed `num_positions`, we need to extend its position embedding (or pass a custom embedding table) and run the transformer on the extended sequence.
4. Return the full output `[B, N+2, C_v]` for the controller.

**References:** `prumerge_utils.py` uses `self.vision_tower(images, output_hidden_states=True)` and `feature_select(image_forward_outs)`; the tower’s structure is `vision_tower.vision_model.encoder.layers[*]`. For a minimal change, implement a wrapper that (a) does patch embed + concat budget, (b) runs the encoder on the extended sequence (may require a shallow copy or subclass of the encoder to accept seq_len N+2).

### 5.5 Integration in `encode_images` and config

**File:** `src/adallava/model/ada_llava_llama.py`.

**Current `encode_images` (L83–95):**

- Dispatches on `self.config.token_selecting` among `"none"`, `"prumerge"`, `"prumerge+"`; for prumerge it calls `prune_merge`/`prune_merge_plus` on the vision tower then `mm_projector`.  

**Required change:**

- Add branch `token_selecting == "adaptive"`: get `vision_tower`; sample or use provided `token_budget`; `vision_output = self.vision_encoder_with_budget(images, token_budget, vision_tower)` → `[B, N+2, C_v]`; `selected, _ = self.vision_token_controller(vision_output, token_budget)`; `image_features = self.get_model().mm_projector(selected)`; return `image_features`.

**New/updated config fields (e.g. in `AdaLlavaConfig` or model `__init__`):**

- `token_selecting: str = "adaptive"` when using this path.  
- `vision_controller_budget_min`, `vision_controller_budget_max`: for sampling r ∈ [min, max] and mapping to K.  
- `vision_controller_tau: float = 5.0`.  
- `num_vision_patches: int` (e.g. 576 for 336×336 ViT) for K range and Gumbel.

**New attributes on `AdaLlavaLlamaForCausalLM` when `token_selecting == "adaptive"`:**
- `self.budget_embedding = BudgetEmbedding(C_v, ...)`  
- `self.vision_token_controller = VisionTokenController(C_v, num_patches=N, tau=...)`  
- `self.vision_encoder_with_budget = VisionEncoderWithBudgetToken(...)`  

Initialization should only create these when `config.token_selecting == "adaptive"`, and `encode_images` should branch on the same flag.

### 5.6 Token budget sampling and `_get_token_budget`

- Add a method `_get_token_budget(self, batch_size: int) -> Tensor` used in training.  
- Implementation: same pattern as `scheduler.get_random_latency(batch_size)` in AdaLLaVA — e.g. sample K uniformly in `[K_min, K_max]` then optionally normalize to [0,1] if the controller expects ratio, or pass K directly if the controller expects integer.  
- For evaluation, `token_budget` is set from the caller (e.g. `generate` or eval script) and passed through so that `encode_images` receives the desired K (or r) per batch or per sample.

### 5.7 Training script

**File:** `src/adallava/train/train.py` (or the script that builds the model and runs training).

- Add `--token_selecting adaptive` and, if needed, `--vision_controller_budget_min`, `--vision_controller_budget_max` to the model/data args.
- When building the model, ensure `AdaLlavaConfig` (or the constructor of `AdaLlavaLlamaForCausalLM`) gets `token_selecting="adaptive"` and the new vision-controller-related arguments so that `budget_embedding` and `vision_token_controller` are created.
- No change to the loss: keep LM loss only. The dataloader does not need to provide `token_budget` if it is sampled inside the model at the start of each forward (as in AdaLLaVA’s latency).

### 5.8 Demo and inspecting intermediate variables

- **Script:** `scripts/demo_vision_token_prune.py` runs one image through the vision token scheduler and prints intermediate variables (token_budget, N, K, keep_mask sum per sample, logits shape and stats, nonzero patch count). Use this to verify that tokens are pruned as expected (mask sum == K).
- **Usage (from repo root):** `PYTHONPATH=src python scripts/demo_vision_token_prune.py --model-path <ckpt> --image-file docs/snowman.jpg --token_budget 0.5`. Add `--no-generate` to skip the generate step and only print vision-path vars.

### 5.9 Evaluation and FLOPs

- Add a CLI or config option for a list of token budgets, e.g. `--eval_budgets 64 128 256 384 576`.
- For each budget K, set the model (or the batch) to use that K when calling `encode_images` / `generate`, run the eval set, and record accuracy.
- FLOPs: use a small hook or an analytic formula. Main terms: (1) vision encoder forward (constant), (2) mm_projector on (1+K) tokens *if* sequence is reduced to K; **current implementation** keeps N positions (masked to 0), so actual FLOPs are over N (see §3.6). The helper in `scripts/plotting/plot_accuracy_flops.py` (`estimate_flops_from_budget`) assumes effective length (1+K)+text for plotting accuracy–FLOPs curves. Plot accuracy vs. FLOPs and compare to full LLaVA, PruMerge, and FastV.

### 5.10 Summary of code touchpoints

| Location | Change |
|----------|--------|
| `src/adallava/model/multimodal_encoder/budget_embedding.py` | New: `BudgetEmbedding` (sinusoidal + MLP). |
| `src/adallava/model/multimodal_encoder/vision_token_controller.py` | New: `VisionTokenController` (logits + Gumbel top-K). |
| `src/adallava/model/multimodal_encoder/vision_with_budget_token.py` | Wrapper that appends budget, runs ViT. |
| `src/adallava/model/ada_llava_llama.py` | In `encode_images`, add branch `"adaptive"`; in `__init__`, create `budget_embedding` and `vision_token_controller` when `token_selecting=="adaptive"`; add `_get_token_budget`. |
| `AdaLlavaConfig` / model ctor | New args: `token_selecting="adaptive"`, `vision_controller_*`. |
| `src/adallava/train/train.py` | New CLI args for adaptive path; pass them into config. |
| Eval script / entrypoint | `--eval_budgets`; loop over K, record accuracy and FLOPs. |
| `src/adallava/model/scheduler/scheduler_utils.py` | Only **reuse**; no change. Use `masked_gumbel_softmax_topk`, `n_times_gumbel_softmax` as references. |

---

## 6. Design Rationale (concise)

- **Budget token inside the encoder:** Puts “how much to keep” and “what is in the image” in one representation that the controller sees, analogous to AdaLLaVA’s latency token in the LLM.
- **Two paths:** “Which” tokens to keep is driven by content and budget via the representation passed to the controller (Path 1); “how many” is set by the scalar token budget (Path 2), matching AdaLLaVA’s use of the latency scalar.
- **Gumbel-Softmax top-K:** Keeps the same differentiable, discrete-selection recipe as AdaLLaVA; temperature and straight-through follow the existing scheduler.
- **LM loss only:** Keeps the protocol aligned with AdaLLaVA; comparison with no-prune and other baselines is done at eval time with accuracy–FLOPs curves.
- **FLOPs:** With fixed LLM width/depth, FLOPs scale with sequence length (vision token count + text); we measure and report them accordingly for each token budget K.

---

## 7. Reference: AdaLLaVA code locations

| Need | File and symbols |
|------|------------------|
| Gumbel top-k / n-from-logits | `src/adallava/model/scheduler/scheduler_utils.py`: `masked_gumbel_softmax`, `n_times_gumbel_softmax`, `masked_gumbel_softmax_topk` |
| Latency encoding (sinusoidal + MLP) | `src/adallava/model/scheduler/simple_scheduler.py`: `latency_encoding()`, `scheduler_up_proj` |
| Latency quantizing | `src/adallava/model/scheduler/scheduler_utils.py`: `latency_quantizing()` |
| Latency token insertion | `src/adallava/model/ada_llava_llama.py`: `insert_latency_token`, L136–141, L186–188 |
| Scheduler call and hidden-state read | `src/adallava/model/language_model/ada_llama/modeling_ada_llama.py`: L337–340 (`latency_token = hidden_states[..., latency_token_position]`, `scheduler(latency_token, latency)`) |
| Vision tower and feature selection | `src/adallava/model/multimodal_encoder/prumerge_utils.py`: `vision_tower(...)`, `feature_select(...)`; `vision_tower.vision_model.encoder.layers` for ViT structure |

---

## 8. Exp1 profiling and logging

**Profiling (FLOPs) now runs correctly for Exp1:**

- **ada_analyzer:** `analyze_generate_task` uses `prompt_len` as LLM input sequence length (for Exp1 = text + K vision tokens); no fixed N. `gen_len` is guarded with `max(0, gen_len)` and `avg_flops` avoids division by zero when `gen_len <= 0`.
- **adallava_wrapper:** Single-GPU uses `device_map="auto"` so builder sets `cuda:0` (avoids tied-weights / vision tower on wrong device). After generate, `prompt_len` / `gen_len` / `vision_tokens_K` are taken from model outputs; generated text is decoded from the token-space slice using `prompt_len_tokens` and safe token IDs to avoid SentencePiece "piece id out of range". Results dict exposes `prompt_len`, `gen_len`, and `vision_tokens_K` for samples JSONL and sweep outputs.
- **cache_utils:** `from_legacy_cache` supports both `(key, value, drop_states)` and `(key, value)`; `get_execution_plan` handles execution_plan as dict or list so legacy caches without execution_plan yield full run.

**Detailed logging:**

- Wrapper logs `[Exp1] prompt_len=... gen_len=... vision_tokens_K=... seq_len=... resp_len=... resp_preview=...` per sample. With `ADALLAVA_DEBUG` or `ADALLAVA_DEBUG_FLOPS`, logs also include N (image placeholder count), K, and prefill_flops for sanity-check.
- `encode_images` (eval, adaptive) logs once per process: `[Exp1 encode_images] eval path: image_features.shape=... device=...`.

**Related debug docs:** `docs/EXP1_EVAL_LOW_ACCURACY_DEBUG.md` (low accuracy / empty resps, K≠N merge fix), `docs/EVALUATION_LAYER_SKIP_AND_FLOPS.md` (layer skip and FLOPs computation).
