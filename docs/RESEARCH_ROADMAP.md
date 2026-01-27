# Research Roadmap: Unified Adaptive Inference for Vision-Language Models

## 1. Key Contributions

1. **Unified adaptive inference framework**  
   We propose a unified adaptive inference framework for vision-language models (VLMs) that simultaneously controls **three computational dimensions**: sequence length, model width, and model depth. Building on AdaLLaVA, our design introduces **token-level control** alongside attention head and layer gating, enabling end-to-end adaptive execution from the vision encoder to the LLM. This is the **first work** to jointly optimize token pruning, head masking, and layer skipping in a single dynamic controller.

2. **Coordinated two-level controller**  
   We introduce a **coordinated two-level controller**:
   - **Vision-side controller**: Adaptively selects and compresses visual tokens before they are passed to the LLM, reducing redundancy early and minimizing downstream compute.
   - **LLM-side controller**: Adjusts attention heads and transformer layers conditioned on both the input and the vision-side decisions.

   This coordinated scheduling enables joint, content-aware control across all three dimensions, achieving more fine-grained and globally efficient inference than prior modular or independent approaches.

3. **RL-based controller training**  
   We develop a **reinforcement learning (RL)–based controller training strategy** to replace the standard Gumbel-Softmax relaxation:
   - The RL-trained controller directly optimizes an **accuracy–FLOPs** trade-off reward.
   - It offers tighter budget adherence and greater flexibility in discrete decision making.
   - **[Target]** Experiments show that the RL strategy yields higher efficiency and more stable scheduling.

4. **Empirical results**  
   - Our full system achieves **state-of-the-art accuracy–efficiency trade-offs** across multiple vision-language benchmarks.
   - We conduct extensive evaluations on [xxx, yyy, zzz — TBD], showing that our method consistently outperforms recent **static** baselines (PruMerge, FastV) and **adaptive** baselines (AdaLLaVA).
   - The proposed approach produces **smooth Pareto curves** between accuracy and FLOPs, validating its practical value for real-world, budget-aware inference.

---

## 2. Motivation and Design

### 2.1 Motivation

While prior works like **AdaLLaVA+PruMerge** or **AdaLLaVA+FastV** combine adaptive execution and token pruning, they do so in a **disjoint** manner:

- Token reduction is performed **independently and statically** before the LLM.
- The downstream controller operates **without awareness** of these decisions.
- This modular control fails to capture **cross-dimensional dependencies**, often leading to suboptimal compute allocation and limited overall efficiency.

To address this, we propose a **two-stage coordinated controller** that jointly optimizes all three dimensions—sequence length, model width, and model depth—in a **unified, content-aware** manner:

- A lightweight **vision-side controller** inserted before the LLM selects informative visual tokens via a learned binary mask.
- It adopts the same **probabilistic, differentiable** design as AdaLLaVA, using **Gumbel-Softmax sampling** to maintain end-to-end trainability.
- Crucially, the **LLM-side controller is conditioned on the pruned token set**, allowing both controllers to coordinate and balance compute adaptively.

### 2.2 Design

- **Two-stage controller design**:
  - **Controller I (vision-side)**: Controls vision token pruning (sequence length).
  - **Controller II (LLM-side)**: Controls model width (head masking) and model depth (layer skipping).
- Joint training and coordination between the two controllers.

---

## 3. Key Experiments

All evaluations use standard vision-language benchmarks (e.g., VQAv2, visual reasoning tasks) with **FLOPs** and **accuracy** as primary metrics.

---

### 3.1 Experiment 1: Adaptive Vision Token Pruning vs. Static Token Reduction

**Goal:** Establish how the proposed adaptive method compares to existing token pruning/merging baselines in efficiency and accuracy.

**Methods compared:**

| Method | Description |
|--------|-------------|
| **LLaVA baseline** | Original LLaVA-1.5 with full visual tokens (no adaptive pruning). |
| **LLaVA-PruMerge** | Adaptive visual token merging; reduces image token count by ~14× with minimal performance loss. |
| **FastV** | Plug-and-play token pruning in early layers; ~45% FLOPs reduction in LLaVA-13B without sacrificing accuracy. |
| **Proposed adaptive controller** | Integrates AdaLLaVA-style adaptation to train a **token-level controller** that predicts useful tokens via learned probabilities, vs. PruMerge/FastV’s **training-free** token prune/merge strategy. |

**Metrics:** Total FLOPs used; task accuracy.

**Expected outcome:** The adaptive controller should match or outperform the efficiency–accuracy trade-off of PruMerge/FastV.

---

### 3.2 Experiment 2: Single vs. Multi-Dimensional Adaptation

**Goal:** Demonstrate the contribution of adapting each dimension (sequence length, model width, model depth) and the advantage of **combining all three**. Validates that controlling more axes yields superior efficiency–flexibility.

**Why this matters:**

- VLM consists of multiple components (vision encoder) and stages (prefill, decode).
- A single controller design is insufficient for such architecture.
- VLM is latency-sensitive (time to first token, time between tokens); we need a low-overhead design.
- AdaLLaVA uses LLM layers to fuse vision tokens, language tokens, and a latency token; ours extends this with **two-stage controller design** and joint training.

**Ablation variants:**

| Variant | Description |
|---------|-------------|
| **Depth-only** | Adaptive layer skipping (e.g., AdaLLaVA-L style). |
| **Width-only** | Adaptive model width (e.g., head/MLP masking, AdaLLaVA-H style). |
| **Sequence-only** | Adaptive visual token pruning; all layers and heads used. |
| **Depth+Width** | Controller adjusts both layers and heads (original AdaLLaVA two-dim control). |
| **All Three** | Proposed full model: tokens, heads, and layers adjusted simultaneously. |

**Metrics:** For each variant, accuracy vs. FLOPs at several budget levels.

**Expected findings:**

- Single-dimension adaptations show **limited flexibility** (e.g., depth-only may save compute but lose accuracy when skipping many layers, since all tokens/heads are still processed in remaining layers).
- The **full three-dimensional controller** should achieve the **best trade-off**: higher accuracy at low FLOPs than any single-dimension variant. Token reduction plus layer/head skipping enables finer-grained optimization.
- **Depth+Width (AdaLLaVA)** will be competitive but likely slightly less efficient than three-dim control at some budgets, demonstrating the value of the new token-length dimension.

---

### 3.3 Experiment 3: Controller Training — RL vs. Gumbel-Softmax Baseline

**Goal:** Compare the proposed **RL-based** controller training with the **Gumbel-Softmax relaxation** baseline (as in AdaLLaVA). Determine whether RL yields better decision policies under the same constraints.

**Setup:** Two versions of the full adaptive model (3-dim control, including vision token pruning):

| Version | Description |
|---------|-------------|
| **Differentiable controller (baseline)** | Gumbel-Softmax for soft masks; controller optimized via backprop with a latency/FLOPs penalty (mirrors AdaLLaVA’s original training). |
| **RL-trained controller (proposed)** | Policy gradient or PPO; reward encourages high QA accuracy minus a penalty for latency/FLOPs, or enforces a target budget. Masking decisions are discrete actions optimized via RL. |

**Metrics:**

- Overall accuracy; average FLOPs; budget adherence.
- Training convergence: iterations to stable policy; stability (RL typically has higher variance).

**Expected results:**

- RL-trained controller should perform **on par or better** than the differentiable baseline on accuracy–efficiency trade-off.
- Example: if Gumbel-Softmax reaches 90% of baseline accuracy at 50% FLOPs, RL might reach ~92% at 50% FLOPs by directly optimizing discrete decisions with an accuracy/FLOPs reward.
- AdaLLaVA already showed good budget adherence; our RL approach should match or improve accuracy at each budget.

**Analysis:**

- Compare **accuracy vs. budget** curves for both training methods.
- Discuss training: e.g., “The RL controller converged to a stable policy yielding X% accuracy at Y GFLOPs, whereas the differentiable controller peaked at Z% for the same budget.”
- If differences are small, that indicates the Gumbel-Softmax baseline is already strong; the experiment then justifies whether the added complexity of RL is worthwhile.

---

## 4. Summary

This roadmap defines a **unified adaptive VLM inference** approach that:

1. Jointly controls **sequence length, model width, and model depth** via a two-level (vision + LLM) controller.
2. Replaces or complements Gumbel-Softmax with **RL-based** controller training for tighter budgets and more flexible discrete control.
3. Is validated through **three core experiments**: adaptive token pruning vs. PruMerge/FastV, single vs. multi-dimensional adaptation, and RL vs. Gumbel-Softmax training.

Implementation and evaluation should follow this structure to keep the narrative and claims aligned with the roadmap.
