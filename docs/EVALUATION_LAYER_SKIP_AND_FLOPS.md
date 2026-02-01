# AdaLLaVA 评估：真实跳层与 FLOPS 计算

本文档详细说明：(1) AdaLLaVA 在 evaluation 时是**真实跳过层**还是用虚拟跳层模拟减少 FLOPS；(2) FLOPS/OPs 计算代码的位置与理论依据；(3) mm_projector 是否支持动态序列长度。

---

## 1. 评估时是真实跳层还是虚拟模拟？

### 1.1 结论：**真实跳层 + 真实减少每层计算**

AdaLLaVA 在 **evaluation（inference）** 时做的是**真实**的层跳过与 head 裁剪，而不是“虚拟模拟”：

- **整层跳过**：当某层的 `drop_states` 表示“所有 head 都丢弃”时，该层**直接不执行**（`continue`），不调用 `decoder_layer`，因此该层完全不产生计算与内存访问。
- **部分 head 执行**：当该层要执行但只保留部分 head 时，通过 **weight slicing** 只对保留的 head 做 Q/K/V 线性变换与后续 attention，计算量按保留的 head 数量比例减少。

因此，**推理阶段的计算和 FLOPS 是真实减少的**，不是事后用公式“模拟”出来的。

### 1.2 代码依据

**整层跳过**（`src/adallava/model/language_model/ada_llama/modeling_ada_llama.py`）：

```python
# AdaLlamaModel.forward，decoder layers 循环内（约 352–356 行）
if not self.training and drop_states is not None and torch.all(drop_states == 0):
    continue   # 该层完全不执行

if not self.training and drop_states is not None and torch.all(drop_states == 1):
    drop_states = None   # 全部保留，等价于无 drop

# 只有未 continue 的层才会执行：
layer_outputs = decoder_layer(
    hidden_states,
    ...
    drop_states=drop_states,
    ...
)
```

**每层内 head 裁剪（inference 为真实 weight slicing，非 mask）**（同文件，`AdaLlamaSdpaAttention.forward` 与 `AdaLlamaMLP.forward`）：

- **Attention**：当 `drop_states is not None` 且 **`not self.training`** 时，对 `q_proj`/`k_proj`/`v_proj`/`o_proj` 均使用 `split_weight(..., drop_states[0,0], self.head_dim, ...)` 得到**物理上更小的权重矩阵**，再 `F.linear(hidden_states, query_weight)` 等，因此**实际参与计算的 head 数 = 保留的 head 数**，FLOPS 真实减少。
- **MLP**：`AdaLlamaMLP` 在 **`not self.training`** 时对 `gate_proj`/`up_proj`/`down_proj` 同样使用 `split_weight` + `F.linear`，中间维按 `drop_states` 切片，FLOPS 真实减少。
- **Mask 仅用于训练**：只有在 **`self.training`** 时，attention 的 `o_proj` 和 MLP 的中间激活才用 `* drop_states` / `* expand` 做 mask（先全量计算再乘 0/1），以便梯度能回传。**评估时不会走 mask 分支**，因此“跳 heads 是假跳过”的说法对 **evaluation** 不成立。

---

## 2. FLOPS/OPs 计算代码位置与流程

### 2.1 计算发生在哪里？

FLOPS 不是通过“再跑一遍带 hook 的模型”量出来的，而是用 **解析公式（analytic）** 算出来的，但公式里用到的 **每层执行情况** 来自**本次推理的真实 execution_plan**。因此：**推理 = 真实跳层/减 head；FLOPS = 用同一 run 的 execution_plan 代入公式求和**。

主要代码位置：

| 功能 | 文件 | 说明 |
|------|------|------|
| 单层、给定 head 数的 OPs 公式 | `src/adallava/eval/ada_analyzer.py` | `AdaptiveAnalyzer.analyze_one_layer(prompt_len, num_heads=..., ...)` |
| 按 execution_plan 对每层汇总 | 同上 | `analyze_all_layers(..., num_heads=execution_plan)`，对每层调用 `analyze_one_layer` 再按 stage 相加 |
| 整段生成任务的 FLOPS | 同上 | `analyze_generate_task(prompt_len, gen_len, num_heads=...)`：prefill 一次 + 每步 decode 一次，累加 OPs |
| 底层 OPs→时间/带宽 | `src/adallava/eval/LLM-Viewer/model_analyzer.py` | `_analyze_to_results(..., OPs=...)`；`roofline_model.roofline_analyze` 用 OPs 与 memory_access 算 bound 与 performance |

也就是说：**FLOPS 的计算逻辑在 `ada_analyzer.py`（以及继承/调用的 LLM-Viewer 的 `model_analyzer`）**，而 **execution_plan 来自本次 generate 的 cache**。

### 2.2 execution_plan 如何进入 FLOPS

- **generate 时**（`src/adallava/model/ada_llava_llama.py`）：  
  `return_dict_in_generate=True` 时，从 `outputs.past_key_values` 中解析出 execution_plan，并做标量化和 -1 标记整层跳过：

```python
execution_plan = DynamicCacheWithExecutionPlan.get_execution_plan_from_legacy_cache(
    outputs.past_key_values, self.config.num_hidden_layers
)
execution_plan = [_.sum().item() // 2 for _ in execution_plan]   # 每层“有效 head 数”的标量
execution_plan = [-1 if _ == 0 else _ for _ in execution_plan]  # 整层跳过记为 -1
setattr(outputs, 'execution_plan', execution_plan)
```

- **评估脚本**（如 `src/adallava/eval/adallava_wrapper.py`、`run_ada_llava.py`）：  
  取 `outputs.execution_plan` 作为 `num_heads` 传给 `analyzer.analyze_generate_task(prompt_len, gen_len, num_heads=num_heads)`。

- **analyze_all_layers**（`ada_analyzer.py`）：  
  把 `num_heads` 当作**每层一个标量**的列表（长度 = 层数），对每一层调用 `analyze_one_layer(prompt_len, curr_num_heads, ...)`（仅当 `curr_num_heads >= 0` 时调用，完全跳过的层记为 -1 不参与），再把各层的 decode/prefill 结果按 `ALL_DATA_NAMES`（含 `"OPs"`）相加，得到整机的 prefill/decode OPs。因此整层跳过时该层 OPs 贡献为 0。

因此：**FLOPS 的“输入”就是这一轮推理真实产生的 execution_plan**，不是虚拟的固定配置。

### 2.3 单层 OPs 公式（理论依据）

`analyze_one_layer` 对每一类算子用**标准 FLOP/OP 公式**（与 LLM-Viewer 的 Llama 配置一致）：

- **Linear 层**（q_proj, k_proj, v_proj, out_proj, gate_proj, up_proj, down_proj）：  
  - decode：`OPs = ic * oc * batchsize * 2 * (num_heads // num_attention_heads)`（乘加各算 1 OP，seq_len=1）。  
  - prefill：同上但乘 `prompt_len`。  
  - 对 K/V 的 proj，`num_heads` 会按 key_value_heads 比例换算，体现“只算保留的 head”的 FLOPS。

- **Attention 的 QK/SV matmul 与 softmax**：  
  - 所有项都带有 `num_heads`（或与 num_heads 成比例），例如 decode 时  
    - `qk_matmul_OPs = prompt_len * head_size * num_heads * batchsize * 2`  
    - `sv_matmul_OPs = 1 * head_size * prompt_len * num_heads * batchsize * 2`  
  - 因此 **head 数减少 → OPs 成比例减少**。

- **Norm、add、mlp_act**：  
  - 若该层被整层跳过，不会进入 `analyze_one_layer` 的“该层”分支（因为按层循环时，该层对应的 `num_heads` 为 -1 或 0，在汇总时会被处理成该层 OPs=0 或等价逻辑）。  
  - 若层执行，则 norm/add/mlp 的 OPs 也按 `num_heads // num_attention_heads` 比例缩放，与“只算保留 head 对应的通道”一致。

**理论依据**：  
- Transformer 单层 FLOPS 与 (序列长度 × hidden_size × 参数量/维度) 以及 **实际参与的 head 数** 成正比。  
- AdaLLaVA 的 execution_plan 记录了每层“实际参与的 head 数”（或整层跳过），因此用该 plan 代入上述公式，得到的是**与真实前向一致**的 FLOPS 估计。  
- 时间估计则再通过 `roofline_model.roofline_analyze(bandwidth, max_OPS, OPs, memory_access)` 由 OPs 和内存访问量得到 arithmetic intensity 与 bound（memory/compute），进而得到 inference_time = OPs / performance。

---

## 3. 小结：真实跳层 vs “虚拟”FLOPS

| 问题 | 答案 |
|------|------|
| 评估时是否真的跳过层？ | **是**。当某层 `drop_states` 全为 0 时，该层 `continue`，不执行。 |
| 是否真的减少每层计算？ | **是**。未跳过的层通过 weight slicing 只算保留的 head，FLOPS 与保留 head 数成正比。 |
| FLOPS 是“虚拟模拟”吗？ | **不是**。FLOPS 是用**解析公式**算的，但公式的输入是**本次推理的真实 execution_plan**，所以是“真实执行 + 按真实执行算 FLOPS”。 |
| 计算 FLOPS 的代码在哪？ | **核心在 `src/adallava/eval/ada_analyzer.py`**（`analyze_one_layer` / `analyze_all_layers` / `analyze_generate_task`），底层 OPs→时间在 `LLM-Viewer/model_analyzer.py` 与 `roofline_model.py`。 |

---

## 4. mm_projector 是否支持动态序列长度？

### 4.1 结论：**结构上支持动态序列长度**

mm_projector 在 LLaVA 中通常是 **MLP（如两段 Linear + 激活）**，对输入形状的要求是：

- 输入：`[B, S, C_in]`（batch, 序列长度, 视觉特征维度）
- 输出：`[B, S, C_out]`（batch, 同一序列长度, LLM 隐藏维度）

线性层和 MLP 是在**最后一维（特征维）**上做仿射变换，**序列维 S 只是 batch 维的延伸**（即把 `(B, S, C)` 视为 `(B*S, C)` 做线性再还原），因此 **S 可以是任意正整数**，不要求固定长度。也就是说，**从结构上看，mm_projector 可以接受动态序列长度**（例如有时 N 个 patch，有时 K 个 patch）。

### 4.2 与“固定长度”说法的关系

- **训练时**：LLaVA 通常用固定的 patch 数 N（如 576），所以**训练**时 mm_projector 看到的序列长度是固定的 N。  
- **结构/实现**：PyTorch 的 `nn.Linear`、`nn.Sequential(Linear, GELU, Linear)` 等都没有对序列维做约束，因此**一旦改成只传入 K 个 token（例如 `[B, K, C]`），mm_projector 不需要改结构就能工作**。

所以：**“mm_projector 是 MLP、接受固定长度输入”指的是“训练时我们习惯用固定 N”**，而不是“MLP 在数学或实现上只支持固定 N”。**从模块接口和数学上看，mm_projector 可以接受动态序列长度。**

### 4.3 当前 AdaLLaVA 中与 Exp1 的实际情况

- **LLM 侧（层/head 自适应）**：不涉及 mm_projector 的序列长度，vision 分支仍输出固定形状（如 `[B, N, C]`）再进 mm_projector。  
- **Exp1 视觉 token 裁剪**：  
  - **训练**：controller 输出 **`[B, N, C]`（未选位置填 0）**，mm_projector 和 LLM 看到 N，便于 backward 与 straight-through。  
  - **评估**：与 AdaLLaVA 跳层一致，**评估与训练行为区分**：在 `encode_images` 中当 `not self.training` 时，只 **gather K 个选中 token** 得到 **`[B, K, C]`** 再送入 mm_projector，mm_projector 和 LLM 看到 K，**FLOPs 真实减少**。LLaVA 的 merge 使用 `image_features.size(1)` 决定插入的 image token 数，因此无需改 `prepare_inputs_labels_for_multimodal`。

---

## 5. Vision token selection scheduler 的 FLOPS 与评估行为

### 5.1 评估时真实保留 K 个 token（已实现）

与 AdaLLaVA 的“评估/训练区分”一致，Exp1 在 **evaluation** 时不再传 `[B, N, C]`（零填充），而是 **只保留 K 个选中 token**：

- 在 `encode_images`（`token_selecting == "adaptive"`）中，当 **`not self.training`** 时，用 `keep_mask` 从 `patch_tokens` 上 **gather** 出 **`[B, K, C]`**，再送入 mm_projector；mm_projector 和 LLM 看到的序列长度为 **K**，**FLOPs 真实减少**。
- **训练**时仍传 **`[B, N, C]`**（未选位置填 0），不改变序列长度，便于梯度与 straight-through。

因此运行 `scripts/eval/vqav2_token_budget_sweep_exp1.sh` 时，前向已按 K 计算，测得的 FLOPs/延迟会随 token_budget 真实下降。

### 5.2 报告/画图时的 FLOPS 公式

- **评估前向**：已按 K 个 token 计算，FLOPs 与 K 一致。  
- **汇总/画图**：`scripts/plotting/plot_accuracy_flops.py` 的 `estimate_flops_from_budget(token_budget, num_patches=576, ...)` 按 **K = round(token_budget × num_patches)** 算 mm_projector 与 LLM 的 FLOPs（`mm_flops = (1+k)*mm_proj_per_token`，`seq_len = (1+k)+avg_text_len`），与真实前向一致。

### 5.3 与 LLM 侧跳层/跳 head 的对比

| 部分 | 实际前向是否省 FLOPS？ | 报告 FLOPS 建议 |
|------|------------------------|------------------|
| LLM 跳层 | **是**：整层 `continue`，不执行 | 用 execution_plan 按真实执行算 OPs（已实现） |
| LLM 跳 head | **是**：inference 用 weight slicing，真实减少 matmul 规模 | 同上，按每层保留 head 数算 OPs（已实现） |
| Vision token 选 K | **是**（评估）：传 `[B, K, C]`，mm_projector 与 LLM 按 K 计算 | 按 K 算；plot 脚本与真实前向一致 |

---

## 6. 相关文件索引

- 层跳过与 drop_states：`src/adallava/model/language_model/ada_llama/modeling_ada_llama.py`（`AdaLlamaModel.forward`，`AdaLlamaSdpaAttention.forward`，`AdaLlamaMLP.forward`）
- execution_plan 的生成与传递：`src/adallava/model/ada_llava_llama.py`（`generate`），`src/adallava/model/language_model/ada_llama/cache_utils.py`（`DynamicCacheWithExecutionPlan`）
- FLOPS/OPs 计算：`src/adallava/eval/ada_analyzer.py`（`AdaptiveAnalyzer`），`src/adallava/eval/LLM-Viewer/model_analyzer.py`，`src/adallava/eval/LLM-Viewer/roofline_model.py`
- 评估入口使用 execution_plan：`src/adallava/eval/adallava_wrapper.py`，`src/adallava/eval/run_ada_llava.py`，`src/adallava/eval/model_vqa_loader.py`
- 视觉 token 与 mm_projector：`src/adallava/model/ada_llava_llama.py`（`encode_images`），`docs/EXP1_ADAPTIVE_VISION_TOKEN_PRUNE_IMPLEMENTATION.md`（§3.6 固定形状与真实 FLOPS）
- Vision token 按 K 估计 FLOPS：`scripts/plotting/plot_accuracy_flops.py`（`estimate_flops_from_budget`）
