# Exp1 Token Budget Sweep: 低 Accuracy 诊断

## 现象

- `logs_token_budget_sweep_vqav2_val_exp1` 下各 token_budget 的 **exact_match 极低**（~0.08%）。
- 对比：AdaLLaVA latency sweep 下同一 VQAv2 的 exact_match 约 **68%**。

## 根本原因（已确认）

**不是设计或训练失败，而是评估时模型没有生成任何答案。**

在 `*_samples_vqav2_val.jsonl` 中可以看到：
- `"resps": [[""]]`、`"filtered_resps": [""]` —— **模型输出为空字符串**。
- 因此 exact_match 几乎为 0（只有极少数样本可能因 metric 聚合方式得到非零）。

## 可能原因（需逐项排查）

1. **生成长度为 0**
   - 模型在 generate 时 **没有生成任何新 token**（只返回了 prompt）。
   - 解码后 “回答” 部分为空 → resps 为 `[""]`。
   - **建议**：在 `generate()` 返回后或 wrapper 里打印：
     - `outputs.sequences.shape` 与 prompt 长度；
     - `outputs.sequences.shape[1] - prompt_len`（新 token 数）。
   - 若新 token 数恒为 0，说明问题在「为何不生成」而不是 metric。

2. **LLaVA merge 对 K≠N 的兼容性**
   - 评估时我们传入 **K 个 vision token**（`[B, K, C]`），序列长度为 K。
   - `prepare_inputs_labels_for_multimodal` 来自父类（llava），若内部假定 **固定 N 个 image token**（例如按 N 做切片、padding 或 position 计算），在 K≠N 时可能：
     - 得到错误的 `inputs_embeds` / `position_ids` / `attention_mask`；
     - 导致首 token 就生成 EOS 或行为异常 → 表现为 0 个新 token。
   - **建议**：用 **token_budget=1.0**（即 K=N）再跑一次 sweep 中一个点；若此时 resps 非空且 accuracy 正常，则多半是 **K≠N 时 merge/position 处理有 bug**，需要在本仓库 override `prepare_inputs_labels_for_multimodal`，按 `image_features.size(1)` 正确处理变长 image 序列。

3. **Exp1 配置 / checkpoint 与 eval 不一致**
   - 若加载的 checkpoint 或 config 里 **未** 将 `token_selecting` 设为 `'adaptive'`，则不会走 vision token controller，可能走错分支或得到非预期形状。
   - **建议**：在 `encode_images` 内（adaptive 分支）加一次 log：例如 `image_features.shape` 和 `self.training`；确认 eval 时确实得到 `[B, K, C]` 且为 eval 模式。

4. **token_budget 未正确传入**
   - wrapper 中通过 `getattr(self.model, "vision_token_controller", None)` 判断后设置 `self.model.current_token_budget = self.token_budget`。
   - 若此处未执行或 `self.token_budget` 被覆盖，eval 可能用了错误 budget（例如 0 或 None），导致 K=0 或异常。
   - **建议**：在设置 `current_token_budget` 后打印一次 `self.token_budget` 和（若可能）controller 内实际用的 K。

## 已加调试（可直接看日志）

- **wrapper**：每次 generate 后打一条 `[Exp1 debug] seq_len=... prompt_len=... gen_len=... resp_len=... resp_preview=...`。若 `gen_len=0` 或 `resp_len=0`，说明模型没生成新 token 或解码为空。
- **encode_images**：eval 且 adaptive 时打一次 `[Exp1 encode_images] eval path: image_features.shape=... device=...`，确认是 `[B, K, C]` 且 device 与主模型一致。
- **单卡加载**：当 `accelerator.num_processes == 1` 时，wrapper 会传 `device_map="auto"` 给 `load_pretrained_model`，由 builder 转为 `cuda:0`（与 demo 一致），避免 vision tower 被放到 cuda:1。

跑完 test 脚本后看 stdout 里上述两处日志即可判断问题在「0 新 token」还是「encode 形状/device」。

## 建议的排查顺序

1. **确认是否 0 新 token**  
   看日志里的 `[Exp1 debug]`：若 `gen_len=0` 且 `resp_len=0`，说明模型生成了 0 个新 token。

2. **用 token_budget=1.0 做对照**  
   只跑一个 batch 或少量样本，比较：
   - token_budget=1.0 时：resps 是否非空、exact_match 是否明显上升；
   - token_budget=0.5 时：是否仍为空。  
   若 1.0 正常而 0.5 仍空，则重点查 **K≠N 时的 merge/position/attention**。

3. **确认 eval 路径**  
   在 `encode_images`（adaptive 分支）里 log：
   - `not self.training` 为 True；
   - `image_features.shape` 为 `(B, K, C)` 且 K 符合当前 token_budget。

4. **必要时 override merge**  
   若确认是 LLaVA 父类在 K≠N 时行为错误，可在 `AdaLlavaLlamaForCausalLM` 中 override `prepare_inputs_labels_for_multimodal`，保证按 `image_features.size(1)` 插入 K 个 image token，并正确维护 `position_ids` 与 `attention_mask`。

## 已实施的修复（K≠N 时输出为空）

**原因**：评估时 `encode_images` 返回 `[B, K, C]`（K 为 token budget 下保留的 token 数），但 LLaVA 的 prompt 经 `tokenizer_image_token` 后含有 **N** 个 `IMAGE_TOKEN_INDEX` 占位符。父类 `prepare_inputs_labels_for_multimodal` 按「占位符数量 = image_features.size(1)」做 merge，因此期望 N 个特征；传入 K 个会导致错位或异常，进而出现 0 新 token / 空输出。

**修复**：在 `AdaLlavaLlamaForCausalLM` 中 override `prepare_inputs_labels_for_multimodal`（`src/adallava/model/ada_llava_llama.py`）：

- 当 `token_selecting == "adaptive"` 且非训练且 `K < N` 时：
  1. 用当前 `encode_images` 得到 K（`image_features.shape[1]`）。
  2. 在 `input_ids` 中统计连续 `IMAGE_TOKEN_INDEX` 数量得到 N。
  3. 将每个 batch 的序列缩短：只保留 **K** 个 image 占位符（去掉 N−K 个），即 `new_input_ids` 长度为 `L - (N - K)`，并同步缩短 `position_ids`、`attention_mask`、`labels`。
  4. 用缩短后的张量调用父类 `prepare_inputs_labels_for_multimodal`；父类再次调用 `encode_images` 得到 `[B, K, C]` 并 merge 到 K 个占位符，行为一致。

- 当 K≥N 或非 adaptive/训练时，直接调用父类，行为与之前一致。

修复后，用 `token_budget=0.5` 等 K<N 的设置跑 `scripts/eval/vqav2_token_budget_test_exp1.sh`，应能得到非空 resps 和正常 accuracy。

### 与 Demo 的差异：未传 `input_ids` 导致 `outputs.sequences` 只有生成部分

**现象**：即使做了上述 K≠N 缩短，日志里仍出现 `seq_len=2`、`prompt_len=353`、`resp_preview=''` 或 `'ing'`，即 `outputs.sequences` 只有 2 个 token，解码后几乎为空。

**原因**：Demo 调用 `model.generate(input_ids, images=..., ...)`，**第一个参数是 `input_ids`**。我们的 `generate()` 里在调用 `super().generate()` 时只传了 `inputs_embeds`，**没有传缩短后的 `input_ids`（即 `inputs`）**。HuggingFace 的 generate 需要 prompt 的 token ids 来拼出完整序列（prompt ids + 生成 ids）；缺少时可能只返回「生成部分」，导致 `outputs.sequences` 只有 1～2 个 token，解码后为空或片段。

**修复**：在 `ada_llava_llama.py` 的 `generate()` 中，调用 `super().generate()` 时**把缩短后的 `inputs` 作为第一个参数传入**：

```python
outputs = super().generate(
    inputs,   # 缩短后的 input_ids，用于构建完整序列
    position_ids=position_ids,
    attention_mask=attention_mask,
    inputs_embeds=inputs_embeds,
    ...
)
```

并令 `prompt_len = inputs.shape[1]`、`gen_len = outputs.sequences.shape[1] - prompt_len`，与 `outputs.sequences` 中 prompt 长度一致。

**补充**：父类 LLaVA 的 `prepare_inputs_labels_for_multimodal` 可能不返回第一个值（返回 `None`），导致 `generate()` 里 `inputs` 为 None、框架只产出「生成部分」、`gen_len` 为负。在 override 中调用父类后，**强制把第一个返回值改为缩短后的 `new_input_ids`** 再 return，这样 `generate()` 一定能拿到缩短后的 `input_ids` 用于拼出完整序列。

### 推理用的是 N 还是 K？Demo 怎么设的？

- **推理序列长度**：评估时用的是 **K**（token budget 下保留的 vision token 数）。流程是：  
  1) wrapper 传入的 prompt 经 `tokenizer_image_token` 后有 **N** 个 image 占位符；  
  2) 我们的 override 在 K&lt;N 时把序列缩短为 **K** 个 image 占位符（总长 `L - (N-K)`）；  
  3) `encode_images` 在 eval 时只 gather K 个 token → `[B, K, C]`；  
  4) 父类 merge 把 K 个特征填进这 K 个占位符；  
  5) 再插入 1 个 latency token，所以实际 forward 的序列长度 = 缩短后长度 + 1 = **K 对应的总长**（不是 N）。
- **Demo**（`scripts/demo_vision_token_prune.py`）：  
  - 同样调用 `model.generate(input_ids, images=..., image_sizes=..., latency=1.0, ...)`，且会设 `model.current_token_budget = args.token_budget`（如 0.5）。  
  - prompt 里只有 **1** 个 `DEFAULT_IMAGE_TOKEN`，经 `tokenizer_image_token` 后变成 **N** 个占位符；  
  - 同一套 `prepare_inputs_labels_for_multimodal` 在 K&lt;N 时缩短为 K 个占位符，所以 demo 推理也是用 **K** 序列长度，与 lmms_eval 一致。

### Demo 0.5 有输出、0.6 为空

可能原因：

1. **未传缩短后的 input_ids**：若未做「强制返回 new_input_ids」的修复，父类可能返回 `None`，`generate()` 只拿到「生成部分」→ 解码为空或片段。0.5 和 0.6 都可能空；若 0.5 有输出而 0.6 空，多半是下面 2 或 3。
2. **首 token 为 EOS**：0.6 时 K=345、序列更长，若模型在更长 prompt 下首 token 就出 EOS（校准/训练分布差异），解码会为空。可用 `--seed` 复现；或打印首生成 token id 确认是否为 EOS。
3. **insert_latency_token 的 position_ids**：已修复为按当前 batch 的 `new_attn_b` 构建 `cur_position_ids`（不再误用 `new_attention_mask[-1]`），避免 B>1 或某些长度下错位。

**后续修复（demo sweep 只有 0.5 有输出）**：

1. **K≥N 时也强制返回 input_ids**：当 token_budget=1.0（K=N）时原先直接 `return super().prepare_inputs_labels_for_multimodal(...)`，父类可能返回 `None` 作为第一项，导致 generate() 只拿到「生成部分」→ 1.0 为空。改为调用父类后强制 `return (input_ids, out[1], ...)`，与 K&lt;N 时一致。
2. **input_ids 与 inputs_embeds 长度对齐**：insert_latency_token 后 inputs_embeds 长度为 new_len+1，而 inputs 仍为 new_len。HF generate() 在两者长度不一致时可能只产出「生成部分」，导致除 0.5 外多数 budget 解码为空。在 generate() 中在 latency 插入位置用 pad_token_id（或 0）扩展 input_ids 至长度 new_len+1，使与 inputs_embeds 一致后再调用 super().generate()。

建议：用最新代码重跑 demo sweep；若仍有异常，可加 `--seed 42` 并打印首生成 token id 判断是否为 EOS。

## 结论

- **Accuracy 极低是因为模型在评估时没有生成任何答案（resps 为空），而不是 vision token 选择或训练目标本身设计错误。**
- 优先修复 **生成流程**（0 新 token 的原因：merge/position/config/token_budget），再重新跑 token_budget sweep 看 accuracy 与 FLOPs 曲线。
