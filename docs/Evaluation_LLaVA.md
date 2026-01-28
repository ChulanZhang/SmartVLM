# Evaluation

We use **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** as the main evaluation interface. Benchmarks, commands, and options are documented in the main [README](../README.md#evaluation).

**FLOPs and efficiency metrics** are computed by **[LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer)** inside the lmms-eval pipeline:

- The adallava wrapper (`adallava_wrapper.py`) builds an `AdaptiveAnalyzer`, which subclasses LLM-Viewer’s `ModelAnalyzer`.
- After each generation, it calls `analyze_generate_task(prompt_len, gen_len, num_heads, ...)`.
- That uses LLM-Viewer’s layer-wise roofline analysis to obtain prefill/decode OPs (FLOPs), inference time, and memory. These are passed into lmms-eval’s task outputs (e.g. `flops`, `avg_flops`, `prefill_flops`, `prefill_time`, `memory_consumption`).

So when you run lmms-eval with the `adallava` model, FLOPs are **not** from a separate tool—they come from LLM-Viewer via `AdaptiveAnalyzer` in this repo.

---

## Optional: LLaVA-style scripts (no FLOPs)

For custom jsonl datasets or submission-only workflows (e.g. VQAv2 testdev), you can still use:

- **[model_vqa_loader.py](../src/adallava/eval/model_vqa_loader.py)** – LLaVA-style jsonl → model inference → answers file.
- **scripts/eval/vqav2.sh** – multi-GPU wrapper around `model_vqa_loader` and the VQAv2 conversion script.

These do **not** run LLM-Viewer or lmms-eval, so they do **not** report FLOPs or other efficiency metrics. Use lmms-eval (see README) if you need FLOPs, time, or memory.

### VQAv2 with LLaVA-style script

1. Download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) and extract to `./data/eval`.
2. Download [test2015](http://images.cocodataset.org/zips/test2015.zip) and put it under `./data/eval/vqav2`.
3. Multi-GPU inference:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh
   ```
4. Submit the produced file to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission) (e.g. under `./data/eval/vqav2/answers_upload` or as pointed by the script).

For VQAv2 with FLOPs and metrics, use lmms-eval and the `vqav2_test` task as in the README.
