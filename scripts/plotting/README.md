# Plotting scripts (Exp1: accuracy–FLOPs curves)

## `plot_accuracy_flops.py`

Plots **accuracy vs FLOPs** (or vs token budget) for the adaptive vision token scheduler.

### Input format

CSV or JSON with at least:

- `token_budget`: ratio in [0, 1] (e.g. 0.25, 0.5, 0.75, 1.0)
- `accuracy`: task accuracy
- `flops` (optional): total FLOPs per sample; if missing, use `--estimate-flops` or plot vs token_budget only

Example CSV:

```text
token_budget,accuracy,flops
0.25,0.68,1.2e12
0.5,0.75,1.8e12
0.75,0.78,2.4e12
1.0,0.80,3.0e12
```

### How to produce the table

1. **Run evaluation at several token budgets**  
   Use `scripts/eval_vision_token_scheduler.sh` (or `model_vqa_loader --token_budget …`) so that results are written under e.g. `data/eval/vqav2/answers/.../token_budget_0.25/`, `token_budget_0.5/`, etc.

2. **Compute accuracy per budget**  
   - For VQAv2: submit each budget’s answers to the official server, or use a local accuracy script if you have one.  
   - For other benchmarks: use the task’s metric on the saved answers.

3. **Get FLOPs**  
   - **Option A**: Run lmms-eval with the adaptive model and `token_budget` in `model_args`; lmms-eval logs FLOPs. Collect (token_budget, accuracy, flops) from those runs.  
   - **Option B**: Use an analytic formula (vision_const + mm_projector(∝ K) + LLM(∝ seq_len)) and fill the `flops` column from `token_budget` and your constants. The script’s `--estimate-flops` uses a built-in placeholder formula; replace the constants in the script or in your own aggregation script for real numbers.

4. **Save a CSV** with columns `token_budget,accuracy,flops` (or `token_budget,accuracy` and pass `--estimate-flops` or `--x token_budget`).

### Usage

```bash
# Plot accuracy vs FLOPs (requires flops column or --estimate-flops)
python scripts/plotting/plot_accuracy_flops.py --input results.csv --output accuracy_flops.pdf

# Plot accuracy vs token budget when no FLOPs are available
python scripts/plotting/plot_accuracy_flops.py --input results.csv --x token_budget --output acc_vs_budget.pdf

# Use built-in FLOPs estimate from token_budget
python scripts/plotting/plot_accuracy_flops.py --input results.csv --estimate-flops --output accuracy_flops.pdf
```

There is **no existing script in the repo** that aggregates accuracy from the vision-token-scheduler eval outputs into this CSV; that aggregation is task-specific (e.g. VQAv2 local accuracy script or submission + server metrics). This directory provides the plotting step only.
