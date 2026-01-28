#!/usr/bin/env python3
"""
Plot accuracy vs FLOPs (or vs token budget) for adaptive vision token scheduler (Exp1).

Input: CSV or JSON with columns/keys: token_budget (ratio in [0,1]), accuracy, and optionally flops.
  - token_budget: float, e.g. 0.25, 0.5, 0.75, 1.0
  - accuracy: float, e.g. task accuracy
  - flops: float (optional), total FLOPs; if missing, plot is accuracy vs token_budget only,
    or use --estimate-flops to apply a simple analytic estimate (vision_const + mm_proj*K + llm*seq_len).

Example CSV:
  token_budget,accuracy,flops
  0.25,0.68,1.2e12
  0.5,0.75,1.8e12
  0.75,0.78,2.4e12
  1.0,0.80,3.0e12

Usage:
  python scripts/plotting/plot_accuracy_flops.py --input results.csv --output accuracy_flops.pdf
  python scripts/plotting/plot_accuracy_flops.py --input results.csv --output acc_vs_budget.pdf  # no flops col
  python scripts/plotting/plot_accuracy_flops.py --input results.csv --estimate-flops --output accuracy_flops.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_table(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suf = p.suffix.lower()
    rows = []
    if suf == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        else:
            rows = [data]
    elif suf in (".csv", ".txt", ""):
        with open(p) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return []
        headers = [h.strip() for h in lines[0].split(",")]
        for ln in lines[1:]:
            vals = [v.strip() for v in ln.split(",")]
            rows.append(dict(zip(headers, vals)))
    else:
        raise ValueError(f"Unsupported format: {suf}")
    # normalize types
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def estimate_flops_from_budget(
    token_budget: float,
    num_patches: int = 576,
    vision_flops: float = 5e10,
    mm_proj_per_token: float = 2e6,
    llm_per_token: float = 1e8,
    avg_text_len: int = 512,
) -> float:
    """Rough FLOPs for one forward: vision (const) + mm_projector on (1+K) + LLM on (1+K)+text.
    Coefficients are placeholders; replace with your own counts if available.
    """
    k = max(1, int(round(token_budget * num_patches)))
    mm_flops = (1 + k) * mm_proj_per_token
    seq_len = (1 + k) + avg_text_len
    llm_flops = seq_len * llm_per_token
    return vision_flops + mm_flops + llm_flops


def main():
    ap = argparse.ArgumentParser(description="Plot accuracy vs FLOPs or vs token budget")
    ap.add_argument("--input", "-i", required=True, help="Input CSV or JSON path")
    ap.add_argument("--output", "-o", default="accuracy_flops.pdf", help="Output figure path")
    ap.add_argument(
        "--x",
        default="flops",
        choices=("flops", "token_budget"),
        help="X axis: flops or token_budget (default: flops if column exists, else token_budget)",
    )
    ap.add_argument(
        "--estimate-flops",
        action="store_true",
        help="If flops column missing, estimate from token_budget via a simple formula",
    )
    ap.add_argument("--title", default="Accuracy vs FLOPs", help="Plot title")
    ap.add_argument("--xlabel", default=None, help="X axis label (default from --x)")
    ap.add_argument("--ylabel", default="Accuracy", help="Y axis label")
    args = ap.parse_args()

    rows = load_table(args.input)
    if not rows:
        raise SystemExit("No rows in input")

    # Prefer token_budget then accuracy then flops
    def get(r, k, default=None):
        for key in (k, k.replace("_", " "), k.replace(" ", "_")):
            if key in r:
                return r[key]
        return default

    budgets = [get(r, "token_budget") for r in rows]
    accs = [get(r, "accuracy") for r in rows]
    flops = [get(r, "flops") for r in rows]

    if any(b is None for b in budgets) or any(a is None for a in accs):
        raise ValueError("Each row must have 'token_budget' and 'accuracy'")

    has_flops = all(f is not None for f in flops)
    if not has_flops and args.estimate_flops:
        flops = [estimate_flops_from_budget(b) for b in budgets]
        has_flops = True
    if args.x == "flops" and not has_flops:
        args.x = "token_budget"

    xlabel = args.xlabel or ("FLOPs" if args.x == "flops" else "Token budget (ratio)")
    xvals = flops if args.x == "flops" else budgets

    fig, ax = plt.subplots()
    ax.plot(xvals, accs, "o-", markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    if args.x == "token_budget":
        ax.set_xticks(budgets)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
