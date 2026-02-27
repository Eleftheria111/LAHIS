"""
LAHIS Full Pipeline for Llama-2-7B + TED Data.

Runs all steps in order:
  1. Data prep     — convert TED JSONL to per-language JSON files
  2. LAHIS matrices — compute head importance per language (GPU-intensive)
  3. Head selection — extract language-specific + language-general heads
  4. Heatmaps      — visualise importance matrices (Fig 1)
  5. Specificity   — dark-diagonal PPL experiment (Fig 2 / §3.4)
  6. Interventions — cross-lingual steering demos (§3.5)

Run all steps:
    python3 run_pipeline.py --model llama2 -b --all

Run individual steps (useful for resuming after a crash):
    python3 run_pipeline.py --model llama2 -b --step data
    python3 run_pipeline.py --model llama2 -b --step matrix --lan en
    python3 run_pipeline.py --model llama2 -b --step select
    python3 run_pipeline.py --model llama2 -b --step heatmap
    python3 run_pipeline.py --model llama2 -b --step specificity
    python3 run_pipeline.py --model llama2 -b --step intervene

All outputs land in:
    LAHIS/data/ted/          — per-language text files
    LAHIS/results/{model}/   — importance matrices, head indices, plots
"""

import argparse
import json
import os
import sys
import torch

MODEL_CHOICES = ["llama2", "aya", "mistral", "llama3"]
STEP_CHOICES  = ["data", "matrix", "select", "heatmap", "specificity", "intervene"]

# Default language set (covers paper's target languages + good TED coverage)
DEFAULT_LANGUAGES = ["en", "fr", "es", "zh", "ru", "de", "ar", "ja", "ko", "vi"]


# ---------------------------------------------------------------------------
# Step 1 — Data preparation
# ---------------------------------------------------------------------------

def step_data(args):
    print("\n" + "="*60)
    print("STEP 1: Building per-language TED files")
    print("="*60)

    import ted_loader as TED
    ted_path = args.ted_path or "../../TED2025/multi_way.jsonl"
    print(f"  Source : {ted_path}")
    print(f"  Dest   : {args.data_dir}")
    print(f"  Langs  : {args.languages}")
    print(f"  join_n : {args.join_n}")

    records = TED.load_ted_jsonl(ted_path)
    print(f"  Loaded {len(records):,} TED records")
    streams = TED.build_monolingual_streams(records, args.languages, join_n_sentences=args.join_n)
    paths = TED.save_monolingual_json(streams, args.data_dir)

    missing = [l for l in args.languages if l not in streams]
    if missing:
        print(f"\n  WARNING: No TED data found for: {missing}")
        print("  These languages will be skipped in later steps.")
    return paths


# ---------------------------------------------------------------------------
# Step 2 — Compute LAHIS importance matrices
# ---------------------------------------------------------------------------

def step_matrix(args, model=None, tokenizer=None):
    print("\n" + "="*60)
    print("STEP 2: Computing LAHIS importance matrices")
    print("="*60)

    from attn_matrix_ted import get_attn_head_matrix_ted
    import model_handler

    if model is None:
        device = _get_device(args)
        model, tokenizer = model_handler.load_model(
            args.model, device, args.half_precision, args.local
        )

    # Determine which languages have TED data
    available = [
        l for l in args.languages
        if os.path.exists(os.path.join(args.data_dir, f"ted_{l}.json"))
    ]
    if not available:
        print("  No TED data found. Run --step data first.")
        return None, None

    langs = [args.lan] if args.lan else available

    results_dir = f"../results/{args.model}"
    os.makedirs(results_dir, exist_ok=True)

    for lan in langs:
        out_path = os.path.join(results_dir, f"{args.model}_{lan}.pth")
        if os.path.exists(out_path) and not args.force:
            print(f"  [{lan}] already exists (use --force to recompute), skipping.")
            continue
        print(f"\n  Computing importance matrix for [{lan}] ...")
        get_attn_head_matrix_ted(
            model, tokenizer,
            lan=lan,
            model_name=args.model,
            data_dir=args.data_dir,
            data_num=args.data_num,
            max_length=args.max_length,
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 3 — Head selection (language-specific + language-general)
# ---------------------------------------------------------------------------

def step_select(args):
    print("\n" + "="*60)
    print("STEP 3: Identifying language-specific and language-general heads")
    print("="*60)

    results_dir = f"../results/{args.model}"
    from collections import Counter

    # Find languages with saved matrices
    available = [
        l for l in args.languages
        if os.path.exists(os.path.join(results_dir, f"{args.model}_{l}.pth"))
    ]
    if not available:
        print("  No importance matrices found. Run --step matrix first.")
        return

    print(f"  Languages with matrices: {available}")
    q = 1.0 - args.p          # e.g. p=0.02 → q=0.98 → top 2%
    min_rep = max(2, len(available) - 1)  # head must appear in most languages

    sorted_head_list = []
    for lan in available:
        matrix = torch.load(
            os.path.join(results_dir, f"{args.model}_{lan}.pth"),
            map_location="cpu",
        ).float()
        flat = matrix.view(-1)
        threshold = torch.quantile(flat, q=q)
        topk_idx = (flat > threshold).nonzero(as_tuple=False).squeeze()
        if topk_idx.dim() == 0:
            topk_idx = topk_idx.unsqueeze(0)
        vals = flat[topk_idx]
        order = torch.argsort(vals, descending=True)
        sorted_head_list.append(topk_idx[order])

    # Language-general heads: appear in top-p% for >= min_rep languages
    all_indices = torch.cat(sorted_head_list).tolist()
    index_counts = Counter(all_indices)
    general_heads = [
        idx for idx, cnt in sorted(index_counts.items(), key=lambda x: -x[1])
        if cnt >= min_rep
    ]
    print(f"\n  Language-general heads (appear in {min_rep}+ languages): {len(general_heads)}")

    # Language-specific heads: top-p% for that language, minus general heads
    specific_heads = {}
    for lan, topk in zip(available, sorted_head_list):
        filtered = [idx for idx in topk.tolist() if idx not in set(general_heads)]
        specific_heads[lan] = filtered

    # Save
    with open(os.path.join(results_dir, "head_indices.json"), "w") as f:
        json.dump(specific_heads, f, indent=2)
    with open(os.path.join(results_dir, "repeated_indices.json"), "w") as f:
        json.dump(general_heads, f, indent=2)

    print("\n  Language-specific head counts (top-specific):")
    for lan, idxs in specific_heads.items():
        print(f"    {lan:8s}: {len(idxs):3d} heads  (top 5: {idxs[:5]})")
    print(f"\n  General head indices (first 10): {general_heads[:10]}")
    print(f"  Saved -> {results_dir}/head_indices.json")
    print(f"  Saved -> {results_dir}/repeated_indices.json")


# ---------------------------------------------------------------------------
# Step 4 — Heatmaps
# ---------------------------------------------------------------------------

def step_heatmap(args):
    print("\n" + "="*60)
    print("STEP 4: Generating heatmaps")
    print("="*60)

    from heatmap_viz import plot_all_languages, plot_top_head_distribution

    results_dir = f"../results/{args.model}"
    output_dir  = os.path.join(results_dir, "heatmaps")

    available = [
        l for l in args.languages
        if os.path.exists(os.path.join(results_dir, f"{args.model}_{l}.pth"))
    ]
    if not available:
        print("  No matrices found. Run --step matrix first.")
        return

    plot_all_languages(
        model_name=args.model,
        languages=available,
        results_dir=results_dir,
        output_path=os.path.join(output_dir, "all_languages.png"),
        topk_p=args.p,
    )
    plot_top_head_distribution(
        model_name=args.model,
        languages=available,
        results_dir=results_dir,
        output_path=os.path.join(output_dir, "head_distribution.png"),
        topk_p=args.p,
    )


# ---------------------------------------------------------------------------
# Step 5 — Specificity (dark diagonal)
# ---------------------------------------------------------------------------

def step_specificity(args, model=None, tokenizer=None):
    print("\n" + "="*60)
    print("STEP 5: Specificity evaluation (dark diagonal)")
    print("="*60)

    from specificity_eval import run_specificity_experiment, plot_dark_diagonal
    import model_handler

    if model is None:
        device = _get_device(args)
        model, tokenizer = model_handler.load_model(
            args.model, device, args.half_precision, args.local
        )

    results_dir = f"../results/{args.model}"
    available = [
        l for l in args.languages
        if os.path.exists(os.path.join(results_dir, f"{args.model}_{l}.pth"))
        and os.path.exists(os.path.join(args.data_dir, f"ted_{l}.json"))
    ]
    if not available:
        print("  Insufficient data. Run --step matrix and --step data first.")
        return model, tokenizer

    results = run_specificity_experiment(
        model, tokenizer,
        languages=available,
        model_name=args.model,
        results_dir=results_dir,
        data_dir=args.data_dir,
        data_num=args.spec_data_num,
        p=args.p,
        test_type="full",
    )
    plot_dark_diagonal(
        results, available, args.model,
        output_path=os.path.join(results_dir, "heatmaps", "dark_diagonal.png"),
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 6 — Interventions
# ---------------------------------------------------------------------------

def step_intervene(args, model=None, tokenizer=None):
    print("\n" + "="*60)
    print("STEP 6: Cross-lingual intervention demos")
    print("="*60)

    from intervention_demo import run_occupation_steering, run_english_suppression
    import model_handler

    if model is None:
        device = _get_device(args)
        model, tokenizer = model_handler.load_model(
            args.model, device, args.half_precision, args.local
        )

    results_dir = f"../results/{args.model}"
    num_layers  = model.config.num_hidden_layers
    num_heads   = model.config.num_attention_heads
    n_heads     = max(1, int(num_layers * num_heads * args.p))

    head_idx_path = os.path.join(results_dir, "head_indices.json")
    if not os.path.exists(head_idx_path):
        print("  head_indices.json not found. Run --step select first.")
        return model, tokenizer

    # A) Occupation steering (lan1 vs lan2)
    try:
        run_occupation_steering(
            model, tokenizer,
            lan1=args.lan1, lan2=args.lan2,
            results_dir=results_dir,
            n_heads=n_heads,
            n_pairs=20,
        )
    except Exception as e:
        print(f"  Occupation steering skipped: {e}")

    # B) English suppression on a non-English language
    if "en" in json.load(open(head_idx_path)):
        ted_path = args.ted_path or "../../TED2025/multi_way.jsonl"
        try:
            run_english_suppression(
                model, tokenizer,
                test_lang=args.lan1,
                ted_path=ted_path,
                results_dir=results_dir,
                n_heads=n_heads,
                n_demos=10,
            )
        except Exception as e:
            print(f"  English suppression skipped: {e}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_device(args):
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LAHIS full pipeline for Llama-2-7B + TED data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama2", choices=MODEL_CHOICES)
    parser.add_argument("--device", default=None)
    parser.add_argument("-b", "--half-precision", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False,
                        help="Load model from local directory (default: download from HuggingFace)")

    # Steps
    parser.add_argument("--all", action="store_true",
                        help="Run all pipeline steps in sequence")
    parser.add_argument("--step", choices=STEP_CHOICES,
                        help="Run a single step")
    parser.add_argument("--lan",  default=None,
                        help="Single language for --step matrix (default: all in --languages)")

    # Data
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--ted-path", default=None,
                        help="Path to TED2025/multi_way.jsonl")
    parser.add_argument("--data-dir", default="../data/ted")
    parser.add_argument("--join-n", type=int, default=5,
                        help="Sentences to join per TED chunk")

    # LAHIS
    parser.add_argument("--data-num", type=int, default=1000,
                        help="TED samples per language for importance matrices")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--p", type=float, default=0.02,
                        help="Top-p fraction of heads = language-specific threshold")

    # Specificity
    parser.add_argument("--spec-data-num", type=int, default=300,
                        help="Samples per language for specificity PPL evaluation")

    # Intervention
    parser.add_argument("--lan1", default="zh",
                        help="First language for interventions")
    parser.add_argument("--lan2", default="fr",
                        help="Second language for interventions")

    # Misc
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if outputs already exist")

    args = parser.parse_args()

    if not args.all and not args.step:
        parser.print_help()
        print("\nProvide --all to run the full pipeline, or --step <name> for a single step.")
        sys.exit(1)

    model = tokenizer = None

    steps = STEP_CHOICES if args.all else [args.step]

    for step in steps:
        if step == "data":
            step_data(args)
        elif step == "matrix":
            model, tokenizer = step_matrix(args, model, tokenizer)
        elif step == "select":
            step_select(args)
        elif step == "heatmap":
            step_heatmap(args)
        elif step == "specificity":
            model, tokenizer = step_specificity(args, model, tokenizer)
        elif step == "intervene":
            model, tokenizer = step_intervene(args, model, tokenizer)

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"Outputs in: ../results/{args.model}/")
    print("="*60)


if __name__ == "__main__":
    main()
