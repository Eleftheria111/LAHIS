"""
LAHIS Specificity Evaluation — the "dark diagonal" experiment (paper §3.4, Fig 2).

For each pair (test_lang, head_lang):
  1. Load the saved importance matrix for head_lang.
  2. Zero out the top-p% heads (language-specific heads of head_lang).
  3. Compute average perplexity on test_lang TED text.

The expected result: disabling lang_i's heads hurts lang_i the most
→ the diagonal of the (languages × languages) matrix is the darkest.

Produces:
  - A (languages × languages) heatmap of perplexity values
  - The raw numbers printed / saved as JSON

Usage (from LAHIS/src/):
    python3 specificity_eval.py --model llama2 -b \\
        --languages en fr es zh ru de --data-num 500
"""

import argparse
import json
import os
import torch
import datasets
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model_handler

MODEL_CHOICES = ["llama2", "aya", "mistral", "llama3"]


def get_top_head_mask(
    model_name: str,
    head_lan: str,
    results_dir: str,
    p: float,
    test_type: str = "specific",
    language_heads_json: str = None,
    repeated_heads_json: str = None,
) -> torch.Tensor:
    """
    Build a [num_layers, num_heads] binary mask (1 = keep, 0 = disable).

    test_type:
      "specific" : zero only language-specific heads (from head_indices.json)
      "full"     : zero top-p% from raw importance matrix (includes general heads)
      "random"   : zero a random p% of heads (control)
      "ori"      : no masking (all ones)
    """
    if language_heads_json is None:
        language_heads_json = os.path.join(results_dir, "head_indices.json")
    if repeated_heads_json is None:
        repeated_heads_json = os.path.join(results_dir, "repeated_indices.json")

    # Load any model to get shape — or just read from saved matrix
    matrix_path = os.path.join(results_dir, f"{model_name}_{head_lan}.pth")
    matrix = torch.load(matrix_path, map_location="cpu").float()
    num_layers, num_heads = matrix.shape
    n = max(1, int(num_layers * num_heads * p))

    head_mask = torch.ones(num_layers, num_heads)

    if test_type == "ori":
        pass
    elif test_type == "random":
        rand_idx = torch.randperm(head_mask.numel())[:n]
        head_mask.view(-1)[rand_idx] = 0.0
    elif test_type == "full":
        _, topk_idx = torch.topk(matrix.view(-1), k=n)
        head_mask.view(-1)[topk_idx] = 0.0
    elif test_type == "specific":
        if not os.path.exists(language_heads_json):
            raise FileNotFoundError(
                f"head_indices.json not found: {language_heads_json}\n"
                "Run language_heads.py first (or use --test-type full)."
            )
        with open(language_heads_json) as f:
            head_indices = json.load(f)
        indices = head_indices.get(head_lan, [])[:n]
        if not indices:
            print(f"  WARNING: no specific heads for '{head_lan}' in {language_heads_json}")
        head_mask.view(-1)[indices] = 0.0
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    return head_mask


def compute_ppl(
    model,
    tokenizer,
    test_lan: str,
    head_mask: torch.Tensor,
    data_dir: str,
    data_num: int,
    max_length: int = 512,
) -> float:
    """Compute average perplexity on TED text for test_lan with a given head_mask."""
    data_file = os.path.join(data_dir, f"ted_{test_lan}.json")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"TED data file not found: {data_file}. "
            f"Run  ted_loader.py --languages {test_lan}  first."
        )

    dataset = datasets.load_dataset("json", data_files=data_file, split="train")
    dataset = dataset.shuffle(seed=37).select(range(min(data_num, len(dataset))))

    is_patched = hasattr(model.model.layers[0].self_attn, "_lahis_mask")
    hm = head_mask.to(model.device).to(model.dtype)

    if is_patched:
        model_handler.set_lahis_head_mask(model, hm)

    ppl_total = 0.0
    count = 0
    for data_dict in tqdm(dataset, desc=f"  PPL test={test_lan}", leave=False):
        text = data_dict.get("text", "")
        if not text:
            continue
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).input_ids.to(model.device)

        with torch.no_grad():
            if is_patched:
                outputs = model(input_ids, labels=input_ids)
            else:
                outputs = model(input_ids, labels=input_ids, head_mask=hm)
            ppl = torch.exp(outputs.loss).item()
            ppl_total += ppl
            count += 1

    if is_patched:
        model_handler.clear_lahis_head_mask(model)

    return round(ppl_total / max(count, 1), 3)


def run_specificity_experiment(
    model,
    tokenizer,
    languages: list,
    model_name: str,
    results_dir: str,
    data_dir: str,
    data_num: int = 500,
    p: float = 0.02,
    test_type: str = "full",
    max_length: int = 512,
) -> dict:
    """
    Full dark-diagonal experiment.

    Returns a dict:
        {
          "ori":   {test_lan: ppl_baseline, ...},
          "masked": {head_lan: {test_lan: ppl, ...}, ...},
          "delta":  {head_lan: {test_lan: ppl_delta, ...}, ...},
        }
    """
    # --- baseline (all heads kept) ---
    print("\n=== Baseline perplexity (no masking) ===")
    ori_ppl = {}
    for lan in languages:
        ori_mask = torch.ones(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
        )
        ppl = compute_ppl(model, tokenizer, lan, ori_mask, data_dir, data_num, max_length)
        ori_ppl[lan] = ppl
        print(f"  {lan:8s}: {ppl:.2f}")

    # --- masked (disable head_lan's top-p% heads, test on all languages) ---
    masked_ppl = {}
    print(f"\n=== Masked PPL (test_type={test_type}, p={p:.0%}) ===")
    for head_lan in languages:
        print(f"\n  Disabling {head_lan} heads:")
        masked_ppl[head_lan] = {}
        try:
            hm = get_top_head_mask(
                model_name, head_lan, results_dir, p, test_type
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        for test_lan in languages:
            ppl = compute_ppl(
                model, tokenizer, test_lan, hm, data_dir, data_num, max_length
            )
            masked_ppl[head_lan][test_lan] = ppl
            marker = " <--" if test_lan == head_lan else ""
            print(f"    head={head_lan}, test={test_lan}: {ppl:.2f}{marker}")

    # --- delta = masked - baseline ---
    delta = {}
    for head_lan, test_ppl_dict in masked_ppl.items():
        delta[head_lan] = {
            t: round(test_ppl_dict[t] - ori_ppl.get(t, 0), 3)
            for t in test_ppl_dict
        }

    results = {"ori": ori_ppl, "masked": masked_ppl, "delta": delta}
    out_path = os.path.join(results_dir, f"specificity_{test_type}_p{int(p*100)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    return results


def plot_dark_diagonal(
    results: dict,
    languages: list,
    model_name: str,
    output_path: str,
    value_key: str = "delta",
):
    """
    Plot the (head_lang × test_lang) matrix of PPL deltas.
    The diagonal should be the darkest (most hurt = most specific).
    """
    n = len(languages)
    matrix = []
    for head_lan in languages:
        row = []
        for test_lan in languages:
            val = results.get(value_key, {}).get(head_lan, {}).get(test_lan, 0.0)
            row.append(val)
        matrix.append(row)

    import numpy as np
    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    im = ax.imshow(mat, cmap="Reds", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([l.upper() for l in languages], fontsize=10)
    ax.set_yticklabels([l.upper() for l in languages], fontsize=10)
    ax.set_xlabel("Language tested (text)", fontsize=11)
    ax.set_ylabel("Language whose heads are disabled", fontsize=11)
    ax.set_title(
        f"Specificity: PPL Δ when disabling top-2% heads\n"
        f"{model_name.upper()} + TED (darker = more hurt)",
        fontsize=12,
    )

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if mat[i,j] < mat.max() * 0.6 else "white")

    fig.colorbar(im, ax=ax, label="PPL increase vs baseline")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Dark-diagonal heatmap saved -> {output_path}")
    plt.close(fig)


def _resolve_device(device_arg):
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def start(args):
    device = _resolve_device(args.device)
    model, tokenizer = model_handler.load_model(
        args.model, device, args.half_precision, args.local
    )

    results_dir = args.results_dir or f"../results/{args.model}"
    results = run_specificity_experiment(
        model, tokenizer,
        languages=args.languages,
        model_name=args.model,
        results_dir=results_dir,
        data_dir=args.data_dir,
        data_num=args.data_num,
        p=args.p,
        test_type=args.test_type,
        max_length=args.max_length,
    )

    out_dir = os.path.join(results_dir, "heatmaps")
    plot_dark_diagonal(
        results, args.languages, args.model,
        output_path=os.path.join(out_dir, f"dark_diagonal_{args.test_type}.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LAHIS specificity (dark-diagonal) evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama2", choices=MODEL_CHOICES)
    parser.add_argument("--device", default=None)
    parser.add_argument("-b", "--half-precision", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False,
                        help="Load model from local directory (default: download from HuggingFace)")
    parser.add_argument("--languages", nargs="+",
                        default=["en", "fr", "es", "zh", "ru", "de"],
                        help="Languages to include in the experiment")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with .pth importance matrices")
    parser.add_argument("--data-dir", default="../data/ted",
                        help="Directory with ted_{lang}.json files")
    parser.add_argument("--data-num", type=int, default=500,
                        help="Samples per language for PPL evaluation")
    parser.add_argument("--p", type=float, default=0.02,
                        help="Fraction of heads to disable (default: top 2%%)")
    parser.add_argument("--test-type", default="full",
                        choices=["full", "specific", "random", "ori"],
                        help="Which heads to disable (full=all top-p, specific=lang-specific only)")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    start(args)
