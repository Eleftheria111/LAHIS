"""
LAHIS Attention Head Importance Matrix — TED data edition.

Computes the (num_layers × num_heads) importance matrix for each language
using the LAHIS scoring formula (paper §2.2):

    importance[l, h] = mean_over_samples(|∂L/∂m_{l,h}| · m_{l,h} · 1000)
                     × freq(∂L/∂m_{l,h} · m_{l,h} < 0)

where m is the soft head mask (initialised to 1, updated each step via AdamW).

Key differences from the original attn_matrix.py:
  1. Loads TED data (ted_{lang}.json) instead of Wikipedia.
  2. Uses set_lahis_head_mask() + clear_lahis_head_mask() instead of passing
     head_mask as a model argument — required because Llama-2 does not route
     the head_mask argument down to attn_output in standard HF code.
  3. Adds --model llama2 support.

Prerequisites:
  - Run  ted_loader.py  first to generate data/ted/ted_{lang}.json
  - model_handler.load_model("llama2", ...) automatically patches attention

Usage (from LAHIS/src/):
    # Single language
    python3 attn_matrix_ted.py --model llama2 -b --lan en --data-num 1000

    # Loop over all target languages (shell)
    for LAN in en fr es zh ru de ar ja ko vi; do
        python3 attn_matrix_ted.py --model llama2 -b --lan $LAN --data-num 1000
    done

Output:
    ../results/{model_name}/{model_name}_{lan}.pth   — importance matrix tensor
"""

import argparse
import os
import torch
import torch.nn as nn
import datasets
from tqdm.auto import tqdm

import model_handler

MODEL_CHOICES = ["llama2", "aya", "mistral", "llama3"]


def get_attn_head_matrix_ted(
    model,
    tokenizer,
    lan: str,
    model_name: str = "llama2",
    data_dir: str = "../data/ted",
    data_num: int = 1000,
    max_length: int = 512,
    lr: float = 1e-2,
) -> torch.Tensor:
    """
    Compute the LAHIS importance matrix for language `lan` using TED data.

    The mask m (shape [num_layers, num_heads]) starts at 1 and is updated
    by AdamW.  After each sample, gradient statistics are accumulated.

    Returns:
        final_matrix  [num_layers, num_heads]  — higher = more important for lan
    """
    data_file = os.path.join(data_dir, f"ted_{lan}.json")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"TED data file not found: {data_file}\n"
            f"Run  python3 ted_loader.py --languages {lan}  first."
        )

    dataset = datasets.load_dataset("json", data_files=data_file, split="train")
    dataset = dataset.shuffle(seed=86).select(range(min(data_num, len(dataset))))
    print(f"[{lan}] Using {len(dataset)} samples from {data_file}")

    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    # Soft head mask — the quantity whose gradients give importance scores
    head_mask = nn.Parameter(
        torch.ones(num_layers, num_heads, dtype=torch.bfloat16),
        requires_grad=True,
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    total_head_importance = torch.zeros_like(head_mask, dtype=torch.float32)
    neg_grad_counts        = torch.zeros_like(head_mask, dtype=torch.int32)

    optimizer = torch.optim.AdamW([head_mask], lr=lr)

    is_patched = hasattr(model.model.layers[0].self_attn, "_lahis_mask")

    for data_dict in tqdm(dataset, desc=f"LAHIS [{lan}]"):
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

        # Distribute mask to all attention layers (gradient flows via slice views)
        if is_patched:
            model_handler.set_lahis_head_mask(model, head_mask)
            outputs = model(input_ids, labels=input_ids)
        else:
            # Fallback: pass via HF head_mask argument (works for aya/mistral/llama3)
            outputs = model(input_ids, labels=input_ids,
                            head_mask=head_mask.to(model.device))

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        # LAHIS importance accumulation (paper §2.2, Eq. 8)
        with torch.no_grad():
            grad = head_mask.grad.float()          # [num_layers, num_heads]
            mask = head_mask.float()
            total_head_importance += (grad.abs() * mask * 1000)
            neg_grad_counts        += (grad * mask < 0).int()

        optimizer.step()

    if is_patched:
        model_handler.clear_lahis_head_mask(model)

    # Re-enable model gradients
    for param in model.parameters():
        param.requires_grad = True

    avg_importance  = total_head_importance / len(dataset)
    avg_neg_freq    = neg_grad_counts.float() / len(dataset)
    final_matrix    = avg_importance * avg_neg_freq

    out_dir = f"../results/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_{lan}.pth")
    torch.save(final_matrix, out_path)
    print(f"  Saved importance matrix  ->  {out_path}")
    print(f"  Score range: [{final_matrix.min():.4f}, {final_matrix.max():.4f}]")

    return final_matrix


def _resolve_device(device_arg):
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def start(input_args):
    device = _resolve_device(input_args.device)
    model, tokenizer = model_handler.load_model(
        input_args.model, device, input_args.half_precision, input_args.local
    )
    get_attn_head_matrix_ted(
        model, tokenizer,
        lan=input_args.lan,
        model_name=input_args.model,
        data_dir=input_args.data_dir,
        data_num=input_args.data_num,
        max_length=input_args.max_length,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute LAHIS head importance matrix using TED data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama2", choices=MODEL_CHOICES)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("-b", "--half-precision", action="store_true", default=False,
                        help="Load model in torch.bfloat16")
    parser.add_argument("--local", action="store_true", default=False,
                        help="Load model from a local directory (default: download from HuggingFace)")
    parser.add_argument("--lan", type=str, default="en",
                        help="Language code (e.g. en, fr, zh, ar)")
    parser.add_argument("--data-dir", default="../data/ted",
                        help="Directory containing ted_{lang}.json files")
    parser.add_argument("--data-num", type=int, default=1000,
                        help="Number of text samples to use")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum token length per sample")

    args = parser.parse_args()
    start(args)
