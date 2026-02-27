"""
LAHIS Cross-Lingual Intervention Demos (paper §3.5 / §3.6).

Two experiments:
  A) Attention steering: given a prompt with contradictory language-A and
     language-B facts, enhancing language-A heads biases the model toward
     the language-A answer and vice versa (§3.5).

  B) English-suppression: suppress English-specific heads to reduce
     off-target English generation when prompting in language X (§3.6).

Additionally builds cross-lingual prompts from TED aligned pairs so the
"context" is real parallel text rather than hand-crafted sentences.

Usage (from LAHIS/src/):
    # Occupation steering demo (same as original multilingual_exps.py)
    python3 intervention_demo.py --model llama2 -b --exp attn-steer \\
        --lan1 zh --lan2 hi

    # TED-based cross-lingual context steering
    python3 intervention_demo.py --model llama2 -b --exp ted-steer \\
        --lan1 fr --lan2 de --ted-path ../../TED2025/multi_way.jsonl

    # English suppression
    python3 intervention_demo.py --model llama2 -b --exp en-suppress \\
        --test-lang fr --n-demos 20
"""

import argparse
import json
import os
import torch
from tqdm.auto import tqdm

import model_handler
import ted_loader as TED

MODEL_CHOICES = ["llama2", "aya", "mistral", "llama3"]
EXP_CHOICES   = ["attn-steer", "ted-steer", "en-suppress"]

# ---------------------------------------------------------------------------
# Occupation templates (reproduced from multilingual_exps.py)
# ---------------------------------------------------------------------------
POS_TEMPLATE = {
    "zh":  "Alice的职位是&&pos&&。",
    "fr":  "Le métier d'Alice est &&pos&&.",
    "hi":  "ऐलिस का पेशा &&pos&& है।",
    "vi":  "Nghề nghiệp của Alice là &&pos&&.",
    "th":  "อาชีพของAliceคือ&&pos&&.",
    "es":  "La profesión de Alice es &&pos&&.",
    "it":  "La professione di Alice è &&pos&&.",
    "pt":  "A profissão de Alice é &&pos&&.",
    "id":  "Pekerjaan Alice adalah &&pos&&.",
    "ja":  "アリスの職業は&&pos&&です。",
    "ko":  "앨리스의 직업은 &&pos&&입니다.",
    "el":  "Το επάγγελμα της Alice είναι &&pos&&.",
    "de":  "Alices Beruf ist &&pos&&.",
    "ru":  "Профессия Алисы — &&pos&&.",
    "ar":  "مهنة أليس هي &&pos&&.",
}

POS_DICT = {
    "en": ["painter", "scientist", "doctor", "gardener", "lawyer"],
    "zh": ["画家", "科学家", "医生", "园丁", "律师"],
    "fr": ["peintre", "scientifique", "médecin", "jardinier", "avocat"],
    "hi": ["चित्रकार", "वैज्ञानिक", "डॉक्टर", "माली", "वकील"],
    "vi": ["họa sĩ", "nhà khoa học", "bác sĩ", "người làm vườn", "luật sư"],
    "th": ["จิตรกร", "นักวิทยาศาสตร์", "หมอ", "คนสวน", "ทนายความ"],
    "es": ["pintor", "científico", "médico", "jardinero", "abogado"],
    "it": ["pittore", "scienziato", "medico", "giardiniere", "avvocato"],
    "pt": ["pintor", "cientista", "médico", "jardineiro", "advogado"],
    "id": ["pelukis", "ilmuwan", "dokter", "tukang kebun", "pengacara"],
    "ja": ["画家", "科学者", "医者", "庭師", "弁護士"],
    "ko": ["화가", "과학자", "의사", "정원사", "변호사"],
    "el": ["ζωγράφος", "επιστήμονας", "γιατρός", "κηπουρός", "δικηγόρος"],
    "de": ["Maler", "Wissenschaftler", "Arzt", "Gärtner", "Anwalt"],
    "ru": ["художник", "учёный", "врач", "садовник", "юрист"],
    "ar": ["رسام", "عالم", "طبيب", "بستاني", "محامٍ"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_head_indices(results_dir: str, lan: str, n: int) -> list:
    path = os.path.join(results_dir, "head_indices.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"head_indices.json not found: {path}\n"
            "Run  language_heads.py  (or attn_matrix_ted.py) first."
        )
    with open(path) as f:
        return json.load(f).get(lan, [])[:n]


def build_head_mask(model, lan: str, results_dir: str, n: int,
                    scale: float) -> torch.Tensor:
    """
    Build a [num_layers, num_heads] mask with value `scale` at the top-n
    language-specific heads for `lan`, 1.0 elsewhere.
    """
    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads
    mask = torch.ones(num_layers, num_heads)
    indices = load_head_indices(results_dir, lan, n)
    mask.view(-1)[indices] = scale
    return mask


def generate_with_mask(model, tokenizer, prompt: str, head_mask: torch.Tensor,
                       max_new_tokens: int = 15) -> str:
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    hm = head_mask.to(model.device).to(model.dtype)

    is_patched = hasattr(model.model.layers[0].self_attn, "_lahis_mask")
    if is_patched:
        model_handler.set_lahis_head_mask(model, hm)

    with torch.no_grad():
        if is_patched:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id)
        else:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 head_mask=hm,
                                 pad_token_id=tokenizer.eos_token_id)

    if is_patched:
        model_handler.clear_lahis_head_mask(model)

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip()


# ---------------------------------------------------------------------------
# Experiment A: occupation-steering (paper §3.5)
# ---------------------------------------------------------------------------

def run_occupation_steering(model, tokenizer, lan1: str, lan2: str,
                             results_dir: str, n_heads: int = 20,
                             enhance_scale: float = 3.0,
                             weaken_scale: float = 0.0,
                             n_pairs: int = 25):
    """
    Prompt structure (mixed-language context, English query):
        [lan1 fact] [filler] [lan2 fact] Alice's occupation is a

    Conditions:
      ori      : no mask
      enhance1 : enhance lan1 heads → should bias towards lan1 answer
      weaken2  : weaken  lan2 heads → same expected direction
    """
    if lan1 not in POS_TEMPLATE or lan2 not in POS_TEMPLATE:
        raise ValueError(f"No template for lan1={lan1} or lan2={lan2}. "
                         f"Add to POS_TEMPLATE dict.")

    filler = " She likes ice-cream. She always does sports at the gym in the morning. "
    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    conditions = {
        "ori":      torch.ones(num_layers, num_heads),
        "enhance1": build_head_mask(model, lan1, results_dir, n_heads, enhance_scale),
        "weaken2":  build_head_mask(model, lan2, results_dir, n_heads, weaken_scale),
    }

    results = {c: {"lan1_wins": 0, "lan2_wins": 0, "total": 0}
               for c in conditions}

    occupations = POS_DICT.get(lan1, POS_DICT["en"])
    pairs = [(i, j) for i in range(len(occupations))
             for j in range(len(occupations)) if i != j][:n_pairs]

    print(f"\n=== Occupation steering: lan1={lan1.upper()}, lan2={lan2.upper()} ===")
    for cond, mask in conditions.items():
        print(f"\n  Condition: {cond}")
        for i, j in tqdm(pairs, desc=f"  [{cond}]", leave=False):
            pos1 = POS_DICT.get(lan1, [])[i] if lan1 in POS_DICT else POS_DICT["en"][i]
            pos2 = POS_DICT.get(lan2, [])[j] if lan2 in POS_DICT else POS_DICT["en"][j]
            en1  = POS_DICT["en"][i]
            en2  = POS_DICT["en"][j]

            prompt = (
                POS_TEMPLATE[lan1].replace("&&pos&&", pos1)
                + filler
                + POS_TEMPLATE[lan2].replace("&&pos&&", pos2)
                + " Alice's occupation is a"
            )
            gen = generate_with_mask(model, tokenizer, prompt, mask, max_new_tokens=8)

            # Score: check if generation contains lan1 or lan2 term
            gen_l = gen.lower()
            hit1 = any(w.lower() in gen_l for w in [en1, pos1])
            hit2 = any(w.lower() in gen_l for w in [en2, pos2])
            results[cond]["total"] += 1
            if hit1:
                results[cond]["lan1_wins"] += 1
            elif hit2:
                results[cond]["lan2_wins"] += 1

        r = results[cond]
        total = r["total"]
        print(f"    lan1 wins: {r['lan1_wins']/total:.1%}  "
              f"lan2 wins: {r['lan2_wins']/total:.1%}  "
              f"(n={total})")

    return results


# ---------------------------------------------------------------------------
# Experiment A (TED variant): real parallel-text context steering
# ---------------------------------------------------------------------------

def run_ted_context_steering(model, tokenizer, lan1: str, lan2: str,
                              ted_path: str, results_dir: str,
                              n_heads: int = 20,
                              enhance_scale: float = 3.0,
                              weaken_scale: float = 0.0,
                              n_demos: int = 50,
                              query_template: str = "What is the topic of this text?"):
    """
    Build prompts using real TED aligned pairs:
        [lan1 sentence] [separator] [lan2 sentence] [query in English]

    Conditions:
      ori      : no intervention
      enhance1 : enhance lan1 heads → expect model to process lan1 context more
      weaken2  : suppress lan2 heads
    """
    print(f"\n=== TED context steering: lan1={lan1.upper()}, lan2={lan2.upper()} ===")

    records = TED.load_ted_jsonl(ted_path)
    pairs = TED.build_aligned_pairs(records, lan1, lan2, max_pairs=n_demos * 2)
    pairs = pairs[:n_demos]
    if not pairs:
        print(f"  No aligned pairs found for {lan1}/{lan2}")
        return

    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    conditions = {
        "ori":      torch.ones(num_layers, num_heads),
        "enhance1": build_head_mask(model, lan1, results_dir, n_heads, enhance_scale),
        "weaken2":  build_head_mask(model, lan2, results_dir, n_heads, weaken_scale),
    }

    print(f"  Using {len(pairs)} TED aligned pairs")
    print(f"  Query: '{query_template}'")

    for cond, mask in conditions.items():
        print(f"\n  [{cond}] Sample generations:")
        for k, (text1, text2) in enumerate(pairs[:5]):
            prompt = f"{text1}  //  {text2}  //  {query_template}:"
            gen = generate_with_mask(model, tokenizer, prompt, mask, max_new_tokens=20)
            print(f"    [{k+1}] ...{gen[:80]}")


# ---------------------------------------------------------------------------
# Experiment B: English-head suppression (§3.6)
# ---------------------------------------------------------------------------

def run_english_suppression(model, tokenizer, test_lang: str,
                             ted_path: str, results_dir: str,
                             n_heads: int = 20,
                             n_demos: int = 20,
                             summarise_prompt: str = None):
    """
    Suppress English-specific heads when prompting in `test_lang`.
    Expect reduced off-target English generation.
    """
    print(f"\n=== English suppression (test_lang={test_lang.upper()}) ===")

    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads

    en_mask = build_head_mask(model, "en", results_dir, n_heads, 0.0)
    ori_mask = torch.ones(num_layers, num_heads)

    records = TED.load_ted_jsonl(ted_path)
    mono = TED.build_monolingual_streams(records, [test_lang], join_n_sentences=3)
    texts = mono.get(test_lang, [])[:n_demos]
    if not texts:
        print(f"  No TED data for {test_lang}")
        return

    # Simple summarisation prompt templates
    sum_templates = {
        "fr": "Résume brièvement: &&text&& [Résumé]:",
        "de": "Fasse kurz zusammen: &&text&& [Zusammenfassung]:",
        "es": "Resume brevemente: &&text&& [Resumen]:",
        "zh": "请简短总结：&&text&&【总结】：",
        "ru": "Кратко подведи итог: &&text&& [Итог]:",
        "ar": "لخّص باختصار: &&text&& [الملخص]:",
        "ja": "簡単にまとめてください：&&text&&【要約】：",
        "ko": "간단히 요약하세요: &&text&& [요약]:",
        "vi": "Tóm tắt ngắn gọn: &&text&& [Tóm tắt]:",
        "th": "สรุปสั้นๆ: &&text&& [สรุป]:",
    }
    template = summarise_prompt or sum_templates.get(
        test_lang, f"Summarise in {test_lang}: &&text&& [Summary]:"
    )

    print(f"  Comparing ori vs english-heads-suppressed ({n_heads} heads zeroed)")
    print(f"  Template: {template[:60]}...\n")

    for k, text in enumerate(texts[:8]):
        prompt = template.replace("&&text&&", text[:300])
        gen_ori  = generate_with_mask(model, tokenizer, prompt, ori_mask, 40)
        gen_supp = generate_with_mask(model, tokenizer, prompt, en_mask,  40)
        print(f"  [{k+1}] Prompt (truncated): ...{text[:60]}...")
        print(f"       ORI : {gen_ori[:80]}")
        print(f"       SUPP: {gen_supp[:80]}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    model.eval()

    results_dir = args.results_dir or f"../results/{args.model}"
    num_layers  = model.config.num_hidden_layers
    num_heads   = model.config.num_attention_heads
    n_heads     = max(1, int(num_layers * num_heads * args.p))

    if args.exp == "attn-steer":
        run_occupation_steering(
            model, tokenizer,
            lan1=args.lan1, lan2=args.lan2,
            results_dir=results_dir,
            n_heads=n_heads,
            n_pairs=args.n_demos,
        )

    elif args.exp == "ted-steer":
        if not args.ted_path:
            raise ValueError("--ted-path is required for ted-steer experiment")
        run_ted_context_steering(
            model, tokenizer,
            lan1=args.lan1, lan2=args.lan2,
            ted_path=args.ted_path,
            results_dir=results_dir,
            n_heads=n_heads,
            n_demos=args.n_demos,
        )

    elif args.exp == "en-suppress":
        if not args.ted_path:
            raise ValueError("--ted-path is required for en-suppress experiment")
        run_english_suppression(
            model, tokenizer,
            test_lang=args.test_lang,
            ted_path=args.ted_path,
            results_dir=results_dir,
            n_heads=n_heads,
            n_demos=args.n_demos,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LAHIS cross-lingual intervention demos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama2", choices=MODEL_CHOICES)
    parser.add_argument("--device", default=None)
    parser.add_argument("-b", "--half-precision", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False,
                        help="Load model from local directory (default: download from HuggingFace)")
    parser.add_argument("--exp", default="attn-steer", choices=EXP_CHOICES,
                        help="Which experiment to run")
    parser.add_argument("--lan1", default="zh",
                        help="Language 1 (for attn-steer / ted-steer)")
    parser.add_argument("--lan2", default="hi",
                        help="Language 2 (for attn-steer / ted-steer)")
    parser.add_argument("--test-lang", default="fr",
                        help="Language to test off-target generation (en-suppress)")
    parser.add_argument("--ted-path", default=None,
                        help="Path to TED2025/multi_way.jsonl (required for ted-steer, en-suppress)")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with head_indices.json (default: ../results/{model})")
    parser.add_argument("--p", type=float, default=0.02,
                        help="Fraction of heads to enhance/suppress")
    parser.add_argument("--n-demos", type=int, default=25,
                        help="Number of test examples per condition")
    args = parser.parse_args()
    start(args)
