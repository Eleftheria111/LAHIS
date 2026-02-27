"""
TED Multilingual Data Loader for LAHIS experiments.

Converts TED2025/multi_way.jsonl into per-language JSON files in the format:
    [{"text": "..."}, ...]
matching the wikipedia_{lang}.json format used by the existing LAHIS pipeline.

Usage (run from LAHIS/src/):
    python3 ted_loader.py --output-dir ../data/ted --languages en fr es zh ru de ar ja ko vi
    python3 ted_loader.py --list-languages          # show all available language codes + counts
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# TED uses non-standard codes for some languages; normalize to LAHIS codes
TED_TO_LAHIS: Dict[str, str] = {
    "zh-cn": "zh",
    "pt-br": "pt",
    # zh-tw kept as-is (distinct from zh-cn)
}
# Reverse mapping for looking up TED codes from LAHIS codes
LAHIS_TO_TED: Dict[str, str] = {v: k for k, v in TED_TO_LAHIS.items()}
LAHIS_TO_TED["zh"] = "zh-cn"   # zh -> zh-cn by default
LAHIS_TO_TED["pt"] = "pt-br"   # pt -> pt-br by default

DEFAULT_LANGS = ["en", "fr", "es", "zh", "ru", "de", "ar", "ja", "ko", "vi", "th", "hi"]
TED_JSONL_DEFAULT = "../../TED2025/multi_way.jsonl"


def load_ted_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_available_languages(records: List[dict]) -> List[str]:
    langs: set = set()
    for rec in records:
        langs.update(rec["lang_list"])
    return sorted(langs)


def _resolve_ted_code(lang: str) -> str:
    """Convert a LAHIS language code (e.g. 'zh') to the TED code (e.g. 'zh-cn')."""
    return LAHIS_TO_TED.get(lang, lang)


def _normalize_to_lahis(ted_code: str) -> str:
    """Convert a TED language code to the LAHIS code used for file naming."""
    return TED_TO_LAHIS.get(ted_code, ted_code)


def build_monolingual_streams(
    records: List[dict],
    languages: Optional[List[str]] = None,
    min_char_len: int = 20,
    max_char_len: int = 600,
    join_n_sentences: int = 5,
) -> Dict[str, List[str]]:
    """
    Build per-language text streams from TED records.

    Groups `join_n_sentences` consecutive sentences from the same language
    into a single chunk to provide better LM loss signal for LAHIS.

    Args:
        records: Parsed JSONL records from multi_way.jsonl.
        languages: Target languages as LAHIS codes (e.g. ['en', 'zh', 'fr']).
                   If None, extracts all available languages.
        min_char_len: Minimum character length for a sentence to be kept.
        max_char_len: Maximum character length for a sentence.
        join_n_sentences: How many consecutive sentences to join per text chunk.

    Returns:
        Dict mapping LAHIS language code -> list of text strings.
    """
    if languages is None:
        # Use all TED codes directly
        target_ted_codes = set(get_available_languages(records))
    else:
        # Accept both LAHIS and TED codes; always resolve to TED codes internally
        target_ted_codes = set()
        for lang in languages:
            target_ted_codes.add(lang)                          # e.g. "zh" (if TED uses it directly)
            target_ted_codes.add(_resolve_ted_code(lang))       # e.g. "zh-cn"

    per_lang_sents: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        para = rec["para_data"]
        for ted_code, text in para.items():
            if ted_code not in target_ted_codes:
                continue
            text = text.replace("\n", " ").strip()
            if min_char_len <= len(text) <= max_char_len:
                lahis_code = _normalize_to_lahis(ted_code)
                per_lang_sents[lahis_code].append(text)

    # Group sentences into chunks for richer context
    streams: Dict[str, List[str]] = {}
    for lang, sents in per_lang_sents.items():
        chunks = []
        for i in range(0, len(sents) - join_n_sentences + 1, join_n_sentences):
            chunk = " ".join(sents[i: i + join_n_sentences])
            chunks.append(chunk)
        if chunks:
            streams[lang] = chunks

    return streams


def build_aligned_pairs(
    records: List[dict],
    lang_a: str,
    lang_b: str,
    min_char_len: int = 15,
    max_pairs: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Build aligned (lang_a, lang_b) sentence pairs for intervention experiments.

    Args:
        lang_a: First language (LAHIS code, e.g. 'zh').
        lang_b: Second language (LAHIS code, e.g. 'hi').
        min_char_len: Minimum char length for each sentence in a pair.
        max_pairs: Cap the number of pairs returned (None = no cap).

    Returns:
        List of (text_a, text_b) aligned sentence pairs.
    """
    ted_a = _resolve_ted_code(lang_a)
    ted_b = _resolve_ted_code(lang_b)

    pairs: List[Tuple[str, str]] = []
    for rec in records:
        para = rec["para_data"]
        if ted_a in para and ted_b in para:
            a = para[ted_a].replace("\n", " ").strip()
            b = para[ted_b].replace("\n", " ").strip()
            if len(a) >= min_char_len and len(b) >= min_char_len:
                pairs.append((a, b))
        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs


def save_monolingual_json(
    streams: Dict[str, List[str]],
    output_dir: str,
) -> Dict[str, str]:
    """
    Save each language stream as ted_{lang}.json (list of {"text": ...} dicts).
    Format matches the wikipedia_{lang}.json files used by attn_matrix.py.

    Returns:
        Dict mapping lang code -> saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}
    for lang, texts in sorted(streams.items()):
        data = [{"text": t} for t in texts]
        out_path = os.path.join(output_dir, f"ted_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        saved[lang] = out_path
        print(f"  [{lang:8s}] {len(data):6,} chunks  ->  {out_path}")
    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build per-language TED text files for LAHIS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ted-path", default=TED_JSONL_DEFAULT,
                        help="Path to TED2025/multi_way.jsonl")
    parser.add_argument("--output-dir", default="../data/ted",
                        help="Output directory for ted_{lang}.json files")
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGS,
                        help="LAHIS language codes to extract (e.g. en fr zh)")
    parser.add_argument("--join-n", type=int, default=5,
                        help="Number of consecutive sentences to join per chunk")
    parser.add_argument("--list-languages", action="store_true",
                        help="Print all available TED language codes with counts and exit")
    args = parser.parse_args()

    print(f"Loading TED data from: {args.ted_path}")
    records = load_ted_jsonl(args.ted_path)
    print(f"  {len(records):,} records loaded\n")

    if args.list_languages:
        from collections import Counter
        counts: Counter = Counter()
        for r in records:
            for l in r["lang_list"]:
                counts[l] += 1
        print(f"Available language codes ({len(counts)}):")
        for lang, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            normalized = _normalize_to_lahis(lang)
            note = f"  (use as '{normalized}' in LAHIS)" if normalized != lang else ""
            print(f"  {lang:12s}: {cnt:8,} sentences{note}")
        raise SystemExit(0)

    print(f"Building streams for: {args.languages}  (join_n={args.join_n})")
    streams = build_monolingual_streams(records, args.languages, join_n_sentences=args.join_n)
    print(f"\nSaving to {args.output_dir}/:")
    save_monolingual_json(streams, args.output_dir)
    print("\nDone. Pass --output-dir value as --data-dir to attn_matrix_ted.py")
