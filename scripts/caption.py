#!/usr/bin/env python3
"""Legendas automáticas para o dataset do LoRA (Qwen3.5-9B em FP8).

Regra para LoRA de identidade: descrever tudo o que NÃO deve ficar colado
à pessoa (roupa, sítio, luz, pose, enquadramento) e NÃO descrever a cara
nem o corpo. Assim o trigger absorve a identidade e o resto fica
controlável por prompt.

Por defeito só substitui legendas que ainda são a legenda base criada pelo
prepare_dataset.py. Usa --overwrite para refazer todas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, load_vlm  # noqa: E402

PROMPT = (
    "Write one caption for this photo, for training an image model. "
    "Describe in one or two short sentences, in plain English: the framing "
    "(close-up, half body, full body), the pose and what the person is doing, "
    "facial expression, clothing and accessories, the location and background, "
    "and the lighting and time of day. Mention if it is a selfie or a mirror photo. "
    "Do NOT describe the face shape, eye color, skin, body type, age, ethnicity, "
    "or hair color. Do NOT say 'the image shows'. Refer to the person only as 'the man'. "
    "Output only the caption."
)


def with_trigger(desc: str, prefix: str) -> str:
    """'The man stands on a pier...' -> 'afnsx man stands on a pier...'."""
    idx = desc.lower().find("the man")
    if idx == -1:
        return f"{prefix}, {desc[:1].lower()}{desc[1:]}"
    return desc[:idx] + prefix + desc[idx + len("the man"):]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--class-word", default="man")
    ap.add_argument("--precision", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--processor", default=None, help="pasta do processor, se diferente de --model")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    base_caption = f"a photo of {args.trigger} {args.class_word}"
    prefix = f"{args.trigger} {args.class_word}"
    todo = []
    for img_path in sorted(args.dir.glob("*.jpg")):
        txt = img_path.with_suffix(".txt")
        current = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        if args.overwrite or not current or current == base_caption:
            todo.append(img_path)
    if not todo:
        print("Nada para legendar (usa --overwrite para refazer).")
        return 0

    print(f"A carregar {args.model} ...")
    model, processor = load_vlm(args.model, args.precision, args.processor)

    for i, img_path in enumerate(todo, 1):
        with Image.open(img_path) as im:
            desc = " ".join(ask(model, processor, im, PROMPT, args.max_new_tokens).split())
        caption = with_trigger(desc, prefix) if desc else base_caption
        img_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        print(f"[{i}/{len(todo)}] {img_path.name}: {caption}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
