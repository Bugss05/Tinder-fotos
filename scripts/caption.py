#!/usr/bin/env python3
"""Legendas automáticas com Qwen3-VL-4B.

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

import torch
from PIL import Image

PROMPT = (
    "Write one caption for this photo, for training an image model. "
    "Describe in one or two short sentences, in plain English: the framing "
    "(close-up, half body, full body), the pose and what the person is doing, "
    "facial expression, clothing and accessories, the location and background, "
    "and the lighting and time of day. Mention if it is a selfie or a mirror photo. "
    "Do NOT describe the face shape, eye color, skin, body type, age, ethnicity, "
    "or hair color. Do NOT say 'the image shows'. Refer to the person only as 'the man'."
)


def load_model(model_dir: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor


def with_trigger(desc: str, prefix: str) -> str:
    """'The man stands on a pier...' -> 'afnsx man stands on a pier...'."""
    idx = desc.lower().find("the man")
    if idx == -1:
        return f"{prefix}, {desc[:1].lower()}{desc[1:]}"
    return desc[:idx] + prefix + desc[idx + len("the man"):]


@torch.inference_mode()
def caption_one(model, processor, img: Image.Image, max_new_tokens: int) -> str:
    img = img.convert("RGB")
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return " ".join(text.split()).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--class-word", default="man")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[erro] GPU não disponível no PyTorch (ROCm).", file=sys.stderr)
        return 1

    base_caption = f"a photo of {args.trigger} {args.class_word}"
    prefix = f"{args.trigger} {args.class_word}"
    images = sorted(args.dir.glob("*.jpg"))
    todo = []
    for img_path in images:
        txt = img_path.with_suffix(".txt")
        current = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        if args.overwrite or not current or current == base_caption:
            todo.append(img_path)
    if not todo:
        print("Nada para legendar (usa --overwrite para refazer).")
        return 0

    print(f"A carregar {args.model} ...")
    model, processor = load_model(args.model)

    for i, img_path in enumerate(todo, 1):
        with Image.open(img_path) as im:
            desc = caption_one(model, processor, im, args.max_new_tokens)
        caption = with_trigger(desc, prefix) if desc else base_caption
        img_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        print(f"[{i}/{len(todo)}] {img_path.name}: {caption}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
