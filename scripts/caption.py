#!/usr/bin/env python3
"""Legendas automáticas para o dataset do LoRA (Qwen3.5-9B em FP8).

Regra para LoRA de identidade: descrever tudo o que VARIA entre fotos
(roupa, sítio, luz, pose, enquadramento, e coisas como um corte antigo)
e NÃO descrever o que é sempre teu (cara, corpo, corte atual). O que não
é descrito fica associado ao trigger; o que é descrito fica controlável
por prompt.

- O prompt está em configs/caption_prompt.txt (edita à vontade).
- Factos por foto em configs/caption_hints.txt (ver o .example).
- --preview N: legenda N fotos e só mostra, não grava nada.
- --debug: mostra o texto exato e a resolução da imagem que entram no modelo.

Por defeito só substitui legendas que ainda são a legenda base criada pelo
prepare_dataset.py. Usa --overwrite para refazer todas.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, load_vlm  # noqa: E402

# Palavras que não deviam aparecer (descrevem identidade ou são ruído).
# Se aparecerem numa legenda, é avisado, a não ser que venham do hint dessa foto.
SUSPECT = [
    "eye", "eyes", "skin", "complexion", "muscular", "muscles", "athletic", "toned", "slim",
    "build", "handsome", "attractive", "young", "years old", "hair", "haircut", "beard",
    "stubble", "jawline", "ethnic", "caucasian", "the image", "this photo", "photo shows",
]


def with_trigger(desc: str, prefix: str) -> str:
    """'The man stands on a pier...' -> 'afnsx man stands on a pier...'."""
    idx = desc.lower().find("the man")
    if idx == -1:
        return f"{prefix}, {desc[:1].lower()}{desc[1:]}"
    return desc[:idx] + prefix + desc[idx + len("the man"):]


def load_hints(path: Path | None, data_dir: Path) -> dict[str, str]:
    """Devolve {img_XXX: factos}. Aceita chaves por número (123), nome original
    (IMG_0123.JPG) ou nome no dataset (img_007), via data/train/sources.tsv."""
    if not path or not path.is_file():
        return {}
    by_orig: dict[str, str] = {}
    by_num: dict[int, str] = {}
    src_map = data_dir / "sources.tsv"
    if src_map.is_file():
        for line in src_map.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                ds, orig = line.split("\t", 1)
                by_orig[orig.lower()] = ds
                by_orig[Path(orig).stem.lower()] = ds
                nums = re.findall(r"\d+", Path(orig).stem)  # mesma regra do select_photos.py
                if nums:
                    by_num.setdefault(int(nums[-1]), ds)
    hints: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, facts = (s.strip() for s in line.split(":", 1))
        k = key.lower()
        target = (
            k if re.fullmatch(r"img_\d{3,}", k) else
            by_orig.get(k) or (by_num.get(int(k)) if k.isdigit() else None)
        )
        if target:
            hints[target] = facts
        else:
            print(f"[aviso] hint sem foto correspondente: '{key}'")
    return hints


def build_prompt(template: str, hint: str | None) -> str:
    extra = (f"These facts about this photo are true and MUST be in the caption, even if a rule "
             f"above says not to describe that: {hint}.") if hint else ""
    return template.replace("{hints}", extra).strip()


def audit(caption: str, hint: str | None) -> list[str]:
    low, allowed = caption.lower(), (hint or "").lower()
    # (?<!-): "neutral-toned background", "well-built shelf"... não são sobre o corpo.
    return [w for w in SUSPECT if re.search(rf"(?<!-)\b{re.escape(w)}\b", low) and w not in allowed]


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--class-word", default="man")
    ap.add_argument("--precision", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--processor", default=None, help="pasta do processor, se diferente de --model")
    ap.add_argument("--prompt-file", type=Path, default=here / "configs/caption_prompt.txt")
    ap.add_argument("--hints", type=Path, default=here / "configs/caption_hints.txt")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", type=int, metavar="N", help="legenda N fotos sem gravar")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    template = args.prompt_file.read_text(encoding="utf-8")
    hints = load_hints(args.hints, args.dir)
    base_caption = f"a photo of {args.trigger} {args.class_word}"
    prefix = f"{args.trigger} {args.class_word}"

    images = sorted(args.dir.glob("*.jpg"))
    if args.preview:
        todo = images[: args.preview]
    else:
        todo = []
        for img_path in images:
            txt = img_path.with_suffix(".txt")
            current = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            if args.overwrite or not current or current == base_caption:
                todo.append(img_path)
    if not todo:
        print("Nada para legendar (usa --overwrite para refazer).")
        return 0
    if hints:
        print(f"{len(hints)} fotos com factos extra: {', '.join(sorted(hints))}")

    print(f"A carregar {args.model} ...")
    model, processor = load_vlm(args.model, args.precision, args.processor)

    flagged = []
    for i, img_path in enumerate(todo, 1):
        hint = hints.get(img_path.stem)
        prompt = build_prompt(template, hint)
        print(f"[{i}/{len(todo)}] {img_path.name}: a gerar...", end="\r", flush=True)
        t_img = time.time()
        with Image.open(img_path) as im:
            desc = " ".join(ask(model, processor, im, prompt, args.max_new_tokens, debug=args.debug).split())
        dt = time.time() - t_img
        caption = with_trigger(desc, prefix) if desc else base_caption
        bad = audit(caption, hint)
        if bad:
            flagged.append((img_path.name, bad))
        if not args.preview:
            img_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        tag = f"  <-- rever: {', '.join(bad)}" if bad else ""
        eta = dt * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {img_path.name} ({dt:.0f}s, faltam ~{eta / 60:.0f} min)"
              f"{' (hint: ' + hint + ')' if hint else ''}\n    {caption}{tag}")

    print()
    if args.preview:
        print("Pré-visualização: nada foi gravado.")
    if flagged:
        print(f"[aviso] {len(flagged)} legendas com palavras suspeitas (identidade ou ruído). Revê-as:")
        for name, bad in flagged:
            print(f"  {name}: {', '.join(bad)}")
    else:
        print("Nenhuma legenda com palavras suspeitas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
