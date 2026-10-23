#!/usr/bin/env python3
"""Descreve fotos e responde a perguntas sobre elas (Qwen3.5-9B em FP8).

Exemplos:
  describe.py foto.jpg                          descrição detalhada
  describe.py foto.jpg -q "Esta foto serve para o dataset? Está desfocada?"
  describe.py pasta/ -q "..." -o respostas.txt  todas as imagens de uma pasta
  describe.py -i                                modo interativo (vários pedidos sem recarregar)

Responde na língua da pergunta.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, load_vlm  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_Q = (
    "Descreve esta foto em detalhe, em português de Portugal: pessoas, roupa, pose, "
    "expressão, local, fundo, luz, enquadramento e qualidade técnica (foco, ruído, exposição)."
)


def images_in(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in EXTS)
    return [path]


def run(model, processor, path: Path, question: str, max_new_tokens: int, out_file=None) -> None:
    for img in images_in(path):
        with Image.open(img) as im:
            answer = ask(model, processor, im, question, max_new_tokens)
        block = f"### {img.name}\n{answer}\n"
        print(block)
        if out_file:
            out_file.write(block + "\n")
            out_file.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("-q", "--question", default=DEFAULT_Q)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--processor", default=None, help="pasta do processor, se diferente de --model")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    if not args.interactive and args.path is None:
        ap.error("indica uma imagem/pasta ou usa -i")

    model, processor = load_vlm(args.model, args.precision, args.processor)

    if not args.interactive:
        with open(args.output, "w", encoding="utf-8") if args.output else _null() as f:
            run(model, processor, args.path, args.question, args.max_new_tokens, f)
        return 0

    print('Modo interativo. Formato: <imagem ou pasta> [pergunta]   (sem imagem = só texto; "q" para sair)')
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in {"q", "quit", "exit"}:
            break
        if not line:
            continue
        parts = shlex.split(line)
        target = Path(parts[0]).expanduser()
        if target.exists():
            question = " ".join(parts[1:]) or DEFAULT_Q
            run(model, processor, target, question, args.max_new_tokens)
        else:
            print(ask(model, processor, None, line, args.max_new_tokens), "\n")
    return 0


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    sys.exit(main())
