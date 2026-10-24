#!/usr/bin/env python3
"""Junta as imagens de teste de vários checkpoints numa só grelha.

Cada subpasta de samples/ (uma por checkpoint, criada pelo 07_generate.sh)
vira uma linha; as imagens de cada pasta, por ordem de nome, viram colunas.
Como o 07 usa sempre os mesmos prompts e seeds, cada coluna é o mesmo prompt.

Uso: python3 scripts/compare_samples.py [samples] [samples/comparacao.jpg]
"""

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CELL = 384  # lado de cada miniatura
LABEL_W = 220


def step_of(name: str) -> int:
    m = re.search(r"step(\d+)", name)
    return int(m.group(1)) if m else 10**9  # o final (sem "step") vai para o fim


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else root / "comparacao.jpg")
    rows = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: (step_of(d.name), d.name))
    rows = [(d, sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)) for d in rows]
    rows = [(d, imgs) for d, imgs in rows if imgs]
    if not rows:
        sys.exit(f"[erro] Nenhuma imagem em subpastas de {root}")
    cols = max(len(imgs) for _, imgs in rows)

    sheet = Image.new("RGB", (LABEL_W + cols * CELL, len(rows) * CELL), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for r, (d, imgs) in enumerate(rows):
        y = r * CELL
        label = d.name.replace("lora_", "").replace("_x", "\nx")
        draw.multiline_text((12, y + CELL // 2 - 30), label, fill=(235, 235, 235), font=font, spacing=6)
        for c, p in enumerate(imgs):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
                sheet.paste(im, (LABEL_W + c * CELL + (CELL - im.width) // 2, y + (CELL - im.height) // 2))

    sheet.save(out, quality=90)
    print(f"{len(rows)} checkpoints x {cols} prompts -> {out}")
    print("Coluna por coluna: mesma prompt e seed. A última coluna é o controlo SEM trigger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
