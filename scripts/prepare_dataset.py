#!/usr/bin/env python3
"""Prepara a pasta de fotos para treino.

- corrige a rotação EXIF (a Canon guarda fotos ao alto como paisagem + tag)
- converte para RGB e redimensiona o lado maior para --max-side
- deteta duplicados quase iguais (rajadas) com um hash perceptual simples
- avisa sobre fotos pequenas ou desfocadas
- copia para --dst com nomes sequenciais e cria a legenda base .txt

Se já existir um .txt ao lado de uma foto na pasta original, é aproveitado.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def dhash(img: Image.Image, size: int = 16) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    diff = (g[:, :-1] > g[:, 1:]).flatten()
    return int("".join("1" if b else "0" for b in diff), 2)


def sharpness(img: Image.Image) -> float:
    """Variância do Laplaciano numa versão de 512px. Valores baixos = desfocado."""
    g = img.convert("L")
    g.thumbnail((512, 512))
    a = np.asarray(g, dtype=np.float32)
    lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--class-word", default="man")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--min-short-side", type=int, default=768, help="avisa abaixo disto")
    ap.add_argument("--dup-threshold", type=int, default=10, help="distância de Hamming (de 256 bits) para duplicado")
    ap.add_argument("--blur-threshold", type=float, default=60.0)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--force", action="store_true", help="apaga --dst se já existir")
    args = ap.parse_args()

    src: Path = args.src.expanduser()
    dst: Path = args.dst
    if not src.is_dir():
        print(f"[erro] Pasta não existe: {src}", file=sys.stderr)
        return 1
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            print(f"[erro] {dst} já tem ficheiros. Usa --force para recriar.", file=sys.stderr)
            return 1
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS and p.is_file())
    if not files:
        print(f"[erro] Nenhuma imagem em {src}", file=sys.stderr)
        return 1

    base_caption = f"a photo of {args.trigger} {args.class_word}"
    kept: list[tuple[str, int]] = []
    sources: list[tuple[str, str]] = []
    warnings: list[str] = []
    skipped_dup = 0

    for path in files:
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    if im.mode in ("RGBA", "LA", "P"):
                        im = im.convert("RGBA")
                        bg = Image.new("RGB", im.size, (255, 255, 255))
                        bg.paste(im, mask=im.getchannel("A"))
                        im = bg
                    else:
                        im = im.convert("RGB")
                im.load()
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{path.name}: não abriu ({e})")
            continue

        h = dhash(im)
        dup_of = next((name for name, kh in kept if bin(h ^ kh).count("1") <= args.dup_threshold), None)
        if dup_of and not args.keep_duplicates:
            skipped_dup += 1
            warnings.append(f"{path.name}: quase igual a {dup_of}, ignorada")
            continue

        w, hgt = im.size
        if min(w, hgt) < args.min_short_side:
            warnings.append(f"{path.name}: pequena ({w}x{hgt})")
        s = sharpness(im)
        if s < args.blur_threshold:
            warnings.append(f"{path.name}: possivelmente desfocada (nitidez {s:.0f})")

        if max(w, hgt) > args.max_side:
            im.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)

        name = f"img_{len(kept) + 1:03d}"
        im.save(dst / f"{name}.jpg", quality=95, subsampling=0)

        src_caption = path.with_suffix(".txt")
        caption = src_caption.read_text(encoding="utf-8").strip() if src_caption.exists() else base_caption
        (dst / f"{name}.txt").write_text(caption + "\n", encoding="utf-8")
        kept.append((f"{name} ({path.name})", h))
        sources.append((name, path.name))

    (dst / "sources.tsv").write_text("".join(f"{a}\t{b}\n" for a, b in sources), encoding="utf-8")
    n = len(kept)
    print(f"\n{n} imagens preparadas em {dst}  ({skipped_dup} duplicados ignorados)")
    for w in warnings:
        print("  -", w)
    if n < 15:
        print("\n[aviso] Menos de 15 fotos: a identidade pode não ficar sólida.")
    elif n > 80:
        print("\n[aviso] Mais de 80 fotos: considera tirar as piores; qualidade > quantidade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
