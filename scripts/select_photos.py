#!/usr/bin/env python3
"""Copia as fotos escolhidas para outra pasta, a partir de uma lista num txt.

Cada entrada do txt pode ser:
  123            o número da foto: encontra IMG_0123.JPG, DSC_0123.jpg, foto123.png...
                 (usa o último grupo de algarismos do nome; zeros à esquerda não importam)
  120-130        um intervalo de números
  praia.jpg      o nome do ficheiro (com ou sem extensão, maiúsculas não importam)
Uma ou mais por linha, separadas por vírgulas. Linhas começadas por # são ignoradas.
Procura também nas subpastas.

Uso:
  python3 scripts/select_photos.py escolhidas.txt ~/Pictures/sessao ~/fotos-lora
  (acrescenta -n para ver o que faria sem copiar)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def number_of(path: Path) -> int | None:
    groups = re.findall(r"\d+", path.stem)
    return int(groups[-1]) if groups else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lista", type=Path, help="txt com os números ou nomes das fotos")
    ap.add_argument("origem", type=Path, help="pasta onde estão as fotos (procura nas subpastas)")
    ap.add_argument("destino", type=Path, help="pasta para onde copiar (é criada se não existir)")
    ap.add_argument("-n", "--dry-run", action="store_true", help="só mostra o que faria")
    args = ap.parse_args()

    src, dst = args.origem.expanduser(), args.destino.expanduser()
    if not args.lista.is_file():
        sys.exit(f"[erro] Não encontrei {args.lista}")
    if not src.is_dir():
        sys.exit(f"[erro] Pasta de origem não existe: {src}")

    by_num: dict[int, list[Path]] = defaultdict(list)
    by_name: dict[str, list[Path]] = defaultdict(list)
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            by_name[p.name.lower()].append(p)
            by_name[p.stem.lower()].append(p)
            n = number_of(p)
            if n is not None:
                by_num[n].append(p)

    # Lê a lista: (texto original, ficheiros encontrados)
    requests: list[tuple[str, list[Path]]] = []
    for raw in args.lista.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0]
        for entry in (e.strip() for e in line.split(",")):
            if not entry:
                continue
            if entry.lower() in by_name:  # nome exato tem prioridade (ex.: "foto 12.jpg")
                requests.append((entry, by_name[entry.lower()]))
                continue
            parts = entry.split()
            if all(RANGE_RE.match(t) or t.isdigit() for t in parts):
                for t in parts:
                    m = RANGE_RE.match(t)
                    a, b = (int(m.group(1)), int(m.group(2))) if m else (int(t), int(t))
                    for n in range(min(a, b), max(a, b) + 1):
                        requests.append((str(n), by_num.get(n, [])))
            else:
                requests.append((entry, []))
    if not requests:
        sys.exit(f"[erro] Lista vazia: {args.lista}")

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    done: set[Path] = set()
    copied, existed, missing, multi = 0, 0, [], []
    for label, found in requests:
        found = sorted(set(found))
        if not found:
            missing.append(label)
            continue
        if len(found) > 1:
            multi.append(f"{label} -> {', '.join(f.name for f in found)}")
        for f in found:
            if f in done:
                continue
            done.add(f)
            target = dst / f.name
            # Só muda o nome se houver colisão (mesmo nome em subpastas diferentes).
            if sum(g.name == f.name for g in found) > 1 or (target.exists() and target.stat().st_size != f.stat().st_size):
                target = dst / f"{f.parent.name}_{f.name}"
            if target.exists():
                existed += 1
                continue
            if args.dry_run:
                print(f"copiava {f} -> {target}")
            else:
                shutil.copy2(f, target)
            copied += 1

    verb = "A copiar" if args.dry_run else "Copiadas"
    print(f"\n{verb}: {copied} | já existiam: {existed}  ->  {dst}")
    if missing:
        print(f"[aviso] Não encontradas ({len(missing)}): {', '.join(missing)}")
    if multi:
        print("[aviso] Entradas que encontraram mais de uma foto (copiei todas, confirma se era isso):")
        for m in multi:
            print("   ", m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
