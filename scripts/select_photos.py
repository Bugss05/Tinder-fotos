#!/usr/bin/env python3
"""Copia as fotos escolhidas (por número) para outra pasta.

O ficheiro de números pode ter um por linha, ou separados por vírgulas ou
espaços, e intervalos como 120-130. Linhas começadas por # são ignoradas.
"123" encontra IMG_0123.JPG (os zeros à esquerda não importam).
Procura também nas subpastas (ex.: DCIM/100CANON, DCIM/101CANON).

Uso:
  python3 scripts/select_photos.py numeros.txt /media/bugss/EOS_DIGITAL/DCIM ~/fotos-lora
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(r"^IMG_(\d+)\.jpe?g$", re.IGNORECASE)


def read_numbers(path: Path) -> list[int]:
    numbers: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        for a, b in _tokens(line):
            numbers.extend(range(a, b + 1))
    return list(dict.fromkeys(numbers))  # tira repetidos, mantém a ordem


def _tokens(line: str):
    for m in re.finditer(r"(\d+)(?:\s*-\s*(\d+))?", line):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        yield (a, b) if a <= b else (b, a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("numeros", type=Path, help="txt com os números das fotos")
    ap.add_argument("origem", type=Path, help="pasta onde estão as fotos (procura nas subpastas)")
    ap.add_argument("destino", type=Path, help="pasta para onde copiar (é criada se não existir)")
    ap.add_argument("-n", "--dry-run", action="store_true", help="só mostra o que faria")
    args = ap.parse_args()

    src, dst = args.origem.expanduser(), args.destino.expanduser()
    if not args.numeros.is_file():
        sys.exit(f"[erro] Não encontrei {args.numeros}")
    if not src.is_dir():
        sys.exit(f"[erro] Pasta de origem não existe: {src}")

    wanted = read_numbers(args.numeros)
    if not wanted:
        sys.exit(f"[erro] Nenhum número em {args.numeros}")

    index: dict[int, list[Path]] = defaultdict(list)
    for p in src.rglob("*"):
        m = NAME_RE.match(p.name)
        if m and p.is_file():
            index[int(m.group(1))].append(p)

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    copied, skipped, missing, ambiguous = 0, 0, [], []
    for n in wanted:
        found = sorted(index.get(n, []))
        if not found:
            missing.append(n)
            continue
        if len(found) > 1:
            # A Canon recomeça a numeração (ex.: 100CANON e 101CANON); copia todas, com prefixo da pasta.
            ambiguous.append(n)
        for f in found:
            name = f.name if len(found) == 1 else f"{f.parent.name}_{f.name}"
            target = dst / name
            if target.exists():
                skipped += 1
                continue
            if args.dry_run:
                print(f"copiava {f} -> {target}")
            else:
                shutil.copy2(f, target)
            copied += 1

    verb = "A copiar" if args.dry_run else "Copiadas"
    print(f"\n{verb}: {copied} | já existiam: {skipped} | pedidas: {len(wanted)}  ->  {dst}")
    if missing:
        print(f"[aviso] Não encontradas ({len(missing)}): {', '.join(map(str, missing))}")
    if ambiguous:
        print(f"[aviso] Número repetido em várias pastas, copiei todas (prefixo = pasta): {', '.join(map(str, ambiguous))}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
