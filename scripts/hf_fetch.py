#!/usr/bin/env python3
"""Descarrega um repo do Hugging Face (retoma se interrompido), verifica que
todos os shards referidos nos *.index.json existem e imprime só o caminho
local no stdout. Uso: hf_fetch.py <repo_id> [padrão_a_ignorar ...]"""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, ignore = sys.argv[1], sys.argv[2:] or None
path = Path(snapshot_download(repo, ignore_patterns=ignore))

missing = []
for index in path.rglob("*.safetensors.index.json"):
    for shard in set(json.loads(index.read_text())["weight_map"].values()):
        f = index.parent / shard
        if not f.is_file() or f.stat().st_size == 0:
            missing.append(str(f.relative_to(path)))
if missing:
    sys.exit(f"[erro] {repo}: faltam shards: {', '.join(sorted(missing))}")
print(path)
