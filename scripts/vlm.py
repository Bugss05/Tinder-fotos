"""VLM partilhada pelo caption.py e pelo describe.py.

Modelo por defeito: Qwen3.5-9B (9.65B, multimodal nativo, Apache-2.0).
Em bf16 ocupa ~19 GB, não cabe nos 16 GB. Por isso as camadas Linear são
guardadas em FP8 (float8_e4m3fn) com uma escala por canal de saída e
desquantizadas para bf16 no momento do matmul ("weight-only FP8").

Porque assim e não um checkpoint -FP8 oficial: esses usam kernels FP8
"block-wise" (Triton/CUDA) que não estão garantidos em RDNA4. Este esquema
só precisa de suportar o dtype float8, o que o 01_setup_env.sh já testa.
Custo: um pouco mais lento que bf16 puro; perda de qualidade desprezável
(erro relativo dos pesos ~2-3%, igual ao --fp8_scaled do musubi).
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max  # 448
MIN_PARAMS_TO_QUANTIZE = 1_000_000  # camadas pequenas (gates, etc.) ficam em bf16
# O lm_head (vocabulário de 248k x 4096 = 1B params) fica em bf16: em FP8 seria
# desquantizado inteiro (2 GB temporários) a cada token gerado. Em bf16 ocupa
# 2 GB fixos e é mais rápido.
SKIP_SUFFIXES = ("lm_head",)


class FP8Linear(nn.Module):
    """Linear com pesos em FP8 + escala por linha; computa em bf16."""

    def __init__(self, linear: nn.Linear, device: torch.device, compute_dtype=torch.bfloat16):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype
        w = linear.weight.detach()
        out_f, in_f = w.shape
        q = torch.empty(out_f, in_f, dtype=FP8, device=device)
        scale = torch.empty(out_f, 1, dtype=compute_dtype, device=device)
        # Por blocos de linhas: o float32 temporário nunca passa de ~256 MB na GPU.
        rows = max(1, (64 * 2**20) // in_f)
        for i in range(0, out_f, rows):
            wc = w[i:i + rows].to(device=device, dtype=torch.float32)
            s = (wc.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
            q[i:i + rows] = (wc / s).to(FP8)
            scale[i:i + rows] = s.to(compute_dtype)
            del wc, s
        self.register_buffer("weight_fp8", q)
        self.register_buffer("weight_scale", scale)
        bias = linear.bias.detach().to(device=device, dtype=compute_dtype) if linear.bias is not None else None
        self.register_buffer("bias", bias)

    @property
    def weight(self) -> torch.Tensor:
        # Só para código externo que inspeciona .weight (dtype, device, shape).
        return self.weight_fp8.to(self.compute_dtype) * self.weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_fp8.to(self.compute_dtype) * self.weight_scale
        return F.linear(x.to(self.compute_dtype), w, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, fp8"


def quantize_to_fp8_(model: nn.Module, device: torch.device) -> tuple[int, int]:
    """Troca, no sítio, cada nn.Linear grande por FP8Linear já na GPU.
    Vai camada a camada: a camada bf16 é libertada da RAM logo a seguir."""
    from tqdm import tqdm

    names = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= MIN_PARAMS_TO_QUANTIZE
        and not name.endswith(SKIP_SUFFIXES)
    ]
    n_params = 0
    for name in tqdm(names, desc="A quantizar para FP8 (RAM -> GPU)", unit="camada"):
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        mod = getattr(parent, child)
        n_params += mod.weight.numel()
        setattr(parent, child, FP8Linear(mod, device))
        del mod  # sem outras referências: a RAM desta camada é libertada já
    gc.collect()
    return len(names), n_params


def warm_cache(model_dir: str | Path) -> None:
    """Lê os .safetensors de seguida (leitura sequencial) para a cache da RAM.
    Sem isto, os pesos são lidos aos saltos, pela ordem das camadas, o que num
    disco mecânico pode ser 5-10x mais lento. Uma 2.ª execução logo a seguir é
    quase instantânea (já está em cache)."""
    from tqdm import tqdm

    files = sorted(Path(model_dir).rglob("*.safetensors"))
    total = sum(f.stat().st_size for f in files)
    with tqdm(total=total, unit="B", unit_scale=True, desc="A ler do disco para a RAM") as bar:
        for f in files:
            with open(f, "rb", buffering=0) as fh:
                while chunk := fh.read(64 * 2**20):
                    bar.update(len(chunk))


def load_vlm(model_dir: str | Path, precision: str = "fp8", processor_dir: str | Path | None = None):
    """Carrega modelo + processor. precision: 'fp8' (defeito) ou 'bf16'.
    processor_dir: só se o processor não estiver na pasta do modelo."""
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # versões mais novas
        from transformers import AutoModelForMultimodalLM as AutoVLM

    if not torch.cuda.is_available():
        raise RuntimeError("GPU não disponível no PyTorch (ROCm). Corre scripts/00_check_system.sh")
    device = torch.device("cuda")

    t0 = time.time()
    step = lambda msg: print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)  # noqa: E731
    step(f"A ler os pesos de {model_dir}")
    warm_cache(model_dir)

    if precision == "bf16":
        model = AutoVLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    else:
        # Carrega em bf16 na RAM (~19 GB para o 9B), quantiza camada a camada
        # diretamente para a GPU, e só depois move o resto (embeddings, norms, convs).
        model = AutoVLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu")
        step("Pesos na RAM. A quantizar camada a camada para a GPU")
        n, p = quantize_to_fp8_(model, device)
        step("A mover o resto (embeddings, normalizações) para a GPU")
        model.to(device)
        gc.collect()
        print(f"FP8: {n} camadas Linear ({p / 1e9:.2f}B params) quantizadas")

    model.eval()
    processor = AutoProcessor.from_pretrained(processor_dir or model_dir)
    used = torch.cuda.memory_allocated() / 2**30
    step(f"VLM carregada ({precision}), VRAM em uso: {used:.1f} GiB")
    return model, processor


@torch.inference_mode()
def ask(model, processor, image: Image.Image | None, prompt: str, max_new_tokens: int = 256,
        max_side: int = 1280, debug: bool = False) -> str:
    """Uma pergunta, com ou sem imagem. Modo 'thinking' desligado, saída determinística."""
    content = []
    if image is not None:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    template_kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    try:
        inputs = processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
    except TypeError:  # modelos sem modo thinking (ex.: Qwen3-VL)
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    inputs = inputs.to(model.device)
    if debug:
        ids = inputs["input_ids"][0]
        print("  [debug] texto enviado ao modelo:")
        text = processor.decode(ids, skip_special_tokens=False)
        text = re.sub(r"(<\|image_pad\|>)+", lambda m: f"<|image_pad|> x{m.group(0).count('<|image_pad|>')}", text)
        print("    " + text.replace("\n", "\n    "))
        if image is not None:
            grid = inputs.get("image_grid_thw")
            print(f"  [debug] imagem: {image.size[0]}x{image.size[1]} px | grelha de patches (t,h,w): "
                  f"{grid.tolist() if grid is not None else 'NENHUMA: a imagem não chegou ao modelo!'}")

    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    if "</think>" in text:  # rede de segurança caso o template ignore enable_thinking
        text = text.split("</think>", 1)[1]
    return text.strip()
