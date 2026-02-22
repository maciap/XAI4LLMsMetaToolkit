# toolkits/linear_cka.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


# ---------- Minimal UI schema ----------
@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None


class ToolkitPlugin:
    id: str
    name: str

    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# -------------------------
# Token prettification (same spirit as your others)
# -------------------------
def _pretty_token(tok: str) -> str:
    if tok in ("[CLS]", "[SEP]", "[PAD]", "[MASK]"):
        return tok
    if tok.startswith("Ġ"):
        tok = " " + tok[1:]
    if tok.startswith("▁"):
        tok = " " + tok[1:]
    return tok


def _pretty_tokens(tokens: List[str]) -> List[str]:
    return [_pretty_token(t) for t in tokens]


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -------------------------
# Linear CKA
# -------------------------
def _center(X: torch.Tensor) -> torch.Tensor:
    # X: [n, d]
    return X - X.mean(dim=0, keepdim=True)


@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Linear CKA between X and Y using:
      CKA(X,Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    with feature-centering (columns).
    """
    Xc = _center(X)
    Yc = _center(Y)

    # cross-cov-ish
    XtY = Xc.t() @ Yc  # [d, d]
    num = (XtY ** 2).sum()

    XtX = Xc.t() @ Xc
    YtY = Yc.t() @ Yc
    denom = torch.sqrt(((XtX ** 2).sum() * (YtY ** 2).sum()).clamp_min(eps))

    return float((num / denom).item())


def _select_model_loader(model_type: str):
    """
    model_type:
      - "causal_lm": AutoModelForCausalLM (works for GPT-like)
      - "encoder": AutoModel (works for BERT-like encoders)
      - "auto": try causal LM first then fallback to AutoModel
    """
    if model_type == "causal_lm":
        return AutoModelForCausalLM
    if model_type == "encoder":
        return AutoModel
    return None


class LinearCKALayers(ToolkitPlugin):
    """
    Compute a layer-by-layer linear CKA similarity matrix for a Transformer.

    Default: use hidden_states from HF forward pass (output_hidden_states=True).
    We build per-layer token representations X_l as [n_tokens, d_model] and compute CKA across layers.
    """

    id = "linear_cka_layers"
    name = "Linear CKA — representation similarity across layers"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name",
                type="text",
                help="Examples: gpt2, distilgpt2, bert-base-uncased, roberta-base ...",
                default="gpt2",
            ),
            FieldSpec(
                key="model_type",
                label="Model type",
                type="select",
                options=["auto", "causal_lm", "encoder"],
                help="auto = try causal LM first then fallback to encoder AutoModel.",
                default="auto",
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                help="We compute layer representations for this input and then CKA across layers.",
                default="The capital of France is",
            ),
            FieldSpec(
                key="max_length",
                label="Max input length (tokens)",
                type="number",
                required=False,
                help="Input will be truncated to this length.",
                default=128,
            ),
            FieldSpec(
                key="token_subset",
                label="Which token positions to use",
                type="select",
                options=["all", "last", "exclude_special"],
                help=(
                    "all = use all positions; last = use only last token; "
                    "exclude_special = drop obvious specials ([CLS]/[SEP]/[PAD])."
                ),
                default="exclude_special",
            ),
            FieldSpec(
                key="max_tokens_used",
                label="Max tokens used (subsample if longer)",
                type="number",
                required=False,
                help="If input has more tokens, uniformly subsample up to this many positions.",
                default=256,
            ),
            FieldSpec(
                key="compute_on_cpu",
                label="Force compute CKA on CPU (safer if VRAM tight)",
                type="checkbox",
                required=False,
                help="Hidden states still come from the model device; we can move reps to CPU before CKA.",
                default=False,
            ),
        ]

    def _load_model(self, model_name: str, model_type: str) -> Dict[str, Any]:
        cache_key = f"{model_name}::{model_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cfg = AutoConfig.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        loader = _select_model_loader(model_type)
        model = None
        arch_used = "unknown"

        if loader is None:  # auto
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg)
                arch_used = "causal_lm"
            except Exception:
                model = AutoModel.from_pretrained(model_name, config=cfg)
                arch_used = "encoder"
        else:
            model = loader.from_pretrained(model_name, config=cfg)
            arch_used = model_type

        model.to(self.device)
        model.eval()

        bundle = {"config": cfg, "tokenizer": tok, "model": model, "arch_used": arch_used}
        self._cache[cache_key] = bundle
        return bundle

    @torch.no_grad()
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "gpt2"
        model_type = (inputs.get("model_type") or "auto").strip()
        text = (inputs.get("text") or "").strip()
        if not text:
            raise ValueError("Input text is empty.")

        max_length = _to_int(inputs.get("max_length", 128), 128)
        max_length = max(8, min(max_length, 4096))

        token_subset = (inputs.get("token_subset") or "exclude_special").strip()
        max_tokens_used = _to_int(inputs.get("max_tokens_used", 256), 256)
        max_tokens_used = max(1, min(max_tokens_used, 4096))

        compute_on_cpu = bool(inputs.get("compute_on_cpu", False))

        bundle = self._load_model(model_name, model_type=model_type)
        tok = bundle["tokenizer"]
        model = bundle["model"]
        arch_used = bundle.get("arch_used", "unknown")

        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        tokens_raw = tok.convert_ids_to_tokens(input_ids[0].tolist())
        tokens_display = _pretty_tokens(tokens_raw)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = out.hidden_states
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states. Cannot compute CKA.")

        # hidden_states: tuple length = n_layers+1 (embeddings + each layer)
        # each: [B, T, D]
        n_layers_plus_emb = len(hidden_states)
        T = hidden_states[0].shape[1]

        # choose token indices
        idxs = list(range(T))
        if token_subset == "last":
            idxs = [T - 1]
        elif token_subset == "exclude_special":
            specials = {"[CLS]", "[SEP]", "[PAD]"}
            idxs = [i for i, t in enumerate(tokens_raw) if t not in specials]
            if not idxs:
                idxs = list(range(T))

        # subsample if too many
        if len(idxs) > max_tokens_used:
            # uniform-ish subsample
            step = len(idxs) / float(max_tokens_used)
            idxs = [idxs[int(i * step)] for i in range(max_tokens_used)]

        # build per-layer matrices X_l: [n, d]
        reps: List[torch.Tensor] = []
        for l in range(n_layers_plus_emb):
            h = hidden_states[l][0]  # [T, D]
            X = h[idxs, :]           # [n, D]
            if compute_on_cpu:
                X = X.detach().float().cpu()
            else:
                X = X.detach().float()
            reps.append(X)

        # compute CKA matrix
        L = n_layers_plus_emb
        cka = torch.zeros((L, L), dtype=torch.float32)

        for i in range(L):
            cka[i, i] = 1.0
            for j in range(i + 1, L):
                v = linear_cka(reps[i], reps[j])
                cka[i, j] = v
                cka[j, i] = v

        # labels
        # layer 0 = embeddings, layer k = transformer block k-1
        layer_labels = ["emb"] + [f"L{i}" for i in range(L - 1)]

        return {
            "plugin": self.id,
            "model": model_name,
            "arch_used": arch_used,
            "device": self.device,
            "text": text,
            "params": {
                "max_length": int(max_length),
                "token_subset": token_subset,
                "max_tokens_used": int(max_tokens_used),
                "compute_on_cpu": bool(compute_on_cpu),
            },
            "tokens_raw": tokens_raw,
            "tokens": tokens_display,
            "token_indices_used": idxs,
            "layer_labels": layer_labels,
            "cka_matrix": cka.cpu().tolist(),
        }