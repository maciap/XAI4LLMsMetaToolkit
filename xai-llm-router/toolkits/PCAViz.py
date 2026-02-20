# toolkits/PCAViz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

try:
    from sklearn.decomposition import PCA
except Exception as e:
    PCA = None


# ---------- Minimal UI schema ----------
@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Any = None   # ✅ your app.py already supports getattr(f,"default",None)


class ToolkitPlugin:
    id: str
    name: str

    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_bool(x: Any, default: bool = False) -> bool:
    try:
        return bool(x)
    except Exception:
        return default


def _detect_arch(model_name: str) -> str:
    """
    Returns: "encdec" | "decoder" | "encoder"
    """
    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "is_encoder_decoder", False):
        return "encdec"

    # Heuristic: many decoder-only configs expose is_decoder=True or model_type in a known list
    if getattr(cfg, "is_decoder", False):
        return "decoder"

    # Some models don't set is_decoder; use architectures/name heuristics
    archs = [a.lower() for a in (getattr(cfg, "architectures", None) or [])]
    if any("causallm" in a or "forcausallm" in a for a in archs):
        return "decoder"

    # Otherwise default to encoder
    return "encoder"


def _drop_special(tokens: List[str], token_ids: List[int], tokenizer) -> Tuple[List[str], List[int], List[int]]:
    """
    Returns (kept_tokens, kept_token_ids, kept_indices)
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    keep_idx = [i for i, tid in enumerate(token_ids) if tid not in special_ids]
    return [tokens[i] for i in keep_idx], [token_ids[i] for i in keep_idx], keep_idx


class EmbeddingPCALayers(ToolkitPlugin):
    """
    Embedding / hidden-state PCA visualization across layers.

    - Auto-detects encoder vs decoder vs enc-dec.
    - For enc-dec, runs the *encoder* to get hidden states.
    - Returns per-layer PCA projections (pc1, pc2, pc3).
    - Supports:
        - basis_mode = "single_basis" (fit once, reuse across layers; comparable axes)
        - basis_mode = "per_layer_basis" (fit each layer; not comparable axes)
    """

    id = "embedding_pca_layers"
    name = "Embedding PCA across layers (2D + 3D)"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name",
                type="text",
                default="gpt2-small",
                help="Works with encoder-only (BERT/RoBERTa), decoder-only (GPT-2/OPT), and encoder-decoder (T5/BART).",
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                default="The capital of France is",
                help="We extract per-token representations and project them with PCA.",
            ),
            FieldSpec(
                key="max_length",
                label="Max input length (tokens)",
                type="number",
                required=False,
                default=64,
                help="Truncate input for speed (try 32–128).",
            ),
            FieldSpec(
                key="basis_mode",
                label="PCA basis mode",
                type="select",
                options=["single_basis", "per_layer_basis"],
                default="single_basis",
                help="Single basis makes axes comparable across layers; per-layer basis shows within-layer structure.",
            ),
            FieldSpec(
                key="single_basis_fit_on",
                label="Single-basis: fit PCA on",
                type="select",
                required=False,
                options=["last_layer", "first_layer", "embeddings"],
                default="last_layer",
                help="Only used when basis_mode=single_basis. Default=last_layer.",
            ),
            FieldSpec(
                key="drop_special_tokens",
                label="Drop special tokens",
                type="checkbox",
                required=False,
                default=True,
                help="For BERT-like tokenizers, drops [CLS]/[SEP]/[PAD] etc.",
            ),
            FieldSpec(
                key="n_components",
                label="PCA components",
                type="number",
                required=False,
                default=3,
                help="Set to 3 to enable draggable 3D (pc1, pc2, pc3).",
            ),
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        arch = _detect_arch(model_name)

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Ensure pad token exists when needed (some decoder tokenizers)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        if arch == "encdec":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif arch == "decoder":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)

        model.to(self.device)
        model.eval()

        bundle = {"tokenizer": tok, "model": model, "arch": arch}
        self._cache[model_name] = bundle
        return bundle

    def _get_hidden_states(self, bundle: Dict[str, Any], input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """
        Returns a tuple/list of hidden states: [layer0_embeddings, layer1, ..., layerN]
        """
        model = bundle["model"]
        arch = bundle["arch"]

        with torch.no_grad():
            if arch == "encdec":
                # Run encoder only (clean and fast)
                encoder = model.get_encoder()
                out = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs = out.hidden_states
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs = out.hidden_states

        if hs is None:
            raise RuntimeError("Model did not return hidden states. Try a different model or check config.")
        return hs

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if PCA is None:
            raise RuntimeError("scikit-learn is required (sklearn.decomposition.PCA). Please install scikit-learn.")

        model_name = (inputs.get("model_name") or "").strip() or "gpt2-small"
        text = (inputs.get("text") or "").strip()
        if not text:
            raise ValueError("Text is empty.")

        max_length = _to_int(inputs.get("max_length", 64), 64)
        max_length = max(8, min(max_length, 2048))

        basis_mode = (inputs.get("basis_mode") or "single_basis").strip()
        fit_on = (inputs.get("single_basis_fit_on") or "last_layer").strip()
        drop_special_tokens = _to_bool(inputs.get("drop_special_tokens", True), True)

        n_components = _to_int(inputs.get("n_components", 3), 3)
        n_components = 2 if n_components < 2 else (3 if n_components >= 3 else 2)

        bundle = self._load(model_name)
        tok = bundle["tokenizer"]

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

        token_ids = input_ids[0].detach().cpu().tolist()
        tokens = tok.convert_ids_to_tokens(token_ids)

        hs = self._get_hidden_states(bundle, input_ids, attention_mask)
        # hs: tuple length L+1, each [B, T, d]
        hs_cpu = [h[0].detach().cpu().numpy() for h in hs]  # list of [T, d]

        kept_indices = list(range(len(tokens)))
        kept_tokens = tokens
        kept_token_ids = token_ids
        if drop_special_tokens:
            kept_tokens, kept_token_ids, kept_indices = _drop_special(tokens, token_ids, tok)

        # Prepare matrices per layer with tokens filtered
        layer_mats = []
        for layer_idx, mat in enumerate(hs_cpu):
            # mat: [T, d]
            mat_k = mat[kept_indices, :]
            layer_mats.append(mat_k)

        # ---- PCA fitting ----
        def _fit_pca(X: np.ndarray) -> PCA:
            p = PCA(n_components=n_components)
            p.fit(X)
            return p

        # single basis pca (fit once on chosen layer)
        base_pca = None
        base_fit_layer = None
        if basis_mode == "single_basis":
            if fit_on == "embeddings":
                base_fit_layer = 0
            elif fit_on == "first_layer":
                base_fit_layer = 1 if len(layer_mats) > 1 else 0
            else:
                base_fit_layer = len(layer_mats) - 1

            base_pca = _fit_pca(layer_mats[base_fit_layer])

        projected: List[Dict[str, Any]] = []
        for layer_idx, X in enumerate(layer_mats):
            if basis_mode == "per_layer_basis":
                pca = _fit_pca(X)
                fit_label = f"layer_{layer_idx}"
            else:
                pca = base_pca
                fit_label = f"single_basis_fit_on_{base_fit_layer}"

            Z = pca.transform(X)  # [T, n_components]

            rows = []
            for i, (t, tid) in enumerate(zip(kept_tokens, kept_token_ids)):
                r = {
                    "i": int(i),
                    "token": str(t),
                    "token_id": int(tid),
                    "pc1": float(Z[i, 0]),
                    "pc2": float(Z[i, 1]),
                }
                if Z.shape[1] >= 3:
                    r["pc3"] = float(Z[i, 2])
                rows.append(r)

            evr = getattr(pca, "explained_variance_ratio_", None)
            evr_list = [float(x) for x in evr.tolist()] if evr is not None else []

            projected.append(
                {
                    "layer": int(layer_idx),
                    "rows": rows,
                    "pca_info": {
                        "method": "sklearn.PCA",
                        "basis_mode": basis_mode,
                        "fit_on": fit_label,
                        "n_components": int(n_components),
                        "explained_variance_ratio": evr_list,
                    },
                }
            )

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "arch_detected": bundle.get("arch", "NA"),
            "text": text,
            "tokens": tokens,  # original tokens (incl specials)
            "params": {
                "max_length": max_length,
                "basis_mode": basis_mode,
                "single_basis_fit_on": fit_on,
                "drop_special_tokens": drop_special_tokens,
                "n_components": int(n_components),
            },
            "projected": projected,
        }
