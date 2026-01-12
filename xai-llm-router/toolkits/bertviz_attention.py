# toolkits/bertviz_attention.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

# BertViz visualization functions
from bertviz import head_view, model_view


# ---------- Minimal UI schema (same style as Captum plugin) ----------
@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""


class ToolkitPlugin:
    id: str
    name: str

    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# ---------- Helpers ----------
def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _html_from_bertviz(obj) -> str:
    """
    BertViz returns an IPython HTML-like object in many environments.
    Convert to a raw HTML string for Streamlit embedding.
    """
    if hasattr(obj, "_repr_html_"):
        return obj._repr_html_()
    if hasattr(obj, "data"):
        return str(obj.data)
    return str(obj)


# ---------- BertViz plugin ----------
class BertVizAttention(ToolkitPlugin):
    """
    Attention visualization (BertViz) for HuggingFace encoder(-like) models.

    Notes:
    - Best with encoder models: bert-base-uncased, roberta-base, distilbert-base-uncased, etc.
    - Produces interactive HTML (JS-based) that Streamlit can embed.
    """

    id = "bertviz_attention"
    name = "BertViz (Attention Visualization) — head_view / model_view"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cache: tokenizer/model by model name
        self._cache: Dict[str, Dict[str, Any]] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (encoder)",
                type="text",
                help="Example: bert-base-uncased or roberta-base.",
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                help="Sentence to visualize attention for.",
            ),
            FieldSpec(
                key="view",
                label="BertViz view",
                type="select",
                options=["head_view", "model_view"],
                help="head_view shows attention per head; model_view shows per-layer rollups.",
            ),
            FieldSpec(
                key="max_length",
                label="Max input length",
                type="number",
                required=False,
                help="Input will be truncated to this length (try 64–256).",
            ),
            FieldSpec(
                key="use_attention_mask",
                label="Use attention mask",
                type="checkbox",
                required=False,
                help="Usually keep on. If off, model sees all tokens as real.",
            ),
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # IMPORTANT: output_attentions=True so we can get the attention tensors
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.to(self.device)
        model.eval()

        bundle = {"tokenizer": tokenizer, "model": model}
        self._cache[model_name] = bundle
        return bundle

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "bert-base-uncased"
        text = (inputs.get("text") or "").strip()
        if not text:
            raise ValueError("Text is empty.")

        view = inputs.get("view") or "head_view"

        max_length = _to_int(inputs.get("max_length", 128), 128)
        max_length = max(8, min(max_length, 1024))

        use_attention_mask = bool(inputs.get("use_attention_mask", True))

        bundle = self._load(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)

        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None and use_attention_mask:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = None

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        attentions = out.attentions  # tuple(num_layers), each [B, H, T, T]
        if attentions is None:
            raise RuntimeError("Model did not return attentions. Try a different encoder model.")

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        attn_cpu = [a.detach().cpu() for a in attentions]

        if view == "head_view":
            viz = head_view(attn_cpu, tokens, html_action="return")
        elif view == "model_view":
            viz = model_view(attn_cpu, tokens, html_action="return")
        else:
            raise ValueError(f"Unknown view: {view}")

        # viz is now an IPython HTML object
        html = getattr(viz, "data", None) or _html_from_bertviz(viz)


        html = _html_from_bertviz(viz)

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "view": view,
            "params": {"max_length": max_length, "use_attention_mask": use_attention_mask},
            "tokens": tokens,
            "html": html,
        }
