# toolkits/alibi_anchors_text.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "number"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""


class ToolkitPlugin:
    id: str
    name: str
    def spec(self) -> List[FieldSpec]: raise NotImplementedError
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]: raise NotImplementedError


def _ensure_spacy_nlp() -> "Any":
    import spacy  # requires tensorflow installed in your alibi 0.5.5 setup
    return spacy.blank("en")


class AlibiAnchorsText(ToolkitPlugin):
    id = "alibi_anchors_text"
    name = "Alibi Anchors (Text)"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._nlp = None

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (sequence classification)",
                type="text",
                help="Example: distilbert-base-uncased-finetuned-sst-2-english",
            ),
            FieldSpec(
                key="sentence",
                label="Input sentence",
                type="textarea",
                help="Text to explain.",
            ),
            FieldSpec(
                key="threshold",
                label="Precision threshold",
                type="number",
                required=False,
                help="Default 0.95 (higher = slower).",
            ),
        ]

    def _get_nlp(self):
        if self._nlp is None:
            self._nlp = _ensure_spacy_nlp()
        return self._nlp

    def _load_model(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        cfg = AutoConfig.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(self.device)
        mdl.eval()

        label_map = None
        if getattr(cfg, "id2label", None):
            label_map = {int(k): str(v) for k, v in cfg.id2label.items()}

        bundle = {"cfg": cfg, "tok": tok, "mdl": mdl, "label_map": label_map}
        self._cache[model_name] = bundle
        return bundle

    def _make_predict_fn(self, mdl, tok) -> Callable[[List[str]], np.ndarray]:
        @torch.no_grad()
        def predict(texts: List[str]) -> np.ndarray:
            enc = tok(
                [str(t) for t in texts],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            logits = mdl(**enc).logits
            preds = torch.argmax(logits, dim=-1)
            return preds.cpu().numpy()
        return predict

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "distilbert-base-uncased-finetuned-sst-2-english"
        sentence = (inputs.get("sentence") or "").strip()
        if not sentence:
            raise ValueError("Sentence is empty.")
        threshold = float(inputs.get("threshold") or 0.95)

        from alibi.explainers import AnchorText  # alibi 0.5.5

        bundle = self._load_model(model_name)
        tok, mdl, label_map = bundle["tok"], bundle["mdl"], bundle["label_map"]

        predict_fn = self._make_predict_fn(mdl, tok)
        nlp = self._get_nlp()

        explainer = AnchorText(predictor=predict_fn, nlp=nlp)

        # model prediction (for display)
        enc1 = tok(sentence, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = mdl(**enc1).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = label_map.get(pred_idx, str(pred_idx)) if label_map else str(pred_idx)

        exp = explainer.explain(sentence, threshold=threshold)
        
        # Normalize anchor to avoid numpy truth-value ambiguity in Streamlit
        anchor = exp.anchor

        if isinstance(anchor, np.ndarray):
            anchor = anchor.tolist()
        elif isinstance(anchor, tuple):
            anchor = list(anchor)


        def _tolist(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return [str(t) for t in x]
            return []

        covered_true, covered_false = [], []
        raw = getattr(exp, "raw", None)

        # In your printout, examples live in exp.data['raw'], not exp.raw
        if raw is None:
            data = getattr(exp, "data", {}) or {}

            raw = data.get("raw")


        if isinstance(raw, dict):
            ex_list = raw.get("examples")
            if isinstance(ex_list, list) and ex_list:
                last = ex_list[-1] if isinstance(ex_list[-1], dict) else None
                if isinstance(last, dict):
                    covered_true = [str(t) for t in _tolist(last.get("covered_true"))]
                    covered_false = [str(t) for t in _tolist(last.get("covered_false"))]
                            


        return {
            "plugin": self.id,
            "model": model_name,
            "sentence": sentence,
            "predicted": {"idx": pred_idx, "label": pred_label, "probs": probs.tolist()},
            "anchor": anchor,
            "precision": float(exp.precision),
            "coverage": float(exp.coverage),
            "examples": {"covered_true": covered_true, "covered_false": covered_false},
            "params": {"threshold": threshold},
        }
