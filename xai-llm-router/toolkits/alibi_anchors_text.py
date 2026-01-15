# toolkits/alibi_anchors_text.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


# ---------- Minimal UI schema (same style as before) ----------
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


def _to_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_label_name(label_map: Optional[Dict[int, str]], idx: int) -> str:
    if label_map and idx in label_map:
        return str(label_map[idx])
    return str(idx)


def _ensure_spacy_nlp() -> "Any":
    """
    AnchorText for sampling_strategy='unknown' or 'similarity' needs a spaCy Language pipeline.

    We prefer a blank English pipeline to avoid model downloads.
    NOTE: for some spaCy configs this requires `spacy-lookups-data` to be installed.
    """
    try:
        import spacy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Alibi Anchors (text) requires spaCy for sampling_strategy='unknown' or 'similarity'. "
            "Install it with: pip install spacy spacy-lookups-data "
            "(and optionally: python -m spacy download en_core_web_sm)."
        ) from e

    # Prefer lightweight pipeline
    try:
        return spacy.blank("en")
    except Exception:
        # Fallback to a small model if available
        try:
            return spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError(
                "spaCy is installed but no English pipeline is available. "
                "Try: python -m spacy download en_core_web_sm"
            ) from e


# ---------- Anchors plugin ----------
class AlibiAnchorsText(ToolkitPlugin):
    """
    Alibi AnchorText for HuggingFace *sequence classification* models.

    - Black-box rule explanation: finds a set of words (anchor) that keeps prediction stable.
    - Predictor for AnchorText returns class labels (N,) for robustness across Alibi versions.
    """

    id = "alibi_anchors_text"
    name = "Alibi Anchors (Text) — AnchorText"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Dict[str, Any]] = {}  # model_name -> bundle

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (sequence classification)",
                type="text",
                help=(
                    "Example: distilbert-base-uncased-finetuned-sst-2-english "
                    "or any AutoModelForSequenceClassification model."
                ),
            ),
            FieldSpec(
                key="sentence",
                label="Input sentence",
                type="textarea",
                help="Type a sentence to explain.",
            ),
            FieldSpec(
                key="sampling_strategy",
                label="Perturbation sampling strategy",
                type="select",
                options=["unknown", "similarity", "language_model"],
                help=(
                    "'unknown' replaces words with UNKs (default, simplest). "
                    "'similarity' and 'language_model' are more advanced."
                ),
            ),
            FieldSpec(
                key="threshold",
                label="Precision threshold",
                type="number",
                required=False,
                help="Target minimum precision for the anchor (e.g., 0.85–0.99). Higher = slower.",
            ),
            FieldSpec(
                key="delta",
                label="Delta (significance)",
                type="number",
                required=False,
                help="Confidence parameter for the precision constraint (default ~0.1).",
            ),
            FieldSpec(
                key="tau",
                label="Tau (bandit tolerance)",
                type="number",
                required=False,
                help="Bandit tolerance (default ~0.15). Larger can be faster but looser.",
            ),
            FieldSpec(
                key="batch_size",
                label="Batch size (model queries)",
                type="number",
                required=False,
                help="How many perturbed samples per model batch. Bigger = faster per step but heavier.",
            ),
            FieldSpec(
                key="coverage_samples",
                label="Coverage samples",
                type="number",
                required=False,
                help="How many samples used to estimate coverage. Bigger = slower.",
            ),
            FieldSpec(
                key="beam_size",
                label="Beam size",
                type="number",
                required=False,
                help="Number of candidate anchors considered per step (1 is fastest).",
            ),
            FieldSpec(
                key="stop_on_first",
                label="Stop on first valid anchor",
                type="checkbox",
                required=False,
                help="If checked, returns the first anchor meeting the precision constraint.",
            ),
            FieldSpec(
                key="max_anchor_size",
                label="Max anchor size (optional)",
                type="number",
                required=False,
                help="Maximum number of words in the anchor. 0 or blank means no limit.",
            ),
            FieldSpec(
                key="min_samples_start",
                label="Min samples start",
                type="number",
                required=False,
                help="Samples used to initialize the search (default ~50–100).",
            ),
            FieldSpec(
                key="n_covered_ex",
                label="Examples to store (covered/counterexamples)",
                type="number",
                required=False,
                help="How many covered/counterexample samples to store (e.g., 5–20).",
            ),
            FieldSpec(
                key="max_length",
                label="Max input length (tokenizer)",
                type="number",
                required=False,
                help="Tokenizer truncation length for the model calls.",
            ),
        ]

    # ---- model loading / caching ----
    def _load_model(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        cfg = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
        model.to(self.device)
        model.eval()

        label_map = None
        if hasattr(cfg, "id2label") and isinstance(cfg.id2label, dict) and cfg.id2label:
            label_map = {int(k): str(v) for k, v in cfg.id2label.items()}

        bundle = {"config": cfg, "tokenizer": tokenizer, "model": model, "label_map": label_map}
        self._cache[model_name] = bundle
        return bundle

    def _make_predict_fn(
        self,
        model,
        tokenizer,
        max_length: int,
    ) -> Callable[[List[str]], np.ndarray]:
        """
        AnchorText predictor: List[str] -> np.ndarray of shape (N,) with integer class labels.
        """

        def predict(texts: List[str]) -> np.ndarray:
            enc = tokenizer(
                [str(t) for t in texts],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()

            return preds  # (N,)

        return predict

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # ---- basic inputs ----
        model_name = (inputs.get("model_name") or "").strip() or "distilbert-base-uncased-finetuned-sst-2-english"
        sentence = (inputs.get("sentence") or "").strip()
        if not sentence:
            raise ValueError("Sentence is empty.")

        sampling_strategy = (inputs.get("sampling_strategy") or "unknown").strip()

        # ---- anchor parameters (with sane clamps) ----
        # NOTE: These are runtime defaults; your Streamlit UI may override with its own defaults.
        threshold = _to_float(inputs.get("threshold", 0.90), 0.90)
        threshold = float(min(max(threshold, 0.5), 0.999))

        delta = _to_float(inputs.get("delta", 0.1), 0.1)
        delta = float(min(max(delta, 1e-6), 0.5))

        tau = _to_float(inputs.get("tau", 0.15), 0.15)
        tau = float(min(max(tau, 0.0), 1.0))

        batch_size = _to_int(inputs.get("batch_size", 128), 128)
        batch_size = max(1, min(batch_size, 2048))

        coverage_samples = _to_int(inputs.get("coverage_samples", 2000), 2000)
        coverage_samples = max(100, min(coverage_samples, 200000))

        beam_size = _to_int(inputs.get("beam_size", 1), 1)
        beam_size = max(1, min(beam_size, 50))

        stop_on_first = bool(inputs.get("stop_on_first", True))

        max_anchor_size_raw = _to_int(inputs.get("max_anchor_size", 3), 3)
        max_anchor_size = None if max_anchor_size_raw <= 0 else int(max_anchor_size_raw)

        min_samples_start = _to_int(inputs.get("min_samples_start", 50), 50)
        min_samples_start = max(10, min(min_samples_start, 10000))

        n_covered_ex = _to_int(inputs.get("n_covered_ex", 5), 5)
        # If UI ever passes 0, treat it as "unset" to keep UX sane.
        if n_covered_ex <= 0:
            n_covered_ex = 5
        n_covered_ex = max(0, min(n_covered_ex, 100))

        max_length = _to_int(inputs.get("max_length", 256), 256)
        max_length = max(8, min(max_length, 1024))

        # ---- load HF model ----
        bundle = self._load_model(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        label_map = bundle["label_map"]

        # ---- predictor for Anchors ----
        predict_fn = self._make_predict_fn(model=model, tokenizer=tokenizer, max_length=max_length)

        # ---- import Alibi + build explainer ----
        try:
            from alibi.explainers import AnchorText  # type: ignore
        except Exception as e:
            raise RuntimeError("Alibi is not installed. Install with: pip install alibi") from e

        nlp = None
        language_model = None

        if sampling_strategy in ("unknown", "similarity"):
            nlp = _ensure_spacy_nlp()
        elif sampling_strategy == "language_model":
            raise ValueError(
                "sampling_strategy='language_model' is not supported in this plugin yet. "
                "Use 'unknown' (recommended) or implement a LanguageModel wrapper."
            )
        else:
            raise ValueError("sampling_strategy must be one of: unknown, similarity, language_model")

        explainer = AnchorText(
            predictor=predict_fn,
            sampling_strategy=sampling_strategy,
            nlp=nlp,
            language_model=language_model,
            seed=0,
        )

        # ---- model prediction for the instance (proper probs for display) ----
        enc1 = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids_1 = enc1["input_ids"].to(self.device)
        attention_mask_1 = enc1["attention_mask"].to(self.device)

        with torch.no_grad():
            logits_1 = model(input_ids=input_ids_1, attention_mask=attention_mask_1).logits
            probs_1 = F.softmax(logits_1, dim=-1).squeeze(0).detach().cpu().numpy()  # (C,)

        pred_idx = int(np.argmax(probs_1))
        pred_label = _safe_label_name(label_map, pred_idx)

        # ---- explain ----
        exp = explainer.explain(
            text=sentence,
            threshold=threshold,
            delta=delta,
            tau=tau,
            batch_size=batch_size,
            coverage_samples=coverage_samples,
            beam_size=beam_size,
            stop_on_first=stop_on_first,
            max_anchor_size=max_anchor_size,
            min_samples_start=min_samples_start,
            n_covered_ex=n_covered_ex,
            verbose=False,
        )

        # ---- parse explanation safely ----
        data = getattr(exp, "data", {}) or {}

        anchor = data.get("anchor", None)
        precision = data.get("precision", None)
        coverage = data.get("coverage", None)

        # examples can be present in various keys depending on version
        examples_out: Dict[str, List[str]] = {"covered": [], "counterexamples": []}

        ex = data.get("examples")
        if isinstance(ex, dict):
            cov = ex.get("covered") or ex.get("covered_examples") or ex.get("covered_true") or []
            cex = ex.get("counterexamples") or ex.get("uncovered") or ex.get("covered_false") or []
            examples_out["covered"] = [str(x) for x in cov][:50]
            examples_out["counterexamples"] = [str(x) for x in cex][:50]
        else:
            cov = data.get("covered") or data.get("covered_examples") or data.get("covered_true") or []
            cex = data.get("counterexamples") or data.get("uncovered") or data.get("covered_false") or []
            if isinstance(cov, list):
                examples_out["covered"] = [str(x) for x in cov][:50]
            if isinstance(cex, list):
                examples_out["counterexamples"] = [str(x) for x in cex][:50]

        if isinstance(anchor, tuple):
            anchor = list(anchor)
        if anchor is not None and not isinstance(anchor, (list, str)):
            anchor = str(anchor)

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "sentence": sentence,
            "predicted": {
                "idx": pred_idx,
                "label": pred_label,
                "probs": probs_1.tolist(),
            },
            "anchor": anchor,
            "precision": float(precision) if isinstance(precision, (int, float, np.floating)) else precision,
            "coverage": float(coverage) if isinstance(coverage, (int, float, np.floating)) else coverage,
            "examples": examples_out,
            "params": {
                "sampling_strategy": sampling_strategy,
                "threshold": threshold,
                "delta": delta,
                "tau": tau,
                "batch_size": batch_size,
                "coverage_samples": coverage_samples,
                "beam_size": beam_size,
                "stop_on_first": stop_on_first,
                "max_anchor_size": max_anchor_size,
                "min_samples_start": min_samples_start,
                "n_covered_ex": n_covered_ex,
                "max_length": max_length,
            },
        }
