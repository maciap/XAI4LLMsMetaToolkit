# toolkits/probing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ---------- Minimal UI schema (same style as other plugins) ----------
@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None  # optional convenience


class ToolkitPlugin:
    id: str
    name: str

    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# ---------- Default examples (30 pos / 30 neg) ----------
DEFAULT_POS = [
    "I really enjoyed this experience.",
    "The movie was absolutely wonderful.",
    "The service was quick and friendly.",
    "I’m very happy with the results.",
    "This product works perfectly.",
    "The staff was helpful and kind.",
    "The presentation was clear and engaging.",
    "I had a great time.",
    "The book was fascinating.",
    "Everything went smoothly.",
    "The design looks beautiful.",
    "The instructions were easy to follow.",
    "This app is very useful.",
    "The performance was impressive.",
    "I love how simple this is.",
    "The support team responded quickly.",
    "The atmosphere was pleasant.",
    "The update improved everything.",
    "It exceeded my expectations.",
    "The solution was elegant.",
    "The explanation was clear.",
    "The food tasted amazing.",
    "I feel satisfied with this choice.",
    "The event was well organized.",
    "The interface is intuitive.",
    "That was a smart decision.",
    "The quality is excellent.",
    "It works exactly as described.",
    "The training was informative.",
    "I would definitely recommend this.",
]

DEFAULT_NEG = [
    "I regret trying this.",
    "The movie was extremely boring.",
    "The service was slow and rude.",
    "I’m very disappointed.",
    "This product stopped working.",
    "The staff was unhelpful.",
    "The presentation was confusing.",
    "I had a terrible time.",
    "The book was dull.",
    "Everything went wrong.",
    "The design looks messy.",
    "The instructions were unclear.",
    "This app is frustrating.",
    "The performance was disappointing.",
    "I hate how complicated this is.",
    "The support team never responded.",
    "The atmosphere was uncomfortable.",
    "The update made things worse.",
    "It failed to meet expectations.",
    "The solution was poorly designed.",
    "The explanation was confusing.",
    "The food tasted awful.",
    "I feel dissatisfied with this choice.",
    "The event was disorganized.",
    "The interface is confusing.",
    "That was a bad decision.",
    "The quality is poor.",
    "It does not work as described.",
    "The training was useless.",
    "I would not recommend this.",
]


# Fixed internal max length (removed from UI)
MAX_LENGTH = 64


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _pool_hidden(
    hs: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    hs: [B, T, D]
    attention_mask: [B, T] with 1 for real tokens, 0 for padding
    pooling: "mean" | "cls" | "last"
    returns: [B, D]
    """
    if pooling == "cls":
        return hs[:, 0, :]

    if pooling == "last":
        lengths = attention_mask.sum(dim=1).clamp(min=1)  # [B]
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        return hs.gather(dim=1, index=idx).squeeze(1)

    mask = attention_mask.unsqueeze(-1).to(hs.dtype)  # [B, T, 1]
    denom = mask.sum(dim=1).clamp(min=1.0)            # [B, 1]
    return (hs * mask).sum(dim=1) / denom


class ProbingBinaryExamples(ToolkitPlugin):
    """
    Binary linear probe trained from positive/negative examples.

    - By default, the UI starts with a built-in 30/30 sentiment set.
    - Users can edit the positive/negative lists directly in the textareas.
    """

    # IMPORTANT: set this to match your methods.json plugin_id if needed
    # If your methods.json says plugin_id="probing", change to id="probing"
    id = "probing_binary_examples"
    name = "Probing"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name",
                type="text",
                help="Encoder (e.g., distilbert-base-uncased) or decoder base model. Keep it small for speed.",
                default="distilbert-base-uncased",
            ),
            FieldSpec(
                key="positives",
                label="Positive examples (one per line)",
                type="textarea",
                required=False,
                help="Edit the default positives or paste your own (one example per line).",
                default="\n".join(DEFAULT_POS),
            ),
            FieldSpec(
                key="negatives",
                label="Negative examples (one per line)",
                type="textarea",
                required=False,
                help="Edit the default negatives or paste your own (one example per line).",
                default="\n".join(DEFAULT_NEG),
            ),
            FieldSpec(
                key="layer_index",
                label="Layer index to probe (-1 = last)",
                type="number",
                required=False,
                help="Hidden states include embeddings + transformer layers. -1 uses the last returned hidden state.",
                default=-1,
            ),
            FieldSpec(
                key="pooling",
                label="Pooling",
                type="select",
                options=["mean", "cls", "last"],
                help="mean is a robust default. cls is typical for BERT-like encoders. last is useful for decoder-like models.",
                default="mean",
            ),
            FieldSpec(
                key="classifier",
                label="Classifier",
                type="select",
                options=["logreg", "linearsvc"],
                help="Logistic regression is the standard linear probe.",
                default="logreg",
            ),
            FieldSpec(
                key="C",
                label="Regularization strength C",
                type="number",
                required=False,
                help="Smaller C = stronger regularization. Recommended for small datasets.",
                default=1.0,
            ),
            FieldSpec(
                key="cv_folds",
                label="CV folds",
                type="number",
                required=False,
                help="Stratified K-fold cross-validation.",
                default=5,
            ),
            FieldSpec(
                key="seed",
                label="Random seed",
                type="number",
                required=False,
                default=0,
                help="Controls CV split shuffling.",
            ),
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        cfg = AutoConfig.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.pad_token is None and tok.sep_token is not None:
            tok.pad_token = tok.sep_token

        model = AutoModel.from_pretrained(model_name, config=cfg)
        model.to(self.device)
        model.eval()

        bundle = {"config": cfg, "tokenizer": tok, "model": model}
        self._cache[model_name] = bundle
        return bundle

    @torch.no_grad()
    def _embed_examples(
        self,
        tokenizer,
        model,
        texts: List[str],
        layer_index: int,
        pooling: str,
    ) -> np.ndarray:
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,  # fixed internal
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = out.hidden_states
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states; cannot run probing.")

        L = len(hidden_states)
        li = layer_index
        if li < 0:
            li = L + li
        if li < 0 or li >= L:
            raise ValueError(f"layer_index={layer_index} is out of range for hidden_states length={L}.")

        hs = hidden_states[li]  # [B, T, D]
        vecs = _pool_hidden(hs, attention_mask, pooling=pooling)  # [B, D]
        return vecs.detach().float().cpu().numpy()

    def _make_clf(self, kind: str, C: float, seed: int):
        if kind == "linearsvc":
            return LinearSVC(C=float(C), class_weight="balanced", random_state=int(seed))
        return LogisticRegression(
            C=float(C),
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=int(seed),
        )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "distilbert-base-uncased"

        layer_index = _to_int(inputs.get("layer_index", -1), -1)
        pooling = inputs.get("pooling") or "mean"

        clf_kind = inputs.get("classifier") or "logreg"
        C = float(inputs.get("C", 1.0) or 1.0)

        cv_folds = _to_int(inputs.get("cv_folds", 5), 5)
        cv_folds = max(2, min(cv_folds, 10))

        seed = _to_int(inputs.get("seed", 0), 0)

        # Always read from the textareas; they come pre-filled with defaults
        pos = _split_lines(inputs.get("positives") or "\n".join(DEFAULT_POS))
        neg = _split_lines(inputs.get("negatives") or "\n".join(DEFAULT_NEG))

        pos = _dedupe_keep_order(pos)
        neg = _dedupe_keep_order(neg)

        if len(pos) < 5 or len(neg) < 5:
            raise ValueError("Need at least 5 positive and 5 negative examples (after removing empty lines).")

        overlap = sorted(set(pos).intersection(set(neg)))
        if overlap:
            raise ValueError(
                f"Some texts appear in BOTH classes ({len(overlap)}). Remove duplicates across pos/neg. Example: {overlap[0]!r}"
            )

        texts = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg), dtype=int)

        bundle = self._load(model_name)
        tok = bundle["tokenizer"]
        model = bundle["model"]

        X = self._embed_examples(
            tokenizer=tok,
            model=model,
            texts=texts,
            layer_index=layer_index,
            pooling=pooling,
        )

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        fold_rows: List[Dict[str, Any]] = []

        for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
            clf = self._make_clf(clf_kind, C=C, seed=seed)
            clf.fit(X[tr], y[tr])

            y_pred = clf.predict(X[te])

            acc = float(accuracy_score(y[te], y_pred))
            bacc = float(balanced_accuracy_score(y[te], y_pred))
            f1 = float(f1_score(y[te], y_pred, average="macro"))
            p, r, f, _ = precision_recall_fscore_support(y[te], y_pred, average="macro", zero_division=0)

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                    "macro_precision": float(p),
                    "macro_recall": float(r),
                    "macro_f1": float(f),
                    "n_test": int(len(te)),
                }
            )

            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(y_pred.tolist())

        y_true_all_np = np.array(y_true_all, dtype=int)
        y_pred_all_np = np.array(y_pred_all, dtype=int)

        cm = confusion_matrix(y_true_all_np, y_pred_all_np, labels=[0, 1]).astype(int)

        accs = [r["accuracy"] for r in fold_rows]
        baccs = [r["balanced_accuracy"] for r in fold_rows]
        f1s = [r["macro_f1"] for r in fold_rows]

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "total": int(len(texts)),
            "params": {
                "max_length": int(MAX_LENGTH),
                "layer_index": int(layer_index),
                "pooling": pooling,
                "classifier": clf_kind,
                "C": float(C),
                "cv_folds": int(cv_folds),
                "seed": int(seed),
            },
            "metrics": {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "balanced_accuracy_mean": float(np.mean(baccs)),
                "balanced_accuracy_std": float(np.std(baccs)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
            },
            "folds": fold_rows,
            "confusion_matrix": {
                "labels": ["neg(0)", "pos(1)"],
                "matrix": cm.tolist(),
            },
        }