# toolkits/captum_classifier_methods.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from captum.attr import IntegratedGradients, Saliency, DeepLift
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


# =========================
# Minimal plugin interfaces
# =========================
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


# =========================
# Helpers
# =========================
def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _merge_wordpieces(tokens: List[str], scores: List[float]) -> Tuple[List[str], List[float]]:
    """
    Merge WordPiece tokens like ["un", "##believable"] -> ["unbelievable"]
    Score is summed over merged pieces.
    Keeps special tokens as-is.
    """
    merged_tokens: List[str] = []
    merged_scores: List[float] = []

    for tok, sc in zip(tokens, scores):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            merged_tokens.append(tok)
            merged_scores.append(float(sc))
            continue

        if tok.startswith("##") and merged_tokens and merged_tokens[-1] not in ("[CLS]", "[SEP]", "[PAD]"):
            merged_tokens[-1] = merged_tokens[-1] + tok[2:]
            merged_scores[-1] = float(merged_scores[-1] + sc)
        else:
            merged_tokens.append(tok)
            merged_scores.append(float(sc))

    return merged_tokens, merged_scores


# ============================================================
# Shared base: HF sequence-classifier attribution (embeds-based)
# ============================================================
class HFClassifierAttributionBase(ToolkitPlugin):
    """
    Shared base for HF sequence-classification attribution methods.

    Subclasses override:
      - id / name / method_name
      - spec() (optionally)
      - _compute_attributions(...)
    """

    id = "hf_classifier_attr_base"
    name = "HF Classifier Attribution Base"
    method_name = "base"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Cache HF resources by model name
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ---------- UI spec helpers ----------
    def _common_spec(self) -> List[FieldSpec]:
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
                key="target_mode",
                label="Target to explain",
                type="select",
                options=["predicted", "label_index"],
                help="Explain the predicted class or a specific label index.",
            ),
            FieldSpec(
                key="target_index",
                label="Target label index (used only if target_mode=label_index)",
                type="number",
                required=False,
                help="0-based class index. Example: SST-2 negative=0, positive=1.",
            ),
            FieldSpec(
                key="max_length",
                label="Max input length",
                type="number",
                required=False,
                help="Input will be truncated to this length.",
            ),
            FieldSpec(
                key="merge_subwords",
                label="Merge subword tokens (##...)",
                type="checkbox",
                required=False,
                help="Recommended for cleaner token display.",
            ),
        ]

    def spec(self) -> List[FieldSpec]:
        # Base spec: subclasses can extend
        return self._common_spec()

    # ---------- model loading / caching ----------
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

    # ---------- Forward for Captum (embeddings -> logits) ----------
    def _forward_from_embeds(self, model, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits  # [B, C]

    # ---------- to be implemented by subclasses ----------
    def _compute_attributions(
        self,
        *,
        model,
        input_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Must return attributions shaped like input_embeds: [1, T, D]
        """
        raise NotImplementedError

    # ---------- main run ----------
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip()
        if not model_name:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        sentence: str = (inputs.get("sentence") or "").strip()
        if not sentence:
            raise ValueError("Sentence is empty.")

        target_mode: str = inputs.get("target_mode") or "predicted"
        merge_subwords: bool = bool(inputs.get("merge_subwords", True))

        max_length = _to_int(inputs.get("max_length", 256), 256)
        max_length = max(8, min(max_length, 1024))

        target_index = _to_int(inputs.get("target_index", 0), 0)

        bundle = self._load_model(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        label_map = bundle["label_map"]

        # tokenize
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)              # [1, T]
        attention_mask = enc["attention_mask"].to(self.device)    # [1, T]

        # predict
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).item())

        num_labels = int(logits.shape[-1])

        # choose target
        if target_mode == "predicted":
            tgt_idx = pred_idx
        else:
            if not (0 <= target_index < num_labels):
                raise ValueError(f"target_index must be in [0, {num_labels-1}] for this model.")
            tgt_idx = int(target_index)

        # embeddings
        embeddings_layer = model.get_input_embeddings()
        input_embeds = embeddings_layer(input_ids)  # [1, T, D]

        # baseline: PAD embeddings if possible else zeros
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            baseline_ids = torch.full_like(input_ids, pad_id)
            baseline_embeds = embeddings_layer(baseline_ids)
        else:
            baseline_embeds = torch.zeros_like(input_embeds)

        # compute attributions (method-specific)
        attributions = self._compute_attributions(
            model=model,
            input_embeds=input_embeds,
            baseline_embeds=baseline_embeds,
            attention_mask=attention_mask,
            target_idx=tgt_idx,
            inputs=inputs,
        )

        # reduce embedding dim -> token attribution
        token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu()  # [T]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
        raw_scores = token_attr.tolist()

        # optional merge subwords
        if merge_subwords:
            tokens, raw_scores = _merge_wordpieces(tokens, raw_scores)

        # normalized scores for display
        max_abs = max((abs(x) for x in raw_scores), default=1e-9)
        norm_scores = [float(x / (max_abs + 1e-9)) for x in raw_scores]

        # label names
        pred_label = label_map.get(pred_idx, str(pred_idx)) if label_map else str(pred_idx)
        tgt_label = label_map.get(tgt_idx, str(tgt_idx)) if label_map else str(tgt_idx)

        per_token = [
            {"token": t, "attr_raw": float(a), "attr_norm": float(n)}
            for t, a, n in zip(tokens, raw_scores, norm_scores)
        ]

        # include method-specific resolved params if present
        params: Dict[str, Any] = {"max_length": max_length, "merge_subwords": merge_subwords}
        if "_resolved_n_steps" in inputs:
            params["n_steps"] = int(inputs["_resolved_n_steps"])

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "sentence": sentence,
            "algorithm": self.method_name,
            "target": {"mode": target_mode, "idx": tgt_idx, "label": tgt_label},
            "predicted": {"idx": pred_idx, "label": pred_label, "probs": probs.squeeze(0).detach().cpu().tolist()},
            "num_labels": num_labels,
            "params": params,
            "attributions": per_token,
        }


# =========================
# Method plugins (3 classes)
# =========================
class CaptumIGClassifierAttribution(HFClassifierAttributionBase):
    id = "captum_ig_classifier"
    name = "Integrated Gradients (Captum)"
    method_name = "IntegratedGradients"

    def spec(self) -> List[FieldSpec]:
        return self._common_spec() + [
            FieldSpec(
                key="n_steps",
                label="IG steps",
                type="number",
                required=False,
                help="More steps = smoother but slower (try 20–100).",
            ),
        ]

    def _compute_attributions(
        self,
        *,
        model,
        input_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        n_steps = _to_int(inputs.get("n_steps", 50), 50)
        n_steps = max(5, min(n_steps, 300))
        inputs["_resolved_n_steps"] = n_steps  # so base includes it in outputs["params"]

        explainer = IntegratedGradients(lambda emb, am: self._forward_from_embeds(model, emb, am))
        return explainer.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=target_idx,
            n_steps=n_steps,
        )


class CaptumSaliencyClassifierAttribution(HFClassifierAttributionBase):
    id = "captum_saliency_classifier"
    name = "Saliency (Captum)"
    method_name = "Saliency"

    def spec(self) -> List[FieldSpec]:
        return self._common_spec()

    def _compute_attributions(
        self,
        *,
        model,
        input_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        explainer = Saliency(lambda emb, am: self._forward_from_embeds(model, emb, am))
        return explainer.attribute(
            inputs=input_embeds,
            additional_forward_args=(attention_mask,),
            target=target_idx,
        )


class CaptumDeepLiftClassifierAttribution(HFClassifierAttributionBase):
    id = "captum_deeplift_classifier"
    name = "DeepLift (Captum)"
    method_name = "DeepLift"

    def spec(self) -> List[FieldSpec]:
        return self._common_spec()

    def _compute_attributions(
        self,
        *,
        model,
        input_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        explainer = DeepLift(lambda emb, am: self._forward_from_embeds(model, emb, am))
        return explainer.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=target_idx,
        )
