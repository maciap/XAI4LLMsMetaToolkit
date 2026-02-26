# toolkits/captum_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from captum.attr import (
    IntegratedGradients,
    Saliency,
    DeepLift,
    InputXGradient,
    GradientShap,
    Occlusion,
    FeatureAblation,
    NoiseTunnel,
)
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
    default: Optional[Any] = None  # optional convenience


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


# ---------- Generic Captum base (NOT registered directly) ----------
class _CaptumClassifierBase(ToolkitPlugin):
    """
    Base implementation shared by all Captum classifier plugins.

    IMPORTANT: This class is not meant to be used directly as a plugin.
    Use the small wrapper classes at the bottom (one per method), each with its own id/name.

    Supported algorithms:
      - IntegratedGradients
      - Saliency
      - DeepLift
      - InputXGradient
      - GradientShap
      - Occlusion (embedding occlusion)
      - FeatureAblation (embedding ablation)
      - NoiseTunnel(Saliency) / NoiseTunnel(IntegratedGradients) / NoiseTunnel(InputXGradient)
    """

    # Child classes must override these:
    id = "captum_classifier_base"
    name = "Captum (Classifier Attribution) — Base"
    FIXED_ALGO: str = "IntegratedGradients"

    # Child classes can drop irrelevant fields from the shared spec:
    DROP_FIELDS: set[str] = set()

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def spec(self) -> List[FieldSpec]:
        # Shared core UI fields across all methods
        fields: List[FieldSpec] = [
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

        # Method-specific knobs (only shown for methods that need them)
        # IG + NT(IG)
        fields.append(
            FieldSpec(
                key="n_steps",
                label="IG steps",
                type="number",
                required=False,
                help="Used for IntegratedGradients and NoiseTunnel(IntegratedGradients). Try 20–100.",
            )
        )

        # NoiseTunnel params
        fields.extend(
            [
                FieldSpec(
                    key="nt_type",
                    label="NoiseTunnel type",
                    type="select",
                    required=False,
                    options=["smoothgrad", "smoothgrad_sq", "vargrad"],
                    help="Used only for NoiseTunnel(...).",
                ),
                FieldSpec(
                    key="nt_samples",
                    label="NoiseTunnel samples",
                    type="number",
                    required=False,
                    help="Used only for NoiseTunnel(...). Typical: 10–50.",
                ),
                FieldSpec(
                    key="nt_stdev",
                    label="NoiseTunnel stdevs",
                    type="number",
                    required=False,
                    help="Used only for NoiseTunnel(...). Typical: 0.01–0.2 (embedding space).",
                ),
            ]
        )

        # GradientShap params
        fields.extend(
            [
                FieldSpec(
                    key="gs_samples",
                    label="GradientShap samples",
                    type="number",
                    required=False,
                    help="Used only for GradientShap. Typical: 10–50.",
                ),
                FieldSpec(
                    key="gs_stdev",
                    label="GradientShap stdevs",
                    type="number",
                    required=False,
                    help="Used only for GradientShap. Typical: 0.01–0.2 (embedding space).",
                ),
            ]
        )

        # Occlusion params
        fields.append(
            FieldSpec(
                key="occlusion_window",
                label="Occlusion window (tokens)",
                type="number",
                required=False,
                help="Used only for Occlusion. 1 occludes one token at a time; >1 uses a sliding window.",
            )
        )

        drop = set(self.DROP_FIELDS or set())
        return [f for f in fields if f.key not in drop]

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

    # ---- Forward for Captum (embeddings -> logits) ----
    def _forward_from_embeds(self, model, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits  # [B, C]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "distilbert-base-uncased-finetuned-sst-2-english"

        sentence: str = (inputs.get("sentence") or "").strip()
        if not sentence:
            raise ValueError("Sentence is empty.")

        # algorithm fixed per plugin class
        algorithm: str = self.FIXED_ALGO

        target_mode: str = (inputs.get("target_mode") or "predicted").strip()
        merge_subwords: bool = bool(inputs.get("merge_subwords", True))

        n_steps = max(5, min(_to_int(inputs.get("n_steps", 50), 50), 300))
        max_length = max(8, min(_to_int(inputs.get("max_length", 256), 256), 1024))
        target_index = _to_int(inputs.get("target_index", 0), 0)

        # NoiseTunnel params
        nt_type = (inputs.get("nt_type") or "smoothgrad").strip()
        nt_samples = max(1, min(_to_int(inputs.get("nt_samples", 20), 20), 200))
        nt_stdev = max(0.0, _to_float(inputs.get("nt_stdev", 0.02), 0.02))

        # GradientShap params
        gs_samples = max(1, min(_to_int(inputs.get("gs_samples", 20), 20), 200))
        gs_stdev = max(0.0, _to_float(inputs.get("gs_stdev", 0.02), 0.02))

        # Occlusion params
        occl_window = max(1, min(_to_int(inputs.get("occlusion_window", 1), 1), 64))

        bundle = self._load_model(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        label_map = bundle["label_map"]

        enc = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)  # [1, T]
        attention_mask = enc["attention_mask"].to(self.device)  # [1, T]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).item())

        num_labels = int(logits.shape[-1])

        if target_mode == "predicted":
            tgt_idx = pred_idx
        else:
            if not (0 <= target_index < num_labels):
                raise ValueError(f"target_index must be in [0, {num_labels-1}] for this model.")
            tgt_idx = int(target_index)

        embeddings_layer = model.get_input_embeddings()
        input_embeds = embeddings_layer(input_ids)  # [1, T, D]

        # baseline for baseline-based methods
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            baseline_ids = torch.full_like(input_ids, pad_id)
            baseline_embeds = embeddings_layer(baseline_ids)
            baseline_kind = "pad_embeddings"
        else:
            baseline_embeds = torch.zeros_like(input_embeds)
            baseline_kind = "zeros"

        forward_fn = lambda emb, am: self._forward_from_embeds(model, emb, am)

        # ---- choose explainer ----
        if algorithm == "IntegratedGradients":
            explainer = IntegratedGradients(forward_fn)
            attributions = explainer.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                target=tgt_idx,
                n_steps=n_steps,
            )

        elif algorithm == "Saliency":
            explainer = Saliency(forward_fn)
            attributions = explainer.attribute(
                inputs=input_embeds,
                additional_forward_args=(attention_mask,),
                target=tgt_idx,
            )

        elif algorithm == "DeepLift":
            explainer = DeepLift(forward_fn)
            attributions = explainer.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                target=tgt_idx,
            )

        elif algorithm == "InputXGradient":
            explainer = InputXGradient(forward_fn)
            attributions = explainer.attribute(
                inputs=input_embeds,
                additional_forward_args=(attention_mask,),
                target=tgt_idx,
            )

        elif algorithm == "GradientShap":
            explainer = GradientShap(forward_fn)
            attributions = explainer.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                target=tgt_idx,
                n_samples=gs_samples,
                stdevs=gs_stdev,
            )

        elif algorithm == "Occlusion":
            explainer = Occlusion(forward_fn)
            attributions = explainer.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                target=tgt_idx,
                sliding_window_shapes=(occl_window, input_embeds.size(-1)),
                strides=(1, input_embeds.size(-1)),
            )

        elif algorithm == "FeatureAblation":
            explainer = FeatureAblation(forward_fn)
            attributions = explainer.attribute(
                inputs=input_embeds,
                baselines=baseline_embeds,
                additional_forward_args=(attention_mask,),
                target=tgt_idx,
            )

        elif algorithm.startswith("NoiseTunnel(") and algorithm.endswith(")"):
            base_name = algorithm[len("NoiseTunnel(") : -1].strip()

            if base_name == "Saliency":
                base = Saliency(forward_fn)
                nt = NoiseTunnel(base)
                attributions = nt.attribute(
                    inputs=input_embeds,
                    additional_forward_args=(attention_mask,),
                    target=tgt_idx,
                    nt_type=nt_type,
                    nt_samples=nt_samples,
                    stdevs=nt_stdev,
                )
            elif base_name == "IntegratedGradients":
                base = IntegratedGradients(forward_fn)
                nt = NoiseTunnel(base)
                attributions = nt.attribute(
                    inputs=input_embeds,
                    baselines=baseline_embeds,
                    additional_forward_args=(attention_mask,),
                    target=tgt_idx,
                    n_steps=n_steps,
                    nt_type=nt_type,
                    nt_samples=nt_samples,
                    stdevs=nt_stdev,
                )
            elif base_name == "InputXGradient":
                base = InputXGradient(forward_fn)
                nt = NoiseTunnel(base)
                attributions = nt.attribute(
                    inputs=input_embeds,
                    additional_forward_args=(attention_mask,),
                    target=tgt_idx,
                    nt_type=nt_type,
                    nt_samples=nt_samples,
                    stdevs=nt_stdev,
                )
            else:
                raise ValueError(f"Unsupported NoiseTunnel base method: {base_name}")

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # reduce embedding dim -> token attribution
        token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu()  # [T]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())

        raw_scores = token_attr.tolist()

        if merge_subwords:
            tokens, raw_scores = _merge_wordpieces(tokens, raw_scores)

        max_abs = max((abs(x) for x in raw_scores), default=1e-9)
        norm_scores = [float(x / (max_abs + 1e-9)) for x in raw_scores]

        pred_label = label_map.get(pred_idx, str(pred_idx)) if label_map else str(pred_idx)
        tgt_label = label_map.get(tgt_idx, str(tgt_idx)) if label_map else str(tgt_idx)

        per_token = [{"token": t, "attr_raw": float(a), "attr_norm": float(n)} for t, a, n in zip(tokens, raw_scores, norm_scores)]

        # record params relevant to chosen algorithm for reproducibility
        algo_params: Dict[str, Any] = {"max_length": max_length, "merge_subwords": merge_subwords}

        if algorithm in ("IntegratedGradients",):
            algo_params.update({"n_steps": n_steps, "baseline": baseline_kind})

        if algorithm in ("DeepLift", "FeatureAblation", "Occlusion", "GradientShap"):
            algo_params.update({"baseline": baseline_kind})

        if algorithm == "GradientShap":
            algo_params.update({"gs_samples": gs_samples, "gs_stdev": gs_stdev})

        if algorithm == "Occlusion":
            algo_params.update({"occlusion_window": occl_window})

        if algorithm.startswith("NoiseTunnel("):
            algo_params.update({"nt_type": nt_type, "nt_samples": nt_samples, "nt_stdev": nt_stdev})
            if algorithm == "NoiseTunnel(IntegratedGradients)":
                algo_params.update({"n_steps": n_steps, "baseline": baseline_kind})

        return {
            "plugin": self.id,  # IMPORTANT: each subclass has its own id
            "model": model_name,
            "device": self.device,
            "sentence": sentence,
            "algorithm": algorithm,
            "target": {"mode": target_mode, "idx": tgt_idx, "label": tgt_label},
            "predicted": {"idx": pred_idx, "label": pred_label, "probs": probs.squeeze(0).detach().cpu().tolist()},
            "num_labels": num_labels,
            "params": algo_params,
            "attributions": per_token,
        }


# ---------- One-plugin-per-method wrappers (REGISTER THESE) ----------
class CaptumIGClassifierAttribution(_CaptumClassifierBase):
    id = "captum_ig_classifier"
    name = "Captum — Integrated Gradients (Classifier)"
    FIXED_ALGO = "IntegratedGradients"
    # show n_steps; hide others
    DROP_FIELDS = {"nt_type", "nt_samples", "nt_stdev", "gs_samples", "gs_stdev", "occlusion_window"}


class CaptumSaliencyClassifierAttribution(_CaptumClassifierBase):
    id = "captum_saliency_classifier"
    name = "Captum — Saliency (Classifier)"
    FIXED_ALGO = "Saliency"
    DROP_FIELDS = {"n_steps", "nt_type", "nt_samples", "nt_stdev", "gs_samples", "gs_stdev", "occlusion_window"}


class CaptumDeepLiftClassifierAttribution(_CaptumClassifierBase):
    id = "captum_deeplift_classifier"
    name = "Captum — DeepLift (Classifier)"
    FIXED_ALGO = "DeepLift"
    DROP_FIELDS = {"n_steps", "nt_type", "nt_samples", "nt_stdev", "gs_samples", "gs_stdev", "occlusion_window"}


class CaptumInputXGradientClassifierAttribution(_CaptumClassifierBase):
    id = "captum_inputxgradient_classifier"
    name = "Captum — Input×Gradient (Classifier)"
    FIXED_ALGO = "InputXGradient"
    DROP_FIELDS = {"n_steps", "nt_type", "nt_samples", "nt_stdev", "gs_samples", "gs_stdev", "occlusion_window"}


class CaptumGradientShapClassifierAttribution(_CaptumClassifierBase):
    id = "captum_gradientshap_classifier"
    name = "Captum — GradientShap (Classifier)"
    FIXED_ALGO = "GradientShap"
    DROP_FIELDS = {"n_steps", "nt_type", "nt_samples", "nt_stdev", "occlusion_window"}


class CaptumOcclusionClassifierAttribution(_CaptumClassifierBase):
    id = "captum_occlusion_classifier"
    name = "Captum — Occlusion (Classifier)"
    FIXED_ALGO = "Occlusion"
    DROP_FIELDS = {"n_steps", "nt_type", "nt_samples", "nt_stdev", "gs_samples", "gs_stdev"}


class CaptumFeatureAblationClassifierAttribution(_CaptumClassifierBase):
    id = "captum_featureablation_classifier"
    name = "Captum — Feature Ablation (Classifier)"
    FIXED_ALGO = "FeatureAblation"
    DROP_FIELDS = {"n_steps", "nt_type", "nt_samples", "nt_stdev", "gs_samples", "gs_stdev", "occlusion_window"}


class CaptumNoiseTunnelSaliencyClassifierAttribution(_CaptumClassifierBase):
    id = "captum_noisetunnel_saliency_classifier"
    name = "Captum — NoiseTunnel(Saliency) (Classifier)"
    FIXED_ALGO = "NoiseTunnel(Saliency)"
    DROP_FIELDS = {"n_steps", "gs_samples", "gs_stdev", "occlusion_window"}


class CaptumNoiseTunnelIGClassifierAttribution(_CaptumClassifierBase):
    id = "captum_noisetunnel_ig_classifier"
    name = "Captum — NoiseTunnel(Integrated Gradients) (Classifier)"
    FIXED_ALGO = "NoiseTunnel(IntegratedGradients)"
    DROP_FIELDS = {"gs_samples", "gs_stdev", "occlusion_window"}


class CaptumNoiseTunnelInputXGradClassifierAttribution(_CaptumClassifierBase):
    id = "captum_noisetunnel_inputxgrad_classifier"
    name = "Captum — NoiseTunnel(Input×Gradient) (Classifier)"
    FIXED_ALGO = "NoiseTunnel(InputXGradient)"
    DROP_FIELDS = {"n_steps", "gs_samples", "gs_stdev", "occlusion_window"}