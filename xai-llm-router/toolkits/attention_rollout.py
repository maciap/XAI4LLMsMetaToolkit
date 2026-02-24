# toolkits/attention_rollout.py
# - Forces eager attention (when supported) so attentions are materialized.
# - Infers task mode (generation vs classification) from the loaded model config when possible.
# - Still supports *both* in one plugin; task_mode is optional and defaults to "auto".
# - Returns rollout vector for a target token + optional full rollout matrix.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
    )
except Exception:
    AutoTokenizer = None
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None


def _head_fuse(attn_hss: torch.Tensor, how: str) -> torch.Tensor:
    """
    attn_hss: (heads, seq, seq)
    returns: (seq, seq)
    """
    how = (how or "mean").strip().lower()
    if how == "mean":
        return attn_hss.mean(dim=0)
    if how == "max":
        return attn_hss.max(dim=0).values
    if how == "min":
        return attn_hss.min(dim=0).values
    raise ValueError(f"Unknown head_fuse='{how}'. Choose from: mean, max, min")


def _rollout_from_attentions(
    attentions: List[torch.Tensor],  # each: (batch, heads, seq, seq) or None
    head_fuse: str = "mean",
    residual_weight: float = 1.0,
    start_layer: int = 0,
) -> torch.Tensor:
    if not attentions:
        raise ValueError("No attention tensors provided (attentions list is empty).")

    start_layer = int(start_layer)
    if start_layer < 0:
        start_layer = 0
    if start_layer >= len(attentions):
        raise ValueError(f"start_layer={start_layer} >= num_layers={len(attentions)}")

    As: List[torch.Tensor] = []
    for layer_attn in attentions[start_layer:]:
        if layer_attn is None:
            continue

        if layer_attn.dim() != 4:
            raise ValueError(f"Expected attention tensor (batch, heads, seq, seq), got {tuple(layer_attn.shape)}")

        A = layer_attn[0]  # (heads, seq, seq)
        A = _head_fuse(A, head_fuse)

        seq = A.size(-1)
        I = torch.eye(seq, device=A.device, dtype=A.dtype)
        A = A + float(residual_weight) * I

        # Row-normalize
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
        As.append(A)

    if not As:
        raise ValueError(
            "No usable attention matrices were produced. "
            "This usually means the model/backend did not return attention weights. "
            "Try a different model or ensure your transformers version supports returning attentions."
        )

    R = As[0]
    for A in As[1:]:
        R = A @ R
    return R


def _to_list2d(x: torch.Tensor) -> List[List[float]]:
    return x.detach().float().cpu().tolist()


def _to_list1d(x: torch.Tensor) -> List[float]:
    return x.detach().float().cpu().tolist()


def _infer_task_mode_from_config(cfg: Any) -> str:
    """
    Best-effort inference:
      - if cfg.architectures mentions SequenceClassification -> classification
      - if cfg.architectures mentions CausalLM -> generation
      - else fall back to generation (common) but we still handle classification if user overrides.
    """
    archs = getattr(cfg, "architectures", None) or []
    archs_str = " ".join([str(a) for a in archs]).lower()

    if "sequenceclassification" in archs_str or "forsequenceclassification" in archs_str:
        return "classification"
    if "causallm" in archs_str or "forcausallm" in archs_str:
        return "generation"

    # Secondary heuristic: presence of num_labels > 1 often indicates classification heads,
    # but can exist for other tasks. Keep conservative.
    num_labels = getattr(cfg, "num_labels", None)
    if isinstance(num_labels, int) and num_labels > 1 and ("bert" in str(getattr(cfg, "model_type", "")).lower() or "roberta" in str(getattr(cfg, "model_type", "")).lower()):
        return "classification"

    return "generation"


def _load_model_and_mode(model_name: str, task_mode: str) -> Tuple[Any, str]:
    """
    Loads HF model using eager attention if supported.
    task_mode: "auto" | "generation" | "classification"
    Returns (model, resolved_mode)
    """
    if AutoConfig is None:
        raise RuntimeError("transformers is not available. Install: pip install transformers")

    cfg = AutoConfig.from_pretrained(model_name)

    if (task_mode or "auto").strip().lower() == "auto":
        resolved = _infer_task_mode_from_config(cfg)
    else:
        resolved = task_mode.strip().lower()
        if resolved not in ("generation", "classification"):
            resolved = "generation"

    load_kwargs = {"attn_implementation": "eager"}  # force eager when supported

    # Try with attn_implementation; fall back if transformers is older.
    try:
        if resolved == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, **load_kwargs)
    except TypeError:
        if resolved == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg)

    return model, resolved


class AttentionRollout:
    id = "attention_rollout"
    name = "Attention Rollout"

    def spec(self) -> List[FieldSpec]:
        return [
            # OPTIONAL: keep this so you can override if inference is wrong for some checkpoints
            FieldSpec(
                key="task_mode",
                label="Task mode",
                type="select",
                options=["auto", "generation", "classification"],
                default="auto",
                help=(
                    "Usually you can leave this on 'auto' and we'll infer from the checkpoint. "
                    "Use override if your checkpoint naming/config is unusual."
                ),
            ),
            FieldSpec(
                key="model_name",
                label="Model name (HuggingFace)",
                type="text",
                default="gpt2",
                help="Examples: gpt2/distilgpt2 (generation); bert-base-uncased/roberta-base (classification).",
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                default="The capital of France is",
                help="We run a forward pass with output_attentions=True and compute attention rollout.",
            ),
            FieldSpec(
                key="device",
                label="Device",
                type="select",
                options=["auto", "cpu", "cuda"],
                default="auto",
                help="Use 'auto' to choose cuda if available.",
            ),
            FieldSpec(
                key="max_length",
                label="Max length (tokenization)",
                type="number",
                default=128,
                help="Tokenizer truncation length.",
            ),
            FieldSpec(
                key="head_fuse",
                label="Head fusion",
                type="select",
                options=["mean", "max", "min"],
                default="mean",
                help="How to fuse heads into a single (seq, seq) matrix per layer.",
            ),
            FieldSpec(
                key="residual_weight",
                label="Residual weight (identity add)",
                type="number",
                default=1.0,
                help="Adds residual via A <- A + residual_weight*I (then row-normalize).",
            ),
            FieldSpec(
                key="start_layer",
                label="Start layer",
                type="number",
                default=0,
                help="Start rollout from this layer index (skip early layers).",
            ),
            FieldSpec(
                key="target_token_index",
                label="Target token index",
                type="number",
                default=-1,
                help="Row of rollout R[target,:]. -1 = last token. For BERT-style classification you often want 0 ([CLS]).",
            ),
            FieldSpec(
                key="return_full_matrix",
                label="Return full rollout matrix",
                type="checkbox",
                default=False,
                help="Include rollout_matrix (seq x seq) in outputs (can be large).",
            ),
            FieldSpec(
                key="top_k",
                label="Top-k influential tokens (for display)",
                type="number",
                default=12,
                help="Top-k source tokens returned for the chosen target token.",
            ),
        ]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not available. Install: pip install transformers")

        task_mode = (inputs.get("task_mode") or "auto").strip().lower()
        model_name = (inputs.get("model_name") or "gpt2").strip()
        text = inputs.get("text") or ""
        max_length = int(inputs.get("max_length") or 128)

        head_fuse = (inputs.get("head_fuse") or "mean").strip().lower()
        residual_weight = float(inputs.get("residual_weight") if inputs.get("residual_weight") is not None else 1.0)
        start_layer = int(inputs.get("start_layer") or 0)
        tgt_idx = int(inputs.get("target_token_index") if inputs.get("target_token_index") is not None else -1)
        top_k = int(inputs.get("top_k") or 12)
        return_full = bool(inputs.get("return_full_matrix", False))

        device_req = (inputs.get("device") or "auto").strip().lower()
        if device_req == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_req in ("cpu", "cuda"):
            if device_req == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            else:
                device = device_req
        else:
            device = "cpu"

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"][0]
        tokens = tok.convert_ids_to_tokens(input_ids)

        model, resolved_mode = _load_model_and_mode(model_name, task_mode)
        model.to(device)
        model.eval()

        with torch.no_grad():
            out = model(**enc, output_attentions=True)

        attentions = getattr(out, "attentions", None)
        if attentions is None:
            raise ValueError(
                "Model output does not contain `.attentions`. "
                "This checkpoint/config may not support returning attentions."
            )

        if isinstance(attentions, (list, tuple)):
            n_layers = len(attentions)
            n_none = sum(a is None for a in attentions)
            if n_layers > 0 and n_none == n_layers:
                raise ValueError(
                    f"All attention tensors are None ({n_none}/{n_layers}). "
                    "Even with eager attention, this model/backend did not return attention weights. "
                    "Try a different model or transformers version."
                )

        R = _rollout_from_attentions(
            attentions=list(attentions),
            head_fuse=head_fuse,
            residual_weight=residual_weight,
            start_layer=start_layer,
        )

        seq = R.size(0)
        if tgt_idx < 0:
            tgt_idx = seq + tgt_idx
        tgt_idx = max(0, min(seq - 1, tgt_idx))

        scores = R[tgt_idx, :]  # (seq,)
        s_min = float(scores.min().item())
        s_max = float(scores.max().item())
        if s_max - s_min < 1e-12:
            scores_norm = torch.zeros_like(scores)
        else:
            scores_norm = (scores - s_min) / (s_max - s_min)

        k = max(1, min(int(top_k), seq))
        vals, idxs = torch.topk(scores_norm, k=k, largest=True)
        top_sources = [
            {"rank": int(i + 1), "index": int(idxs[i].item()), "token": str(tokens[int(idxs[i].item())]), "score": float(vals[i].item())}
            for i in range(k)
        ]

        outputs: Dict[str, Any] = {
            "plugin": self.id,
            "model": model_name,
            "task_mode": resolved_mode,  # resolved (auto -> generation/classification)
            "device": device,
            "text": text,
            "tokens": tokens,
            "target_token_index": int(tgt_idx),
            "token_scores": _to_list1d(scores_norm),
            "top_sources": top_sources,
            "params": {
                "max_length": max_length,
                "head_fuse": head_fuse,
                "residual_weight": residual_weight,
                "start_layer": start_layer,
                "return_full_matrix": return_full,
                "top_k": top_k,
                "task_mode_input": task_mode,
            },
            "notes": [
                "token_scores[i] = normalized rollout relevance of source token i to the chosen target token.",
                "Rollout is attention-based, not causal: it ignores MLPs/value vectors and nonlinearities.",
                "We force eager attention when supported so attentions are materialized.",
            ],
        }

        if return_full:
            outputs["rollout_matrix"] = _to_list2d(R)

        # Optional: add task-specific prediction info
        logits = getattr(out, "logits", None)
        if resolved_mode == "generation" and logits is not None and logits.dim() == 3:
            last_logits = logits[0, -1]
            pred_id = int(torch.argmax(last_logits).item())
            outputs["predicted_next"] = {"id": pred_id, "token": tok.convert_ids_to_tokens([pred_id])[0]}
        elif resolved_mode == "classification" and logits is not None and logits.dim() == 2:
            pred_idx = int(torch.argmax(logits[0]).item())
            id2label = getattr(model.config, "id2label", None) or {}
            outputs["predicted"] = {"idx": pred_idx, "label": id2label.get(pred_idx, str(pred_idx))}

        return outputs