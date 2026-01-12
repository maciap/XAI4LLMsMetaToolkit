# toolkits/captum_ig.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------- Minimal plugin contract ----------
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


# ---------- Captum Integrated Gradients plugin ----------
class CaptumIGSentiment(ToolkitPlugin):
    """
    Token attributions for sentiment classification using Integrated Gradients.

    Model: distilbert-base-uncased-finetuned-sst-2-english
    Attribution: IntegratedGradients on input embeddings; sums over embedding dim.
    """

    id = "captum_ig_sentiment"
    name = "Captum (Integrated Gradients) — Sentiment"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Captum explainer
        self.ig = IntegratedGradients(self._forward_from_embeds)

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="sentence",
                label="Input sentence",
                type="textarea",
                help="Type a sentence to explain (sentiment classifier).",
            ),
            FieldSpec(
                key="target",
                label="Target class",
                type="select",
                options=["predicted", "positive", "negative"],
                required=True,
                help="Explain the predicted class or force a class.",
            ),
            FieldSpec(
                key="n_steps",
                label="IG steps",
                type="number",
                required=True,
                help="More steps = slower but smoother attributions (try 20–100).",
            ),
        ]

    # ---- Captum forward needs a pure function of embeddings ----
    def _forward_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities (or logits) for classification, shape: [batch, num_labels]
        We return logits (preferred).
        """
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits  # [B, C]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sentence: str = (inputs.get("sentence") or "").strip()
        if not sentence:
            raise ValueError("Sentence is empty.")

        target_choice: str = inputs.get("target", "predicted")
        n_steps_raw = inputs.get("n_steps", 50)

        # Streamlit number_input returns float sometimes; normalize
        try:
            n_steps = int(n_steps_raw)
        except Exception:
            n_steps = 50
        n_steps = max(5, min(n_steps, 300))

        # Tokenize
        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)              # [1, T]
        attention_mask = enc["attention_mask"].to(self.device)    # [1, T]

        # Compute predicted class
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).item())

        # SST-2 labels are typically: 0=NEGATIVE, 1=POSITIVE
        label_map = {0: "negative", 1: "positive"}
        pred_label = label_map.get(pred_idx, str(pred_idx))

        if target_choice == "predicted":
            target_idx = pred_idx
        elif target_choice == "positive":
            target_idx = 1
        else:
            target_idx = 0

        # Build embeddings for IG
        embeddings_layer = self.model.get_input_embeddings()
        input_embeds = embeddings_layer(input_ids)  # [1, T, D]

        # Baseline: pad tokens (or zeros). Here we use PAD id if available; else zeros.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            baseline_ids = torch.full_like(input_ids, pad_id)
            baseline_embeds = embeddings_layer(baseline_ids)
        else:
            baseline_embeds = torch.zeros_like(input_embeds)

        # Captum attribute
        attributions = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=target_idx,
            n_steps=n_steps,
        )  # [1, T, D]

        # Reduce embedding dim -> token attribution
        token_attr = attributions.sum(dim=-1).squeeze(0)  # [T]
        token_attr = token_attr.detach().cpu()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())

        # Normalize for display (optional)
        # Keep raw + normalized
        raw = token_attr.tolist()
        max_abs = max((abs(x) for x in raw), default=1e-9)
        norm = [float(x / (max_abs + 1e-9)) for x in raw]

        # Prepare a compact result
        per_token = [{"token": t, "attr_raw": float(a), "attr_norm": float(n)} for t, a, n in zip(tokens, raw, norm)]

        return {
            "model": self.model_name,
            "device": self.device,
            "sentence": sentence,
            "predicted": {"label": pred_label, "probs": probs.squeeze(0).detach().cpu().tolist()},
            "explained_target": {"idx": target_idx, "label": label_map.get(target_idx, str(target_idx))},
            "n_steps": n_steps,
            "attributions": per_token,
        }
