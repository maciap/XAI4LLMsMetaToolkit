# toolkits/logit_lens.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _pretty_token(tok: str) -> str:
    """
    Make HF token strings more human-readable.
    - GPT-2 BPE uses 'Ġ' to indicate a leading space.
    - SentencePiece often uses '▁' for a leading space.
    """
    if tok in ("[CLS]", "[SEP]", "[PAD]", "[MASK]"):
        return tok

    # GPT-2 / RoBERTa-style BPE space marker
    if tok.startswith("Ġ"):
        tok = " " + tok[1:]

    # SentencePiece space marker (e.g., LLaMA)
    if tok.startswith("▁"):
        tok = " " + tok[1:]

    return tok


def _pretty_tokens(tokens: List[str]) -> List[str]:
    return [_pretty_token(t) for t in tokens]





# ---------- Minimal UI schema (same style as other plugins) ----------
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


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _get_final_norm_module(model) -> Optional[torch.nn.Module]:
    """
    Best-effort: return the module that acts like the model's final normalization
    (applied before lm_head), or None if unknown.

    Works for many common HF causal LMs:
    - GPT-2 style: model.transformer.ln_f
    - LLaMA/Mistral style: model.model.norm
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f

    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm

    # common fallbacks
    for attr in ["ln_f", "final_layer_norm", "norm"]:
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, torch.nn.Module):
                return mod
    return None


class LogitLens(ToolkitPlugin):
    """
    HF-native Logit Lens for causal LMs.

    For a chosen token position, for each layer hidden state:
        hidden -> (optional final norm) -> lm_head -> vocab logits -> top-k tokens

    Returns JSON-friendly outputs for Streamlit visualization.
    """

    id = "logit_lens"
    name = "Logit Lens (HF-native) — layer-wise vocab projection"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (causal LM)",
                type="text",
                help="Example: gpt2, EleutherAI/gpt-neo-125M, meta-llama/... (if available).",
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                help="We will compute the logit lens for this text (no generation required).",
            ),
            FieldSpec(
                key="max_length",
                label="Max input length",
                type="number",
                required=False,
                help="Input will be truncated to this many tokens.",
            ),
            FieldSpec(
                key="position_mode",
                label="Token position to inspect",
                type="select",
                options=["last", "index"],
                help="Inspect the last token, or a specific 0-based token index.",
            ),
            FieldSpec(
                key="position_index",
                label="Position index (used only if position_mode=index)",
                type="number",
                required=False,
                help="0-based index into the tokenized input.",
            ),
            FieldSpec(
                key="top_k",
                label="Top-k tokens per layer",
                type="number",
                required=False,
                help="How many highest scoring tokens to show per layer (e.g., 5–50).",
            ),
            FieldSpec(
                key="score_type",
                label="Score type",
                type="select",
                options=["prob", "logit"],
                help="Probability is more interpretable; logits are raw pre-softmax scores.",
            )
        ]

    def _load_model(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        cfg = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Ensure pad token exists for some tokenizers (helps truncation/padding stability)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg)
        model.to(self.device)
        model.eval()

        bundle = {"config": cfg, "tokenizer": tokenizer, "model": model}
        self._cache[model_name] = bundle
        return bundle

    @torch.no_grad()
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "gpt2"
        text = (inputs.get("text") or "").strip()
        if not text:
            raise ValueError("Input text is empty.")

        max_length = _to_int(inputs.get("max_length", 128), 128)
        max_length = max(8, min(max_length, 2048))

        position_mode = inputs.get("position_mode") or "last"
        position_index = _to_int(inputs.get("position_index", 0), 0)

        top_k = _to_int(inputs.get("top_k", 10), 10)
        top_k = max(1, min(top_k, 50))

        score_type = inputs.get("score_type") or "prob"
        #apply_final_norm = bool(inputs.get("apply_final_norm", True))

        bundle = self._load_model(model_name)
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
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # output_hidden_states gives a tuple: (embeddings_out, layer1_out, ..., layerN_out)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states; cannot compute logit lens.")

        tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        tokens_display = _pretty_tokens(tokens_raw)
        seq_len = len(tokens_raw)

  
        if position_mode == "last":
            pos = seq_len - 1
        else:
            pos = int(position_index)

        if pos < 0 or pos >= seq_len:
            raise ValueError(f"Position {pos} is out of range for sequence length {seq_len}.")

        lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("Model has no lm_head / output embeddings; not a causal LM?")

        ##final_norm = _get_final_norm_module(model) if apply_final_norm else None
        final_norm = _get_final_norm_module(model)  # auto-detect


        layers_out: List[Dict[str, Any]] = []

        # We'll also track the final layer top-1 token probability across layers
        final_layer_logits = None
        final_top_id = None

        # Precompute all logits for top-k extraction
        for layer_idx, hs in enumerate(hidden_states):
            # hs: [B, T, D]
            x = hs[:, pos, :]  # [B, D]
            if final_norm is not None and layer_idx != (len(hidden_states) - 1):
                    x = final_norm(x)

            logits = lm_head(x)[0]  # [V]
            if layer_idx == len(hidden_states) - 1:
                final_layer_logits = logits
                final_top_id = int(torch.argmax(logits).item())

            if score_type == "prob":
                probs = F.softmax(logits, dim=-1)
                vals, idxs = torch.topk(probs, k=top_k)
                #top = [{"token": tokenizer.convert_ids_to_tokens(int(i)), "score": float(v)} for v, i in zip(vals, idxs)]
                top = [{"token": _pretty_token(tokenizer.convert_ids_to_tokens(int(i))), "score": float(v)} for v, i in zip(vals, idxs)]

            else:
                vals, idxs = torch.topk(logits, k=top_k)
                #top = [{"token": tokenizer.convert_ids_to_tokens(int(i)), "score": float(v)} for v, i in zip(vals, idxs)]
                top = [{"token": _pretty_token(tokenizer.convert_ids_to_tokens(int(i))), "score": float(v)} for v, i in zip(vals, idxs)]


            layers_out.append({"layer": layer_idx, "top": top})

        tracked_probs: Optional[List[float]] = None
        tracked_token: Optional[Dict[str, Any]] = None

        if final_top_id is not None:
            # Probability of the final-layer top token across layers:
            # p = exp(logit[token] - logsumexp(logits))
            tracked_probs = []
            for layer_idx, hs in enumerate(hidden_states):
                x = hs[:, pos, :]  # [B, D]
                # Auto-faithful normalization:
                # Apply the model's final norm for intermediate layers, but do NOT apply it
                # to the final layer to avoid double-normalization artifacts.
                if final_norm is not None and layer_idx != (len(hidden_states) - 1):
                    x = final_norm(x)

                logits = lm_head(x)[0]
                logp = logits[final_top_id] - torch.logsumexp(logits, dim=-1)
                tracked_probs.append(float(torch.exp(logp).item()))

            tracked_token = {
                "id": int(final_top_id),
                "token": _pretty_token(tokenizer.convert_ids_to_tokens(int(final_top_id))),
            }

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,

            "tokens_raw": tokens_raw,
            "tokens": tokens_display,   # keep "tokens" as the pretty one for app simplicity
            "position": int(pos),

            "top_k": int(top_k),
            "score_type": score_type,
            "normalization_mode": "auto_faithful",
            "final_norm_detected": bool(final_norm is not None),
            "layers": layers_out,
            "tracked_token": tracked_token,
            "tracked_probs": tracked_probs,
        }
