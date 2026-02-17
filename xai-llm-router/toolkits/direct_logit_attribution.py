# toolkits/direct_logit_attribution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Token prettification (same as your LogitLens)
# -------------------------
def _pretty_token(tok: str) -> str:
    if tok in ("[CLS]", "[SEP]", "[PAD]", "[MASK]"):
        return tok
    if tok.startswith("Ġ"):
        tok = " " + tok[1:]
    if tok.startswith("▁"):
        tok = " " + tok[1:]
    return tok


def _pretty_tokens(tokens: List[str]) -> List[str]:
    return [_pretty_token(t) for t in tokens]


# ---------- Minimal UI schema ----------
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


def _get_unembedding_matrix(model) -> torch.Tensor:
    """
    Return W_U as [d_model, vocab], so hidden @ W_U gives logits.
    """
    lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise ValueError("Model has no lm_head / output embeddings; not a causal LM?")
    W = lm_head.weight  # typically [vocab, d_model]
    if W.dim() != 2:
        raise ValueError(f"Unexpected lm_head.weight shape: {tuple(W.shape)}")
    return W.t().contiguous()


def _get_lm_bias(model) -> Optional[torch.Tensor]:
    lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    if lm_head is None:
        return None
    b = getattr(lm_head, "bias", None)
    if b is None:
        return None
    return b.detach()


def _infer_arch_and_layers(model) -> Tuple[str, List[torch.nn.Module]]:
    """
    Identify common HF decoder transformer stacks and return (arch_name, layers_list).
    Supports:
      - GPT-2 style: model.transformer.h
      - LLaMA/Mistral style: model.model.layers
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2_style", list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama_style", list(model.model.layers)
    return "unknown", []


class DirectLogitAttribution(ToolkitPlugin):
    """
    Direct Logit Attribution (DLA) for causal LMs.

    For a chosen token position and target token, compute per-component contributions
    to the target logit via dot(component_output, W_U[:, target_id]).

    Components included:
      - layer_i.attn_out  (post projection output of attention module if available)
      - layer_i.mlp_out   (post projection output of MLP module if available)
      - optional lm_head.bias
    """

    id = "direct_logit_attribution"
    name = "Direct Logit Attribution — component contributions to a target logit"

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
                help="We compute DLA for this text (no generation required).",
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
                key="target_mode",
                label="Target token",
                type="select",
                options=["predicted_next", "manual_token"],
                help="Attribute contributions to the model's predicted next token, or a manual single token.",
            ),
            FieldSpec(
                key="target_token",
                label="Manual target token (single-token string)",
                type="text",
                required=False,
                help="Used only if target_mode=manual_token. Must tokenize to exactly one token.",
            ),
            FieldSpec(
                key="include_bias",
                label="Include lm_head bias (if present)",
                type="checkbox",
                required=False,
                help="Some models have an lm_head.bias; include it as its own component.",
            ),
            FieldSpec(
                key="top_n",
                label="Show top-N components",
                type="number",
                required=False,
                help="How many components to return (sorted by absolute contribution).",
            ),
        ]

    def _load_model(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        cfg = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

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
        max_length = max(8, min(max_length, 4096))

        position_mode = inputs.get("position_mode") or "last"
        position_index = _to_int(inputs.get("position_index", 0), 0)

        target_mode = inputs.get("target_mode") or "predicted_next"
        target_token_str_in = (inputs.get("target_token") or "").strip()

        include_bias = bool(inputs.get("include_bias", True))
        top_n = _to_int(inputs.get("top_n", 50), 50)
        top_n = max(5, min(top_n, 500))

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

        tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        tokens_display = _pretty_tokens(tokens_raw)
        seq_len = len(tokens_raw)

        if position_mode == "last":
            pos = seq_len - 1
        else:
            pos = int(position_index)

        if pos < 0 or pos >= seq_len:
            raise ValueError(f"Position {pos} is out of range for sequence length {seq_len}.")

        # -------------------------
        # Hook attention + MLP outputs at the chosen position
        # -------------------------
        arch, layers = _infer_arch_and_layers(model)
        if not layers:
            raise ValueError(
                "Unsupported model architecture for DLA hooks. "
                "Currently supports GPT-2 style (model.transformer.h) and LLaMA/Mistral style (model.model.layers)."
            )

        captured_attn: List[torch.Tensor] = []
        captured_mlp: List[torch.Tensor] = []
        handles: List[Any] = []

        def _make_hook(kind: str):
            def hook_fn(mod, inp, out):
                out_t = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(out_t) or out_t.dim() != 3:
                    return
                vec = out_t[0, pos, :].detach()
                if kind == "attn":
                    captured_attn.append(vec)
                else:
                    captured_mlp.append(vec)
            return hook_fn

        # Attach hooks
        for block in layers:
            if arch == "gpt2_style":
                if hasattr(block, "attn"):
                    handles.append(block.attn.register_forward_hook(_make_hook("attn")))
                if hasattr(block, "mlp"):
                    handles.append(block.mlp.register_forward_hook(_make_hook("mlp")))
            elif arch == "llama_style":
                if hasattr(block, "self_attn"):
                    handles.append(block.self_attn.register_forward_hook(_make_hook("attn")))
                if hasattr(block, "mlp"):
                    handles.append(block.mlp.register_forward_hook(_make_hook("mlp")))

        # Forward pass (we also want logits at pos)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

        # Remove hooks
        for h in handles:
            h.remove()

        logits = out.logits  # [B, T, V]
        if logits is None:
            raise ValueError("Model did not return logits; cannot compute DLA.")

        pred_next_id = int(torch.argmax(logits[0, pos, :]).item())
        pred_next_tok = _pretty_token(tokenizer.convert_ids_to_tokens(pred_next_id))

        # Determine target token id
        if target_mode == "predicted_next":
            target_id = pred_next_id
            target_tok = pred_next_tok
        else:
            if not target_token_str_in:
                raise ValueError("target_mode=manual_token but target_token is empty.")
            enc_t = tokenizer.encode(target_token_str_in, add_special_tokens=False)
            if len(enc_t) != 1:
                raise ValueError(
                    f"Manual target token must map to exactly 1 token id, got {enc_t} (len={len(enc_t)})."
                )
            target_id = int(enc_t[0])
            target_tok = _pretty_token(tokenizer.convert_ids_to_tokens(target_id))

        # Unembedding direction u = W_U[:, target_id]
        W_U = _get_unembedding_matrix(model).to(self.device)  # [d_model, vocab]
        u = W_U[:, target_id].detach()

        # Sanity: compute total logit from logits tensor
        total_logit = float(logits[0, pos, target_id].item())

        # Component contributions
        rows: List[Dict[str, Any]] = []

        # Attention contributions (layer order = hook order)
        for i, vec in enumerate(captured_attn):
            rows.append(
                {
                    "component": f"layer_{i:02d}.attn_out",
                    "type": "attn",
                    "layer": i,
                    "contribution": float((vec @ u).item()),
                }
            )

        # MLP contributions
        for i, vec in enumerate(captured_mlp):
            rows.append(
                {
                    "component": f"layer_{i:02d}.mlp_out",
                    "type": "mlp",
                    "layer": i,
                    "contribution": float((vec @ u).item()),
                }
            )

        # Optional lm_head bias
        if include_bias:
            bias = _get_lm_bias(model)
            if bias is not None:
                rows.append(
                    {
                        "component": "lm_head.bias",
                        "type": "bias",
                        "layer": None,
                        "contribution": float(bias[target_id].item()),
                    }
                )

        # Sort by absolute contribution and keep top_n
        rows_sorted = sorted(rows, key=lambda r: abs(r["contribution"]), reverse=True)[:top_n]

        # Compute abs + share of abs (for UI)
        abs_sum = sum(abs(r["contribution"]) for r in rows_sorted) or 1.0
        for r in rows_sorted:
            r["abs_contribution"] = float(abs(r["contribution"]))
            r["share_abs_%"] = float(100.0 * abs(r["contribution"]) / abs_sum)
            r["sign"] = "+" if r["contribution"] >= 0 else "-"

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "arch_detected": arch,

            "tokens_raw": tokens_raw,
            "tokens": tokens_display,
            "position": int(pos),

            "predicted_next": {"id": pred_next_id, "token": pred_next_tok},
            "target": {"id": int(target_id), "token": target_tok, "mode": target_mode},

            "total_logit": float(total_logit),
            "include_bias": bool(include_bias),

            "top_n": int(top_n),
            "components": rows_sorted,
            "notes": [
                "DLA is linear: contribution = dot(component_output, unembedding_direction).",
                "This is not causal: it ignores softmax coupling, layernorm interactions, and other nonlinear effects.",
                "Hook points are module outputs (attn module and mlp module); naming/semantics vary by architecture.",
            ],
        }
