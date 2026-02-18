
# toolkits/sae_feature_explorer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE


# ---------- Minimal UI schema (same pattern as your other plugins) ----------
@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""


def _pretty_token(tok: str) -> str:
    # transformer-lens style tokens: "Ġ" indicates leading space
    if tok.startswith("Ġ"):
        return " " + tok[1:]
    if tok.startswith("▁"):
        return " " + tok[1:]
    return tok


class SAEFeatureExplorer:
    id = "sae_feature_explorer"
    name = "Sparse Autoenconders (SAELens + Neuronpedia)"

    RELEASE_OPTIONS = [
        "gpt2-small-res-jb",
        "gemma-scope-2b-pt-res-canonical",
    ]

    def spec(self) -> List[FieldSpec]:
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.append("cuda")

        return [
            FieldSpec(
                "model_name",
                "TransformerLens model name",
                "text",
                help="E.g. gpt2-small. Must match the SAE release you load.",
            ),
            FieldSpec(
                "release",
                "SAE release",
                "select",
                options=self.RELEASE_OPTIONS,
                help="SAELens 'release' string used by SAE.from_pretrained.",
            ),
            FieldSpec(
                "sae_id",
                "SAE id / hook point",
                "text",
                help='E.g. "blocks.6.hook_resid_pre" for GPT-2 residual stream SAEs.',
            ),
            FieldSpec(
                "dtype",
                "Compute dtype",
                "select",
                options=["float32", "bfloat16", "float16"],
                help="float32 is safest; bf16 is a good speed/memory tradeoff on modern GPUs.",
            ),
            FieldSpec(
                "device",
                "Device",
                "select",
                options=device_options,
                help="Use cuda if available.",
            ),
            FieldSpec(
                "text",
                "Input text",
                "textarea",
                help="We will run the model, take the SAE hook activations, and encode them to sparse features.",
            ),
            FieldSpec(
                "position_index",
                "Token position (-1 = last token)",
                "number",
                help="Which token position to inspect (0-based). -1 means last token.",
            ),
            FieldSpec(
                "top_k",
                "Top-k features to show",
                "number",
                help="How many SAE features to display.",
            ),
            FieldSpec(
                "per_token",
                "Show per-token top features",
                "checkbox",
                required=False,
                help="If checked, returns top features for each token position (can be slower).",
            ),
        ]
    

    # --- robust loader: SAELens versions sometimes return (sae, cfg, sparsity) vs just sae ---
    def _load_sae(self, release: str, sae_id: str, device: str, dtype: str) -> SAE:
        obj = SAE.from_pretrained(release=release, sae_id=sae_id, device=device, dtype=dtype)
        if isinstance(obj, tuple):
            sae = obj[0]
        else:
            sae = obj
        return sae

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = str(inputs["model_name"]).strip()
        release = str(inputs["release"]).strip()
        sae_id = str(inputs["sae_id"]).strip()

        dtype_str = str(inputs.get("dtype", "float32"))
        device = str(inputs.get("device", "cpu"))
        text = str(inputs.get("text", ""))

        pos = int(float(inputs.get("position_index", -1)))
        top_k = int(float(inputs.get("top_k", 10)))
        per_token = bool(inputs.get("per_token", False))

        if not text.strip():
            raise ValueError("Please provide some input text.")

        # Map dtype
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        torch_dtype = dtype_map.get(dtype_str, torch.float32)

        # Load model + SAE
        model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch_dtype)
        sae = self._load_sae(release=release, sae_id=sae_id, device=device, dtype=dtype_str)

        tokens = model.to_tokens(text)
        #str_toks = [_pretty_token(t) for t in model.to_str_tokens(tokens)[0]]

        print(" tokens: ", tokens)

        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
            

        str_tokens = model.to_str_tokens(tokens)

        print(" str_tokens: ", str_tokens)

        # str_tokens can be either:
        # - List[List[str]] (batch, seq)
        # - List[str]       (seq)
        if len(str_tokens) > 0 and isinstance(str_tokens[0], list):
            tok_list = str_tokens[0]
        else:
            tok_list = str_tokens

        str_toks = [_pretty_token(t) for t in tok_list]

        # Run and cache
        _, cache = model.run_with_cache(tokens)

        # SAE hook naming for TransformerLens cache:
        # for GPT2 residual stream, cache key is like cache["resid_pre", layer]
        # sae_id is like "blocks.6.hook_resid_pre"
        # We'll parse layer and stream from sae_id for GPT-2-like hook names.
        # If parsing fails, we fallback to direct cache key = sae.cfg.hook_name when possible.
        acts = None
        parsed = False
        try:
            # "blocks.6.hook_resid_pre" -> layer=6, stream="resid_pre"
            parts = sae_id.split(".")
            layer = int(parts[1])
            hook = parts[2].replace("hook_", "")  # resid_pre
            acts = cache[hook, layer]  # [1, seq, d_model]
            parsed = True
        except Exception:
            pass

        if acts is None:
            # last resort: try sae.cfg.hook_name if present
            hook_name = getattr(getattr(sae, "cfg", None), "hook_name", None)
            if hook_name:
                # hook_name is often exactly like sae_id; try same parsing on cfg.hook_name
                parts = str(hook_name).split(".")
                layer = int(parts[1])
                hook = parts[2].replace("hook_", "")
                acts = cache[hook, layer]
                parsed = True

        if acts is None:
            raise RuntimeError(
                "Could not locate activations in cache for this SAE id. "
                "Try a GPT-2-style sae_id like blocks.<L>.hook_resid_pre."
            )

        # Encode: [batch, pos, d_sae]
        with torch.no_grad():
            feat_acts = sae.encode(acts)

        seq_len = feat_acts.shape[1]
        if pos < 0:
            pos = seq_len + pos
        pos = max(0, min(seq_len - 1, pos))

        v = feat_acts[0, pos]  # [d_sae]
        top = torch.topk(v, k=min(top_k, v.shape[0]))

        top_features = [
            {"feature_id": int(i), "activation": float(a)}
            for i, a in zip(top.indices.detach().cpu().tolist(), top.values.detach().cpu().tolist())
        ]

        per_token_rows = []
        if per_token:
            k = min(5, v.shape[0])  # fixed small per-token k
            for tpos in range(seq_len):
                vv = feat_acts[0, tpos]
                tt = torch.topk(vv, k=k)
                per_token_rows.append(
                    {
                        "pos": tpos,
                        "token": str_toks[tpos],
                        "top_features": [
                            {"feature_id": int(i), "activation": float(a)}
                            for i, a in zip(tt.indices.detach().cpu().tolist(), tt.values.detach().cpu().tolist())
                        ],
                    }
                )

        return {
            "plugin": self.id,
            "model": model_name,
            "release": release,
            "sae_id": sae_id,
            "parsed_cache_key": parsed,
            "text": text,
            "tokens": str_toks,
            "position": pos,
            "top_k": top_k,
            "top_features": top_features,
            "per_token": per_token,
            "per_token_top": per_token_rows,
            "notes": [
                "These are SAE feature activations on the selected hook (sparse code).",
                "A 'feature' is one learned direction; larger activation => stronger presence at that token position.",
            ],
        }
