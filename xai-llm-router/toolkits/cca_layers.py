from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

from svcca_lib import cca_core


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None


def _svd_reduce_neurons(acts: np.ndarray, max_neurons: int) -> np.ndarray:
    """
    Reduce (neurons, samples) -> (k, samples) with k<=max_neurons using SVD on centered activations.

    IMPORTANT:
      - This is NOT just to satisfy cca_core's assert.
      - Choosing k << samples is necessary to avoid degenerate "all ones" CCA.
    """
    acts = np.asarray(acts, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError("acts must be 2D (neurons, samples).")

    n_neurons, _n_samples = acts.shape
    if max_neurons < 1:
        raise ValueError("max_neurons must be >= 1.")

    if n_neurons <= max_neurons:
        return acts

    # Center per neuron
    x = acts - np.mean(acts, axis=1, keepdims=True)

    # SVD: x = U S Vt; top-k representation over samples is S_k * Vt_k
    # This yields (k, samples)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    k = int(max_neurons)
    return (s[:k, None] * vt[:k, :])


def cca_ecco_style_mean(
    acts1: np.ndarray,
    acts2: np.ndarray,
    *,
    epsilon: float = 1e-10,
    cca_dims: int = 10,
) -> float:
    """
    Ecco-style scalar CCA score: cca_core.get_cca_similarity(...)[\"mean\"][0]

    Key fix vs your previous version:
      - DO NOT reduce to (samples-1). That makes CCA trivial (~all 1s).
      - Instead use SVCCA-style reduction to a fixed k = min(cca_dims, samples-1).
    """
    acts1 = np.asarray(acts1, dtype=np.float64)
    acts2 = np.asarray(acts2, dtype=np.float64)

    if acts1.shape[1] != acts2.shape[1]:
        raise ValueError("acts1 and acts2 must have the same number of samples (token positions).")

    n_samples = acts1.shape[1]
    if n_samples < 3:
        raise ValueError("CCA needs at least 3 token samples to be meaningful.")

    # Choose k small enough to be informative, but also satisfy cca_core (neurons < samples)
    k = int(max(1, min(int(cca_dims), n_samples - 1)))

    a1 = _svd_reduce_neurons(acts1, max_neurons=k)
    a2 = _svd_reduce_neurons(acts2, max_neurons=k)

    # cca_core asserts neurons < datapoints
    if not (a1.shape[0] < a1.shape[1]):
        raise ValueError(
            f"SVCCA CCA requires neurons < samples, got {a1.shape[0]} >= {a1.shape[1]}. "
            f"Increase token samples or lower cca_dims."
        )

    res = cca_core.get_cca_similarity(a1, a2, epsilon=epsilon, verbose=False)
    return float(res["mean"][0])


class CCALayers:
    id = "cca_layers"
    name = "CCA similarity across layers"

    def __init__(self):
        self._model: HookedTransformer | None = None
        self._model_name: str | None = None

    def spec(self):
        return [
            FieldSpec(
                key="model_name",
                label="Model (TransformerLens name)",
                type="text",
                default="gpt2-small",
                help="Example: gpt2-small, opt-125m, etc.",
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                default="Machine learning models learn hierarchical representations of text. "
                        "Early layers often capture lexical or syntactic features. "
                        "Middle layers may represent phrasal structure and compositional semantics, "
                        "while deeper layers increasingly encode task-relevant abstractions.",
                help="Use a paragraph (>=30-50 tokens) for stable CCA.",
            ),
            FieldSpec(
                key="max_length",
                label="Max tokens",
                type="number",
                default=128,
                help="Manual truncation length (TransformerLens to_tokens() kwarg compatibility fix).",
            ),
            FieldSpec(
                key="token_subset",
                label="Token subset",
                type="select",
                options=["all", "last", "first", "middle"],
                default="all",
                help="Which token positions to use as samples for CCA. Prefer 'all'.",
            ),
            FieldSpec(
                key="max_tokens_used",
                label="Max tokens used (downsample)",
                type="number",
                default=64,
                help="If sequence is long, cap token samples deterministically.",
            ),
            FieldSpec(
                key="cca_dims",
                label="CCA dims (SVD reduce before CCA)",
                type="number",
                default=10,
                help="SVCCA-style: reduce neuron dimension to k before CCA. Too large can saturate; try 20–100.",
            ),
            FieldSpec(
                key="epsilon",
                label="CCA epsilon (stability)",
                type="number",
                default=1e-10,
                help="Small diagonal stabilizer passed to cca_core.",
            ),
            FieldSpec(
                key="compute_on_cpu",
                label="Compute on CPU",
                type="checkbox",
                default=True,
                help="Safer for Streamlit deployments.",
            ),
        ]

    def _load_model(self, model_name: str):
        if self._model is None or self._model_name != model_name:
            self._model = HookedTransformer.from_pretrained(model_name)
            self._model_name = model_name

    @torch.no_grad()
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = str(inputs.get("model_name", "gpt2-small"))
        text = str(inputs.get("text", ""))
        max_length = int(float(inputs.get("max_length", 128)))
        token_subset = str(inputs.get("token_subset", "all"))
        max_tokens_used = int(float(inputs.get("max_tokens_used", 64)))
        cca_dims = int(float(inputs.get("cca_dims", 50)))
        epsilon = float(inputs.get("epsilon", 1e-10))
        compute_on_cpu = bool(inputs.get("compute_on_cpu", True))

        self._load_model(model_name)
        assert self._model is not None

        device = torch.device("cpu") if compute_on_cpu else self._model.cfg.device
        self._model.to(device)

        toks = self._model.to_tokens(text).to(device)  # (batch, seq)

        # Manual truncation
        if max_length and toks.shape[1] > max_length:
            toks = toks[:, :max_length]

        _, cache = self._model.run_with_cache(toks, remove_batch_dim=False)

        # Build reps: emb + resid_post per layer
        reps: List[torch.Tensor] = []
        reps.append(cache["hook_embed"][0])  # (seq, d_model)

        n_layers = int(self._model.cfg.n_layers)
        for l in range(n_layers):
            reps.append(cache[f"blocks.{l}.hook_resid_post"][0])  # (seq, d_model)

        tokens_str = self._model.to_str_tokens(toks[0])
        seq_len = len(tokens_str)

        # Token indices
        if token_subset == "last":
            idxs = [seq_len - 1]
        elif token_subset == "first":
            idxs = [0]
        elif token_subset == "middle":
            idxs = [seq_len // 2]
        else:
            idxs = list(range(seq_len))

        # Downsample deterministically if needed
        if len(idxs) > max_tokens_used and max_tokens_used > 0:
            step = max(1, len(idxs) // max_tokens_used)
            idxs = idxs[::step][:max_tokens_used]

        if len(idxs) < 3:
            raise ValueError("CCA needs at least 3 token samples. Use token_subset=all and longer text/max_length.")

        # Convert to (neurons, samples)
        rep_mats: List[np.ndarray] = []
        for r in reps:
            r_sel = r[idxs, :]  # (samples, d_model)
            rep_mats.append(r_sel.detach().float().cpu().numpy().T)  # (d_model, samples)

        L = len(rep_mats)
        M = np.zeros((L, L), dtype=np.float64)

        # Effective k used (bounded by samples-1)
        effective_k = int(max(1, min(cca_dims, len(idxs) - 1)))

        for i in range(L):
            M[i, i] = 1.0
            for j in range(i + 1, L):
                s = cca_ecco_style_mean(rep_mats[i], rep_mats[j], epsilon=epsilon, cca_dims=cca_dims)
                M[i, j] = s
                M[j, i] = s

        labels = ["emb"] + [f"L{l}" for l in range(n_layers)]

        return {
            "plugin": self.id,
            "model": model_name,
            "arch_used": "decoder",
            "tokens": tokens_str,
            "token_indices_used": idxs,
            "layer_labels": labels,
            "cca_matrix": M.tolist(),
            "params": {
                "token_subset": token_subset,
                "max_tokens_used": max_tokens_used,
                "max_length": max_length,
                "compute_on_cpu": compute_on_cpu,
                "epsilon": epsilon,
                "cca_dims": cca_dims,
                "svd_reduce_to": effective_k,
                "n_token_samples": len(idxs),
            },
        }