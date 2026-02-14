# router_scoring.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

# sentence-transformers is lightweight and fast for CPU embeddings
from sentence_transformers import SentenceTransformer


# -------------------------
# Label descriptions / prompts
# -------------------------
LABEL_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "task": {
        "NA": "The task is not specified or not relevant.",
        "classification": "The user wants to assign labels or categories to text, such as sentiment or topic classification.",
        "generation": "The user wants the model to generate or write new text, such as explanations, summaries, or stories.",
        "seq2seq": "The user wants to transform one sequence into another, such as translation, summarization, or paraphrasing.",
        "QA": "The user wants to answer questions based on text, documents, or knowledge sources.",
        "NER": "The user wants to extract structured entities like names, locations, or organizations from text.",
        "RAG": "The user wants retrieval augmented generation, combining document retrieval with text generation.",
        "agents": "The user wants an agent that plans, reasons step by step, or uses tools and actions.",
        "general_NLP": "The user has general NLP analysis or explanation needs not tied to a single task.",
        "multimodal": "The user works with text together with images, audio, video, or other modalities.",
    },
    "access": {
        "NA": "The level of access to the model is not specified.",
        "black_box": "The method only requires input-output access to the model, without internal information.",
        "gray_box": "The method requires partial internal access such as logits, gradients, or attention weights.",
        "white_box": "The method requires full access to internal weights, activations, or computation graphs.",
        "mixed": "The method supports or combines multiple access levels depending on configuration.",
    },
    "arch": {
        "NA": "The model architecture is not specified.",
        "encoder": "The model is encoder-only, such as BERT-style architectures.",
        "decoder": "The model is decoder-only, such as GPT-style language models.",
        "encdec": "The model is encoder-decoder, such as T5 or BART.",
        "transformer_general": "The method applies broadly to transformer architectures regardless of exact type.",
    },
    "scope": {
        "NA": "The explanation scope is not specified.",
        "local": "The explanation focuses on individual predictions or specific examples.",
        "global": "The explanation focuses on overall model behavior or general patterns.",
        "both": "The explanation supports both local and global analysis.",
    },
    "granularity": {
        "NA": "The explanation granularity is not specified.",
        "token": "The explanation focuses on individual tokens or words.",
        "span": "The explanation focuses on spans or phrases of text.",
        "sentence": "The explanation focuses on whole sentences.",
        "document": "The explanation focuses on entire documents.",
        "example": "The explanation compares or analyzes whole input examples.",
        "concept": "The explanation focuses on high-level semantic concepts.",
        "neuron": "The explanation analyzes individual neurons or units inside the model.",
        "head": "The explanation focuses on attention heads.",
        "layer": "The explanation focuses on entire layers of the model.",
        "circuit": "The explanation focuses on circuits or interactions between components.",
        "dataset": "The explanation focuses on dataset-level patterns or statistics.",
        "component_graph": "The explanation represents the model as a computational or causal graph.",
    },
    "goal": {
        "NA": "The goal or target audience is not specified.",
        "research_debug": "The goal is debugging or understanding model behavior for research purposes.",
        "mech_interp": "The goal is mechanistic interpretability and understanding internal computations.",
        "model_eval": "The goal is evaluating model quality, robustness, or performance.",
        "fairness_audit": "The goal is auditing bias, fairness, or ethical properties.",
        "end_user_explain": "The goal is explaining model decisions to non-expert end users.",
        "general_tooling": "The goal is building general-purpose explanation or analysis tools.",
    },
    "fidelity": {
        "NA": "The fidelity of the explanation is not specified.",
        "high": "The explanation is faithful to the true internal behavior of the model.",
        "medium": "The explanation balances faithfulness and interpretability.",
        "low": "The explanation prioritizes simplicity over faithfulness.",
        "mixed": "The method provides explanations with varying fidelity levels.",
    },
    "format": {
        "NA": "The explanation format is not specified.",
        "visual_UI": "The explanation is presented through an interactive visual user interface.",
        "notebook_viz": "The explanation is shown through plots or visualizations in notebooks.",
        "text_rationale": "The explanation is provided as natural language text.",
        "rules": "The explanation is provided as rules or symbolic logic.",
        "ranked_examples": "The explanation is given through ranked or retrieved examples.",
        "metrics": "The explanation is summarized through numerical metrics or scores.",
        "API_only": "The method exposes explanations only through an API.",
        "interactive_dialogue": "The explanation is delivered through an interactive dialogue with the user.",
    },
}


# -------------------------
# Small helpers
# -------------------------
def norm_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def softmax(x: np.ndarray, temperature: float = 0.7) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    z = (x / t) - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


# -------------------------
# Embedder (cached manually)
# -------------------------
# We avoid Streamlit caching here so the module is pure Python.
# You can still cache at the Streamlit level in app.py if you want.
_EMBEDDER: Optional[SentenceTransformer] = None
_EMBEDDER_NAME: Optional[str] = None

# Cache label embeddings per (model_name, dim, dim_values_tuple)
_LABEL_EMB_CACHE: Dict[Tuple[str, str, Tuple[str, ...]], np.ndarray] = {}


def get_embedder(model_name: str) -> SentenceTransformer:
    global _EMBEDDER, _EMBEDDER_NAME
    if _EMBEDDER is None or _EMBEDDER_NAME != model_name:
        _EMBEDDER = SentenceTransformer(model_name)
        _EMBEDDER_NAME = model_name
    return _EMBEDDER


def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    model = get_embedder(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def build_label_prompts(dim: str, dim_values: List[str]) -> List[str]:
    desc_map = LABEL_DESCRIPTIONS.get(dim, {})
    prompts = []
    for v in dim_values:
        desc = desc_map.get(v, f"The label is {v}.")
        prompts.append(f"For the user's request: {desc}")
    return prompts


def predict_dim_probs(
    user_text: str,
    dim: str,
    dim_values: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    temperature: float = 0.7,
) -> Dict[str, float]:
    """
    Return {label: probability} for a single dimension.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        # If no text, return uniform distribution
        p = 1.0 / max(len(dim_values), 1)
        return {v: p for v in dim_values}

    # Cache label embeddings so we only embed label prompts once.
    key = (model_name, dim, tuple(dim_values))
    if key in _LABEL_EMB_CACHE:
        label_embs = _LABEL_EMB_CACHE[key]
    else:
        label_prompts = build_label_prompts(dim, dim_values)
        label_embs = embed_texts(model_name, label_prompts)
        _LABEL_EMB_CACHE[key] = label_embs

    user_emb = embed_texts(model_name, [user_text])[0]
    sims = label_embs @ user_emb  # cosine similarities (normalized)
    probs = softmax(sims, temperature=temperature)
    return {v: float(p) for v, p in zip(dim_values, probs)}


def predict_all_probs(
    user_text: str,
    dim_values_map: Dict[str, List[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    temperature: float = 0.7,
) -> Dict[str, Dict[str, float]]:
    """
    Return {dim: {label: prob}} for all dimensions in dim_values_map.
    """
    out: Dict[str, Dict[str, float]] = {}
    for dim, values in dim_values_map.items():
        out[dim] = predict_dim_probs(
            user_text=user_text,
            dim=dim,
            dim_values=values,
            model_name=model_name,
            temperature=temperature,
        )
    return out


# -------------------------
# Soft scoring for a method using text_probs
# -------------------------
def method_text_score(
    method: Dict[str, Any],
    text_probs: Dict[str, Dict[str, float]],
) -> Tuple[float, List[str]]:
    """
    Returns (soft_score, reasons) where soft_score is in [0,1] roughly.
    We take max probability among the method's supported labels for each dimension,
    then average across dimensions that are present for that method.
    """
    if not text_probs:
        return 0.0, []

    total = 0.0
    count = 0
    reasons: List[str] = []

    # task_input
    tasks = norm_list(method.get("task_input", []))
    if tasks:
        p = max(text_probs.get("task", {}).get(t, 0.0) for t in tasks)
        total += p; count += 1
        reasons.append(f"text→task {p:.2f}")

    # access_arch.access
    accs = norm_list(method.get("access_arch", {}).get("access", "NA"))
    if accs:
        p = max(text_probs.get("access", {}).get(a, 0.0) for a in accs)
        total += p; count += 1
        reasons.append(f"text→access {p:.2f}")

    # access_arch.arch
    archs = norm_list(method.get("access_arch", {}).get("arch", "NA"))
    if archs:
        p = max(text_probs.get("arch", {}).get(a, 0.0) for a in archs)
        total += p; count += 1
        reasons.append(f"text→arch {p:.2f}")

    # target_scope
    sc = method.get("target_scope", "NA")
    if sc:
        scs = norm_list(sc)
        p = max(text_probs.get("scope", {}).get(s, 0.0) for s in scs)
        total += p; count += 1
        reasons.append(f"text→scope {p:.2f}")

    # granularity
    grans = norm_list(method.get("granularity", []))
    if grans:
        p = max(text_probs.get("granularity", {}).get(g, 0.0) for g in grans)
        total += p; count += 1
        reasons.append(f"text→gran {p:.2f}")

    # goal / audience
    goals = norm_list(method.get("user_goal_audience", []))
    if goals:
        p = max(text_probs.get("goal", {}).get(g, 0.0) for g in goals)
        total += p; count += 1
        reasons.append(f"text→goal {p:.2f}")

    # fidelity
    fid = method.get("fidelity", "NA")
    if fid:
        fids = norm_list(fid)
        p = max(text_probs.get("fidelity", {}).get(f, 0.0) for f in fids)
        total += p; count += 1
        reasons.append(f"text→fidelity {p:.2f}")

    # format
    fmts = norm_list(method.get("format", []))
    if fmts:
        p = max(text_probs.get("format", {}).get(f, 0.0) for f in fmts)
        total += p; count += 1
        reasons.append(f"text→format {p:.2f}")

    if count == 0:
        return 0.0, []

    return total / count, reasons


def rank_methods(
    methods: List[Dict[str, Any]],
    user_text: str,
    dim_values_map: Dict[str, List[str]],
    base_scores: Optional[Dict[str, float]] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    temperature: float = 0.7,
    text_weight: float = 0.5,
    soft_scale: float = 10.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Returns (ranked_methods, text_probs).

    - base_scores: optional mapping {method_name: numeric_score} (e.g., from your current discrete score()).
    - final_score = base_score + text_weight * soft_scale * soft_score
    """
    text_probs = {}
    if (user_text or "").strip():
        text_probs = predict_all_probs(
            user_text=user_text,
            dim_values_map=dim_values_map,
            model_name=model_name,
            temperature=temperature,
        )

    ranked = []
    for m in methods:
        name = m.get("name", "NA")

        base = 0.0
        if base_scores and name in base_scores:
            base = float(base_scores[name])

        soft, soft_reasons = method_text_score(m, text_probs) if text_probs else (0.0, [])
        final = base + float(text_weight) * float(soft_scale) * float(soft)

        ranked.append({
            "method": m,
            "name": name,
            "base_score": base,
            "soft_score": soft,
            "final_score": final,
            "soft_reasons": soft_reasons,
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked, text_probs
