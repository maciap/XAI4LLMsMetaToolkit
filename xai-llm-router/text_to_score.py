# router_scoring.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import hashlib
import numpy as np

from sentence_transformers import SentenceTransformer


# -------------------------
# Small helpers
# -------------------------
def norm_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _join_clean(xs: List[str]) -> str:
    return "; ".join([str(x).strip() for x in xs if str(x).strip()])


def _stable_hash(s: str) -> int:
    # stable across runs (unlike Python's built-in hash())
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:16], 16)


def _clamp01(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    # cosine sims are in [-1, 1]; this just guards numeric weirdness
    return float(max(lo, min(hi, float(x))))


# -------------------------
# Embedder (cached manually)
# -------------------------
_EMBEDDER: Optional[SentenceTransformer] = None
_EMBEDDER_NAME: Optional[str] = None

# cache text embeddings per (model_name, method_name, section, stable_text_hash)
_TEXT_EMB_CACHE: Dict[Tuple[str, str, str, int], np.ndarray] = {}


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


# -------------------------
# Text extraction (ONLY these fields)
# -------------------------
def method_sections(method: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract ONLY the textual sections used for ranking:
      - overview
      - main functionalities
      - strengths
      - limitations
    """
    desc = method.get("description", {}) or {}

    overview = str(desc.get("overview") or desc.get("summary") or "").strip()

    funcs = norm_list(desc.get("main_functionalities", []))
    funcs_txt = _join_clean(funcs)

    strengths = norm_list(method.get("strengths", []))
    strengths_txt = _join_clean(strengths)

    limitations = norm_list(method.get("limitations", []))
    limitations_txt = _join_clean(limitations)

    return {
        "overview": overview,
        "funcs": funcs_txt,
        "strengths": strengths_txt,
        "limitations": limitations_txt,
    }


def _get_cached_section_emb(
    method: Dict[str, Any],
    section_name: str,
    section_text: str,
    model_name: str,
) -> np.ndarray:
    name = method.get("name", "NA")
    key = (model_name, name, section_name, _stable_hash(section_text or ""))
    if key in _TEXT_EMB_CACHE:
        return _TEXT_EMB_CACHE[key]
    emb = embed_texts(model_name, [section_text or ""])[0]
    _TEXT_EMB_CACHE[key] = emb
    return emb


# -------------------------
# Weighted semantic scoring
# score = w1*sim(q, overview) + w2*sim(q, funcs) + w3*sim(q, strengths) - w4*sim(q, limitations)
# -------------------------
def method_semantic_score_weighted(
    method: Dict[str, Any],
    user_text: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    w1: float = 0.4,
    w2: float = 0.4,
    w3: float = 0.1,
    w4: float = 0.1,
) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Returns:
      - final_score (float)
      - sims (dict of per-section sims)
      - reasons (list of strings)
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return 0.0, {"overview": 0.0, "funcs": 0.0, "strengths": 0.0, "limitations": 0.0}, []

    u_emb = embed_texts(model_name, [user_text])[0]

    secs = method_sections(method)

    # Compute cosine sims per section (missing text => sim=0)
    sims: Dict[str, float] = {}
    for k in ("overview", "funcs", "strengths", "limitations"):
        txt = secs.get(k, "") or ""
        if not txt.strip():
            sims[k] = 0.0
            continue
        m_emb = _get_cached_section_emb(method, k, txt, model_name)
        sims[k] = _clamp01(float(m_emb @ u_emb))

    score = (
        float(w1) * sims["overview"]
        + float(w2) * sims["funcs"]
        + float(w3) * sims["strengths"]
        - float(w4) * sims["limitations"]
    )

    reasons = [
        f"sim_overview={sims['overview']:.2f} (w={w1})",
        f"sim_funcs={sims['funcs']:.2f} (w={w2})",
        f"sim_strengths={sims['strengths']:.2f} (w={w3})",
        f"sim_limitations={sims['limitations']:.2f} (w={w4}, subtract)",
        f"final={score:.3f}",
    ]

    return float(score), sims, reasons


def rank_methods(
    methods: List[Dict[str, Any]],
    user_text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    w1: float = 0.4,
    w2: float = 0.4,
    w3: float = 0.1,
    w4: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Weighted semantic ranker:
      score = w1*sim(q, overview) + w2*sim(q, main functionalities) + w3*sim(q, strengths) - w4*sim(q, limitations)

    Returns (ranked_methods, text_probs) for drop-in compatibility with your app.
    text_probs is {} because we are not predicting category distributions.
    """
    user_text = (user_text or "").strip()
    ranked: List[Dict[str, Any]] = []

    for m in methods:
        name = m.get("name", "NA")
        final, sims, reasons = method_semantic_score_weighted(
            m,
            user_text,
            model_name=model_name,
            w1=w1,
            w2=w2,
            w3=w3,
            w4=w4,
        )
        ranked.append(
            {
                "method": m,
                "name": name,
                "base_score": 0.0,
                "soft_score": final,     # weighted score
                "final_score": final,    # rank directly by it
                "soft_reasons": reasons, # includes per-section sims
                "section_sims": sims,    # optional extra debug info
            }
        )

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked, {}