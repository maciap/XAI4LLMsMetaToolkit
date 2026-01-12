import json
from typing import Any, Dict, List, Tuple

import streamlit as st

from text_to_score import rank_methods  # NEW



from toolkits.captum_ig import CaptumIGSentiment

PLUGINS = {
    CaptumIGSentiment.id: CaptumIGSentiment(),
}



# -------------------------
# Config: dimension values
# -------------------------
DIM_VALUES = {
    "task": ["NA", "classification", "generation", "seq2seq", "QA", "NER", "RAG", "agents", "general_NLP", "multimodal"],
    "access": ["NA", "black_box", "gray_box", "white_box", "mixed"],
    "arch": ["NA", "encoder", "decoder", "encdec", "transformer_general"],
    "scope": ["NA", "local", "global", "both"],
    "granularity": ["NA", "token", "span", "sentence", "document", "example", "concept", "neuron", "head", "layer", "circuit", "dataset", "component_graph"],
    "goal": ["NA", "research_debug", "mech_interp", "model_eval", "fairness_audit", "end_user_explain", "general_tooling"],
    "fidelity": ["NA", "high", "medium", "low", "mixed"],
    "format": ["NA", "visual_UI", "notebook_viz", "text_rationale", "rules", "ranked_examples", "metrics", "API_only", "interactive_dialogue"],
}

DEFAULTS = {
    "task": "general_NLP",
    "access": "NA",
    "arch": "NA",
    "scope": "NA",
    "granularity": "NA",
    "goal": "research_debug",
    "fidelity": "NA",
    "format": "NA",
}


# -------------------------
# Helpers
# -------------------------
def norm_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def load_methods(path: str = "methods.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "toolkits" in data:
        return data["toolkits"]
    if isinstance(data, list):
        return data
    raise ValueError("methods.json must be a list or a dict with key 'toolkits'.")


def feasible(user: Dict[str, str], m: Dict[str, Any]) -> Tuple[bool, str]:
    # Task
    if user["task"] != "NA":
        tasks = norm_list(m.get("task_input", []))
        if user["task"] not in tasks and "general_NLP" not in tasks:
            return False, f"task mismatch (needs {tasks or 'NA'})"

    # Access
    if user["access"] != "NA":
        acc = m.get("access_arch", {}).get("access", "NA")
        accs = norm_list(acc)
        if user["access"] not in accs and "mixed" not in accs and "NA" not in accs:
            return False, f"access mismatch (method {accs})"

    # Architecture
    if user["arch"] != "NA":
        arch = m.get("access_arch", {}).get("arch", "NA")
        archs = norm_list(arch)
        if user["arch"] not in archs and "transformer_general" not in archs and "NA" not in archs:
            return False, f"arch mismatch (method {archs})"

    # Scope
    if user["scope"] != "NA":
        sc = m.get("target_scope", "NA")
        if sc not in ["NA", "both"] and user["scope"] != sc:
            return False, f"scope mismatch (method {sc})"

    return True, ""


def score(user: Dict[str, str], m: Dict[str, Any]) -> Tuple[int, List[str]]:
    s = 0
    reasons = []

    if user["task"] != "NA" and user["task"] in norm_list(m.get("task_input", [])):
        s += 2; reasons.append("task match")

    if user["granularity"] != "NA" and user["granularity"] in norm_list(m.get("granularity", [])):
        s += 2; reasons.append("granularity match")

    if user["goal"] != "NA" and user["goal"] in norm_list(m.get("user_goal_audience", [])):
        s += 2; reasons.append("goal match")

    if user["format"] != "NA" and user["format"] in norm_list(m.get("format", [])):
        s += 2; reasons.append("format match")

    if user["fidelity"] != "NA":
        f = m.get("fidelity", "NA")
        if f == user["fidelity"]:
            s += 2; reasons.append("fidelity match")
        elif f == "mixed":
            s += 1; reasons.append("fidelity partially supported")

    return s, reasons


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="XAI Router for LLMs", layout="wide")

st.title("XAI Method Router (LLMs)")
st.caption("Pick categories OR describe your desiderata in text → get ranked feasible toolkits/methods.")

try:
    methods = load_methods("methods.json")
except Exception as e:
    st.error(f"Failed to load methods.json: {e}")
    st.stop()

with st.sidebar:
    st.header("Inputs")

    mode = st.radio(
        "Input mode",
        ["Structured (categories)", "Text (desiderata)"],
        index=0,
    )

    top_k = st.slider("Max recommendations", 5, 50, 20)

    # Prepare placeholders
    user: Dict[str, str] = {}
    user_text = ""
    excluded: List[Dict[str, Any]] = []

    if mode == "Structured (categories)":
        user["task"] = st.selectbox("Task", DIM_VALUES["task"], index=DIM_VALUES["task"].index(DEFAULTS["task"]))
        user["access"] = st.selectbox("Model access", DIM_VALUES["access"], index=DIM_VALUES["access"].index(DEFAULTS["access"]))
        user["arch"] = st.selectbox("Architecture", DIM_VALUES["arch"], index=DIM_VALUES["arch"].index(DEFAULTS["arch"]))
        user["scope"] = st.selectbox("Explanation scope", DIM_VALUES["scope"], index=DIM_VALUES["scope"].index(DEFAULTS["scope"]))
        user["granularity"] = st.selectbox("Granularity", DIM_VALUES["granularity"], index=DIM_VALUES["granularity"].index(DEFAULTS["granularity"]))
        user["goal"] = st.selectbox("Goal / audience", DIM_VALUES["goal"], index=DIM_VALUES["goal"].index(DEFAULTS["goal"]))
        user["fidelity"] = st.selectbox("Fidelity requirement", DIM_VALUES["fidelity"], index=DIM_VALUES["fidelity"].index(DEFAULTS["fidelity"]))
        user["format"] = st.selectbox("Explanation format", DIM_VALUES["format"], index=DIM_VALUES["format"].index(DEFAULTS["format"]))

        show_excluded = st.checkbox("Show excluded methods", value=True)

    else:
        user_text = st.text_area(
            "Describe your desiderata",
            placeholder=(
                "Example: I need white-box mechanistic interpretability for a transformer, "
                "focusing on attention heads and circuits; both local and global insights; "
                "prefer an interactive UI."
            ),
            height=160,
        )

        add_hard = st.checkbox("Add hard constraints too", value=False)

        # Default hard constraints to NA (so feasible() won't filter)
        user = {k: "NA" for k in DIM_VALUES.keys()}

        if add_hard:
            st.markdown("**Hard constraints (optional)**")
            user["task"] = st.selectbox("Task (hard)", DIM_VALUES["task"], index=0)
            user["access"] = st.selectbox("Model access (hard)", DIM_VALUES["access"], index=0)
            user["arch"] = st.selectbox("Architecture (hard)", DIM_VALUES["arch"], index=0)
            user["scope"] = st.selectbox("Explanation scope (hard)", DIM_VALUES["scope"], index=0)
            user["granularity"] = st.selectbox("Granularity (hard)", DIM_VALUES["granularity"], index=0)
            user["goal"] = st.selectbox("Goal / audience (hard)", DIM_VALUES["goal"], index=0)
            user["fidelity"] = st.selectbox("Fidelity requirement (hard)", DIM_VALUES["fidelity"], index=0)
            user["format"] = st.selectbox("Explanation format (hard)", DIM_VALUES["format"], index=0)

        temperature = st.slider("Text model temperature", 0.2, 1.5, 0.7, 0.05)
        show_text_prefs = st.checkbox("Show predicted preferences", value=True)

        # Excluded list doesn’t matter in text mode, but we can show it if add_hard is enabled
        show_excluded = st.checkbox("Show excluded methods", value=False)


# -------------------------
# Compute recommendations
# -------------------------
recommended: List[Dict[str, Any]] = []
excluded = []

if mode == "Structured (categories)":
    for m in methods:
        ok, why = feasible(user, m)
        if not ok:
            excluded.append({"name": m.get("name", "NA"), "why": why, "notes": m.get("notes", "")})
            continue

        sc, reasons = score(user, m)
        recommended.append({
            "name": m.get("name", "NA"),
            "score": float(sc),
            "reasons": reasons,
            "notes": m.get("notes", ""),
            "meta": {
                "scope": m.get("target_scope", "NA"),
                "access": m.get("access_arch", {}).get("access", "NA"),
                "arch": m.get("access_arch", {}).get("arch", "NA"),
                "granularity": m.get("granularity", "NA"),
                "format": m.get("format", "NA"),
                "fidelity": m.get("fidelity", "NA"),
            },
        })

    recommended.sort(key=lambda x: x["score"], reverse=True)
    text_probs = {}

else:
    if not user_text.strip():
        st.warning("Write a short description to get text-based recommendations.")
        recommended = []
        text_probs = {}
    else:
        # Apply feasible() only if the user enabled hard constraints.
        # Note: even if add_hard is off, user dict is all "NA" so it won't filter anyway.
        filtered_methods = []
        for m in methods:
            ok, why = feasible(user, m)
            if not ok:
                excluded.append({"name": m.get("name", "NA"), "why": why, "notes": m.get("notes", "")})
                continue
            filtered_methods.append(m)

        ranked, text_probs = rank_methods(
            methods=filtered_methods,
            user_text=user_text.strip(),
            dim_values_map=DIM_VALUES,
            base_scores=None,
            temperature=temperature,
            text_weight=1.0,
            soft_scale=10.0,
        )

        for item in ranked:
            m = item["method"]
            recommended.append({
                "name": item["name"],
                "score": float(item["final_score"]),
                "reasons": ["text-match"] + (item.get("soft_reasons") or []),
                "notes": m.get("notes", ""),
                "meta": {
                    "scope": m.get("target_scope", "NA"),
                    "access": m.get("access_arch", {}).get("access", "NA"),
                    "arch": m.get("access_arch", {}).get("arch", "NA"),
                    "granularity": m.get("granularity", "NA"),
                    "format": m.get("format", "NA"),
                    "fidelity": m.get("fidelity", "NA"),
                },
            })


# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader(f"Recommended ({min(top_k, len(recommended))} shown / {len(recommended)} total)")
    for item in recommended[:top_k]:
        with st.container(border=True):
            st.markdown(f"### {item['name']}")
            st.write(
                f"**Score:** {item['score']:.2f}  |  "
                f"**Why:** {', '.join(item['reasons']) if item['reasons'] else 'generic fit'}"
            )
            st.json(item["meta"], expanded=False)
            if item["notes"]:
                st.caption(item["notes"])

with col2:
    if mode == "Structured (categories)":
        st.subheader("Your selections")
        st.json(user, expanded=True)

        if show_excluded:
            st.subheader(f"Excluded ({len(excluded)})")
            for it in excluded[:50]:
                with st.container(border=True):
                    st.markdown(f"**{it['name']}**")
                    st.caption(it["why"])
                    if it["notes"]:
                        st.caption(it["notes"])
            if len(excluded) > 50:
                st.caption(f"Showing first 50 excluded methods (of {len(excluded)}).")

    else:
        st.subheader("Your text")
        st.write(user_text if user_text.strip() else "_(empty)_")

        st.subheader("Hard constraints")
        st.json(user, expanded=True)

        if text_probs and show_text_prefs:
            st.subheader("Predicted preferences (top-3)")
            for dim, dist in text_probs.items():
                top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
                st.write(f"**{dim}**: " + ", ".join([f"{k} ({v:.2f})" for k, v in top]))

        if show_excluded and excluded:
            st.subheader(f"Excluded by hard constraints ({len(excluded)})")
            for it in excluded[:50]:
                with st.container(border=True):
                    st.markdown(f"**{it['name']}**")
                    st.caption(it["why"])
                    if it["notes"]:
                        st.caption(it["notes"])
            if len(excluded) > 50:
                st.caption(f"Showing first 50 excluded methods (of {len(excluded)}).")
