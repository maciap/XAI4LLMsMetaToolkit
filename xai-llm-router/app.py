# app.py
import json
from typing import Any, Dict, List, Tuple

import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt

from text_to_score import rank_methods

from toolkits.captum_classifier import CaptumClassifierAttribution
from toolkits.bertviz_attention import BertVizAttention
from toolkits.logit_lens import LogitLens

# NEW: Anchors plugin
from toolkits.alibi_anchors_text import AlibiAnchorsText

import streamlit.components.v1 as components


@st.cache_resource
def get_plugins():
    plugin1 = CaptumClassifierAttribution()
    plugin2 = BertVizAttention()
    plugin3 = LogitLens()
    plugin4 = AlibiAnchorsText()  # NEW
    return {
        plugin1.id: plugin1,
        plugin2.id: plugin2,
        plugin3.id: plugin3,
        plugin4.id: plugin4,  # NEW
    }


PLUGINS = get_plugins()

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
        s += 2
        reasons.append("task match")

    if user["granularity"] != "NA" and user["granularity"] in norm_list(m.get("granularity", [])):
        s += 2
        reasons.append("granularity match")

    if user["goal"] != "NA" and user["goal"] in norm_list(m.get("user_goal_audience", [])):
        s += 2
        reasons.append("goal match")

    if user["format"] != "NA" and user["format"] in norm_list(m.get("format", [])):
        s += 2
        reasons.append("format match")

    if user["fidelity"] != "NA":
        f = m.get("fidelity", "NA")
        if f == user["fidelity"]:
            s += 2
            reasons.append("fidelity match")
        elif f == "mixed":
            s += 1
            reasons.append("fidelity partially supported")

    return s, reasons



def render_plugin_form(plugin):
    vals = {}

    ANCHOR_DEFAULTS = {
        "threshold": 0.90,
        "coverage_samples": 2000,
        "batch_size": 128,
        "beam_size": 1,
        "min_samples_start": 50,
        "n_covered_ex": 5,
        "max_anchor_size": 3,
        "delta": 0.1,
        "tau": 0.15,
        "max_length": 256,
    }

    for f in plugin.spec():
        if f.type == "textarea":
            vals[f.key] = st.text_area(f.label, help=getattr(f, "help", ""))
        elif f.type == "text":
            vals[f.key] = st.text_input(f.label, help=getattr(f, "help", ""))
        elif f.type == "select":
            vals[f.key] = st.selectbox(f.label, f.options or [], help=getattr(f, "help", ""))
        elif f.type == "number":
            # Base defaults you already had
            if f.key == "n_steps":
                default = 50
            elif f.key == "max_length":
                # Keep your old default *unless* Anchors overrides it below
                default = 128
            elif f.key == "top_k":
                default = 10
            elif f.key == "position_index":
                default = -1  # -1 means "last token"
            else:
                default = 0

            # Anchors-specific overrides (only affect those keys)
            if f.key in ANCHOR_DEFAULTS:
                default = ANCHOR_DEFAULTS[f.key]

            # Choose a reasonable step: ints vs floats
            step = 1.0
            if f.key in ("threshold", "delta", "tau"):
                step = 0.05

            vals[f.key] = st.number_input(
                f.label,
                value=float(default),
                step=float(step),
                help=getattr(f, "help", ""),
            )

        elif f.type == "checkbox":
            vals[f.key] = st.checkbox(f.label, help=getattr(f, "help", ""))
        else:
            st.warning(f"Unknown field type: {f.type} (field {f.key})")

    return vals



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
        recommended.append(
            {
                "name": m.get("name", "NA"),
                "plugin_id": m.get("plugin_id"),
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
            }
        )

    recommended.sort(key=lambda x: x["score"], reverse=True)
    text_probs = {}

else:
    if not user_text.strip():
        st.warning("Write a short description to get text-based recommendations.")
        recommended = []
        text_probs = {}
    else:
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
            recommended.append(
                {
                    "name": item["name"],
                    "plugin_id": m.get("plugin_id"),
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
                }
            )


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

            cbtn, cinfo = st.columns([1, 3])
            with cbtn:
                if st.button("Select", key=f"select_{item['name']}"):
                    st.session_state["selected_method"] = item["name"]
                    st.session_state["selected_plugin_id"] = item.get("plugin_id")
                    st.session_state.pop("last_outputs", None)

            with cinfo:
                if not item.get("plugin_id"):
                    st.caption("Not runnable yet (no plugin attached).")

            if item["notes"]:
                st.caption(item["notes"])

    st.divider()
    st.subheader("Run selected method")

    selected_plugin_id = st.session_state.get("selected_plugin_id")
    if not selected_plugin_id:
        st.info("Select a method above to run it.")
    else:
        plugin = PLUGINS.get(selected_plugin_id)
        if plugin is None:
            st.error(f"No runnable plugin registered for: {selected_plugin_id}")
        else:
            st.markdown(f"**Selected:** {plugin.name}")
            inputs = render_plugin_form(plugin)

            if st.button("Run explanation", key="run_expl"):
                try:
                    outputs = plugin.run(inputs)
                    st.session_state["last_outputs"] = outputs
                except Exception as e:
                    st.error(f"Run failed: {e}")

    outputs = st.session_state.get("last_outputs")
    if outputs and outputs.get("attributions"):
        st.subheader("Result")

        # --- User-facing helper text ---
        with st.expander("ℹ️ How to read this explanation", expanded=True):
            st.write(
                "- **Model**: a text classification model that assigns one label to your sentence.\n"
                "- **Prediction**: the label the model believes is most likely.\n"
                "- **Label being explained**: the label whose score we attribute back to the input words.\n"
                "- **Word importance**: tokens with larger absolute scores influence the label more.\n\n"
                "A **positive** attribution pushes the model *toward* the explained label; "
                "a **negative** attribution pushes it *away*."
            )

        # Summary
        pred = outputs.get("predicted", {})
        tgt = outputs.get("target", {})
        params = outputs.get("params", {})

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Algorithm:** {outputs.get('algorithm', 'NA')}")
        st.write(f"**Prediction:** {pred.get('label', pred.get('idx', 'NA'))}")
        st.write(f"**Label being explained:** {tgt.get('label', tgt.get('idx', 'NA'))}")

        # Only meaningful for IG
        if outputs.get("algorithm") == "IntegratedGradients":
            st.caption(f"Integrated Gradients steps: {params.get('n_steps', 'NA')}")

        # Label mapping (if available)
        label_map = outputs.get("label_map")
        if label_map:
            st.markdown("**Label meanings (index → name)**")
            st.json(label_map, expanded=False)

        st.caption("Each bar shows how strongly a token contributes to the explained label (normalized).")

        # Table + plot
        df = pd.DataFrame(outputs["attributions"])
        df_plot = df[~df["token"].isin(["[CLS]", "[SEP]", "[PAD]"])].copy()

        st.dataframe(df_plot[["token", "attr_raw", "attr_norm"]], use_container_width=True)

        fig = plt.figure()
        plt.bar(range(len(df_plot)), df_plot["attr_norm"].tolist())
        plt.xticks(range(len(df_plot)), df_plot["token"].tolist(), rotation=45, ha="right")
        plt.ylabel("Attribution (normalized)")
        plt.tight_layout()
        st.pyplot(fig)

        # ---- BertViz output ----
    elif outputs and outputs.get("plugin") == "bertviz_attention" and outputs.get("html"):
        st.subheader("Result")

        with st.expander("ℹ️ What you are seeing", expanded=True):
            st.write(
                "- Interactive attention visualization from BertViz.\n"
                "- Shows attention patterns by layer/head.\n"
                "- Attention ≠ importance, but it's useful for inspection/debugging."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**View:** {outputs.get('view', 'NA')}")

        components.html(outputs["html"], height=850, scrolling=True)

    

    # ---- Alibi Anchors output ----
    elif outputs and outputs.get("plugin") == "alibi_anchors_text":
        st.subheader("Result")

        with st.expander("ℹ️ How to read Anchors", expanded=True):
            st.write(
                "- **Anchors** are IF-THEN style rules (a set of words/spans) that 'lock in' the model prediction locally.\n"
                "- **Precision**: estimated probability the model keeps the same prediction when the anchor holds.\n"
                "- **Coverage**: how often the anchor applies under the perturbation distribution.\n"
                "- Anchors are **black-box**: they only need your model’s `predict_fn`."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        pred = outputs.get("predicted", {})
        st.write(f"**Prediction:** {pred.get('label', pred.get('idx', 'NA'))}")

        # Anchor rule (tokens/spans)
        anchor = outputs.get("anchor", None)
        if anchor is None:
            st.warning("No anchor found (try loosening the threshold / increasing samples).")
        else:
            # anchor can be list[str] or a string; handle both
            if isinstance(anchor, list):
                st.markdown("**Anchor (rule):** " + " ∧ ".join([f"`{a}`" for a in anchor]))
            else:
                st.markdown(f"**Anchor (rule):** `{anchor}`")

        # Precision / coverage
        precision = outputs.get("precision", None)
        coverage = outputs.get("coverage", None)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Precision", f"{precision:.3f}" if isinstance(precision, (int, float)) else "NA")
        with c2:
            st.metric("Coverage", f"{coverage:.3f}" if isinstance(coverage, (int, float)) else "NA")

        # Optional: show raw params used
        params = outputs.get("params", None)
        if params:
            st.caption("Parameters")
            st.json(params, expanded=False)

        # Examples where anchor applies / fails (if your plugin returns them)
        examples = outputs.get("examples", {})
        if isinstance(examples, dict) and examples:
            st.markdown("### Examples")
            ex_cols = st.columns(2)

            with ex_cols[0]:
                st.markdown("**Where the anchor holds**")
                ok_ex = examples.get("covered") or examples.get("covered_examples") or []
                if ok_ex:
                    for i, ex in enumerate(ok_ex[:10]):
                        st.write(f"{i+1}. {ex}")
                else:
                    st.caption("No examples provided.")

            with ex_cols[1]:
                st.markdown("**Where it fails / counterexamples**")
                bad_ex = examples.get("counterexamples") or examples.get("uncovered") or []
                if bad_ex:
                    for i, ex in enumerate(bad_ex[:10]):
                        st.write(f"{i+1}. {ex}")
                else:
                    st.caption("No counterexamples provided.")

    
    # ---- Logit Lens output ----
    elif outputs and outputs.get("plugin") == "logit_lens" and outputs.get("layers"):
        st.subheader("Result")

        with st.expander("ℹ️ How to read Logit Lens", expanded=True):
            st.write(
                "- **Logit lens** projects the hidden state at each layer into the vocabulary space.\n"
                "- For a chosen **token position**, it shows which tokens each layer 'leans toward' predicting.\n"
                "- It is a **diagnostic / mechanistic** view: useful for debugging and understanding representation evolution.\n"
                "- We use **auto-faithful normalization**: if the model has a final LayerNorm, we apply it to intermediate layers "
                 "**but not to the final layer**."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Text length (tokens):** {len(outputs.get('tokens', []))}")
        st.write(f"**Position inspected:** {outputs.get('position', 'NA')} (0-based index)")
        st.write(f"**Normalization mode:** {outputs.get('normalization_mode', 'NA')}")
        st.write(f"**Final norm detected:** {outputs.get('final_norm_detected', False)}")

        # Show token list with indices (readable)
        toks = outputs.get("tokens", [])
        if toks:
            preview = " ".join([f"{i}:{t}" for i, t in enumerate(toks)])
            st.caption("Tokenization (index:token)")
            st.code(preview)

        # Controls
        layers = outputs["layers"]
        n_layers = len(layers)
        top_k = int(outputs.get("top_k", 10))

        # Layer selector (slider)
        layer_idx = st.slider("Layer", 0, n_layers - 1, n_layers - 1)
        layer_obj = layers[layer_idx]

        st.markdown(f"### Top-{top_k} tokens at layer {layer_idx}")
        df = pd.DataFrame(layer_obj["top"])
        # columns: token, score
        st.dataframe(df, use_container_width=True)

        # Bar plot for selected layer
        fig = plt.figure()
        plt.bar(range(len(df)), df["score"].tolist())
        plt.xticks(range(len(df)), df["token"].tolist(), rotation=45, ha="right")
        plt.ylabel(f"Score ({outputs.get('score_type','prob')})")
        plt.title(f"Layer {layer_idx}: Top-{top_k} tokens")
        plt.tight_layout()
        st.pyplot(fig)

        # Track final-layer top token across layers (nice high-signal viz)
        tracked = outputs.get("tracked_token")
        tracked_probs = outputs.get("tracked_probs")

        if tracked and tracked_probs:
            st.markdown("### Consistency across layers (tracked token)")
            st.write(
                f"Tracked token = **{tracked.get('token','NA')}** "
                f"(from final layer top-1)."
            )
            fig2 = plt.figure()
            plt.plot(list(range(len(tracked_probs))), tracked_probs)
            plt.xlabel("Layer")
            plt.ylabel("Probability")
            plt.title("Probability of the final-layer top token across layers")
            plt.tight_layout()
            st.pyplot(fig2)



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

        if "text_probs" in locals() and text_probs and show_text_prefs:
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
