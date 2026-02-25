import json
import re
import io
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px

from text_to_score import rank_methods
# from toolkits.captum_classifier import CaptumClassifierAttribution
from toolkits.captum_classifier_methods import (
    CaptumIGClassifierAttribution,
    CaptumSaliencyClassifierAttribution,
    CaptumDeepLiftClassifierAttribution,
)

from toolkits.bertviz_attention import BertVizAttention
from toolkits.logit_lens import LogitLens
from toolkits.alibi_anchors_text import AlibiAnchorsText
from toolkits.direct_logit_attribution import DirectLogitAttribution
from toolkits.sae_feature_explorer import SAEFeatureExplorer
import torch  # safe local import for cuda check
import streamlit.components.v1 as components
from toolkits.inseq_proxy_http import (
    InseqDecoderIG_HTTP, InseqEncDecIG_HTTP,
    InseqDecoderGradientSHAP_HTTP, InseqEncDecGradientSHAP_HTTP,
    InseqDecoderDeepLIFT_HTTP, InseqEncDecDeepLIFT_HTTP,
    InseqDecoderInputXGradient_HTTP, InseqEncDecInputXGradient_HTTP,
    InseqDecoderLIME_HTTP, InseqEncDecLIME_HTTP,
    InseqDecoderDiscretizedIG_HTTP, InseqEncDecDiscretizedIG_HTTP
)
from toolkits.meta_transparency import MetaTransparencyGraph  # adjust import path
from toolkits.attention_rollout import AttentionRollout

import tempfile
import os
from pyvis.network import Network

from toolkits.PCAViz import EmbeddingPCALayers
import plotly.express as px
import plotly.graph_objects as go
from toolkits.linear_cka import LinearCKALayers

from toolkits.cca_layers import CCALayers

import html as _html

from Navigator_utils import (
    _parse_node,
    render_meta_graph_svg,
    captum_method_explainer_text,
    render_token_highlight,
    render_downloads,
    render_captum_result,
    _to_node_id,
    _infer_edges_and_nodes,
    render_meta_flow_pyvis,
    _now_stamp,
    _to_json_bytes,
    _fig_to_png_bytes,
    _make_prefix,
    norm_list,
    load_methods,
    feasible,
    _acc_rank_order,
    score,
    render_plugin_form,
    _safe,
    _chip,
    render_selected_tool_card,
)

import html
import re
import streamlit.components.v1 as components

_NODE_RE = re.compile(r"^(X0|A|M|I)(\d+)?_(\d+)$")  # X0_3 OR A6_3 etc.

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())


@st.cache_resource
def get_plugins():
    plugin1 = CaptumIGClassifierAttribution()
    plugin2 = BertVizAttention()
    plugin3 = LogitLens()
    plugin4 = AlibiAnchorsText()
    plugin5 = DirectLogitAttribution()
    plugin6 = SAEFeatureExplorer()

    # --- Inseq (existing IG) ---
    plugin7 = InseqDecoderIG_HTTP()
    plugin8 = InseqEncDecIG_HTTP()

    # --- Inseq new methods ---
    plugin15 = InseqDecoderGradientSHAP_HTTP()
    plugin16 = InseqEncDecGradientSHAP_HTTP()

    plugin17 = InseqDecoderDeepLIFT_HTTP()
    plugin18 = InseqEncDecDeepLIFT_HTTP()

    plugin19 = InseqDecoderInputXGradient_HTTP()
    plugin20 = InseqEncDecInputXGradient_HTTP()

    plugin21 = InseqDecoderLIME_HTTP()
    plugin22 = InseqEncDecLIME_HTTP()

    plugin23 = InseqDecoderDiscretizedIG_HTTP()
    plugin24 = InseqEncDecDiscretizedIG_HTTP()

    # --- Others ---
    plugin9 = MetaTransparencyGraph()
    plugin10 = CaptumSaliencyClassifierAttribution()
    plugin11 = CaptumDeepLiftClassifierAttribution()
    plugin12 = EmbeddingPCALayers()
    plugin13 = LinearCKALayers()
    plugin14 = CCALayers()
    plugin25 = AttentionRollout()

    return {
        plugin1.id: plugin1,
        plugin2.id: plugin2,
        plugin3.id: plugin3,
        plugin4.id: plugin4,
        plugin5.id: plugin5,
        plugin6.id: plugin6,

        # Inseq
        plugin7.id: plugin7,
        plugin8.id: plugin8,
        plugin15.id: plugin15,
        plugin16.id: plugin16,
        plugin17.id: plugin17,
        plugin18.id: plugin18,
        plugin19.id: plugin19,
        plugin20.id: plugin20,
        plugin21.id: plugin21,
        plugin22.id: plugin22,
        plugin23.id: plugin23,
        plugin24.id: plugin24,

        plugin9.id: plugin9,
        plugin10.id: plugin10,
        plugin11.id: plugin11,
        plugin12.id: plugin12,
        plugin13.id: plugin13,
        plugin14.id: plugin14,
        plugin25.id: plugin25,
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
    "accessibility": ["NA", "experts", "mid experts", "non experts"],
}

DEFAULTS = {
    "task": "NA",
    "access": "NA",
    "arch": "NA",
    "scope": "NA",
    "accessibility": "NA",
}

HARD_DIMS = ["task", "access", "arch", "scope"]
PREF_DIMS = ["accessibility"]


# -------------------------
# Compare view renderer
# -------------------------
def _compare_key(item: Dict[str, Any]) -> str:
    """
    Unique key for compare/selection even if plugin_id is missing.
    """
    pid = item.get("plugin_id")
    if pid:
        return f"plugin::{pid}"
    # fall back to name + notes hash-ish stable string
    nm = str(item.get("name", "NA"))
    return f"method::{nm}"


def render_compare_view(anchor_item: Dict[str, Any], other_items: List[Dict[str, Any]]):
    """
    Compare view: selected (anchor) + up to 2 other tools.
    Shows:
      - metadata comparison
      - main functionalities side-by-side
      - strengths side-by-side
      - limitations side-by-side
    """
    items = [anchor_item] + (other_items or [])
    items = items[:3]

    st.markdown("---")
    st.subheader("🔍 Comparison")

    # ---- Metadata ----
    rows = []
    anchor_k = _compare_key(anchor_item)
    for it in items:
        meta = it.get("meta", {}) or {}
        is_anchor = (_compare_key(it) == anchor_k)
        rows.append({
            "tool": (it.get("name", "NA") + ("  🧭" if is_anchor else "")),
            "plugin_id": it.get("plugin_id", "NA"),
            "scope": meta.get("scope", "NA"),
            "access": meta.get("access", "NA"),
            "arch": meta.get("arch", "NA"),
            "granularity": meta.get("granularity", "NA"),
            "format": meta.get("format", "NA"),
            "fidelity": meta.get("fidelity", "NA"),
            "accessibility": meta.get("accessibility", it.get("accessibility", "NA")),
            "score": float(it.get("score", 0.0)),
        })
    st.markdown("### Metadata")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---- helpers ----
    def main_funcs(it: Dict[str, Any]) -> List[str]:
        desc = it.get("description", {}) or {}
        mf = desc.get("main_functionalities", []) or []
        return [str(x) for x in mf if str(x).strip()]

    def strengths(it: Dict[str, Any]) -> List[str]:
        return [str(x) for x in (it.get("strengths", []) or []) if str(x).strip()]

    def limitations(it: Dict[str, Any]) -> List[str]:
        return [str(x) for x in (it.get("limitations", []) or []) if str(x).strip()]

    def render_section(title: str, getter):
        st.markdown(f"### {title}")
        cols = st.columns(len(items), gap="large")
        for col, it in zip(cols, items):
            with col:
                #if _compare_key(it) == anchor_k:
                #    st.markdown("**🧭 Selected (anchor)**")
                st.markdown(f"#### {it.get('name','NA')}")
                vals = getter(it)
                if not vals:
                    st.caption("NA")
                else:
                    for v in vals:
                        st.markdown(f"- {v}")

    render_section("Main functionalities", main_funcs)
    render_section("Strengths", strengths)
    render_section("Limitations", limitations)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="LLM Explainability Navigator 🧭", layout="wide")

# ---- Session state (important fixes) ----
if "selected_item" not in st.session_state:
    st.session_state["selected_item"] = None
if "selected_key" not in st.session_state:
    st.session_state["selected_key"] = None
if "selected_plugin_id" not in st.session_state:
    st.session_state["selected_plugin_id"] = None
if "last_outputs" not in st.session_state:
    st.session_state["last_outputs"] = None

# Compare holds "other" tools only (max 2)
if "compare_keys" not in st.session_state:
    st.session_state["compare_keys"] = []          # list[str]
if "compare_items" not in st.session_state:
    st.session_state["compare_items"] = {}         # key -> item dict

# keep if you still use it elsewhere
if "compare_outputs" not in st.session_state:
    st.session_state["compare_outputs"] = {}

# Top image
st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("images/logo_app.png", width=220)

st.markdown("""
<style>

/* Use Streamlit theme variables */
:root {
  --radius: 16px;
}

/* Page spacing */
.block-container {
  padding-top: 1.6rem;
  padding-bottom: 2rem;
  max-width: 1650px !important;
}

/* Headings */
h1, h2, h3 {
  letter-spacing: -0.3px;
}

/* Cards */
.xai-card {
  border-radius: var(--radius);
  background: var(--secondary-background-color);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 1.2rem;
}

/* Expander styling */
div[data-testid="stExpander"] > details {
  border-radius: var(--radius) !important;
  background: var(--secondary-background-color) !important;
}

/* Buttons */
.stButton button {
  border-radius: 12px;
  font-weight: 600;
}

/* Inputs */
div[data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea {
  border-radius: 12px;
}

/* Make captions readable in dark mode */
.stCaption {
  color: var(--text-color) !important;
  opacity: 0.7;
}

.xai-chip {
  display: inline-block;
  padding: 0.22rem 0.65rem;
  margin: 0 0.35rem 0.35rem 0;
  border-radius: 999px;

  background: transparent !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
  color: #E5E7EB !important;

  font-size: 0.85rem;
  font-weight: 600;
}

/* --- Columns --- */
div[data-testid="column"] * {
  min-width: 0 !important;
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

div[data-testid="stMarkdownContainer"] span {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<h1 style="
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
">
    LLM Explainability Navigator 🧭
</h1>
<p style="
    font-size: 1.1rem;
    color: var(--text-color);
    margin-top: 0;
">
    Discover the tools for explaining LLMs that fit your needs.
</p>
""",
    unsafe_allow_html=True,
)

try:
    methods = load_methods("methods.json")
except Exception as e:
    st.error(f"Failed to load methods.json: {e}")
    st.stop()

# -------------------------
# Sidebar: hard constraints vs preferences
# -------------------------
with st.sidebar:
    st.header("Tell me what you are looking for!")

    mode = st.radio(
        "How would you like to search?",
        ["Pick with filters", "Describe it in words"],
        index=0,
    )

    top_k = st.slider("Max recommendations", 5, 50, 20)

    hard: Dict[str, str] = {k: "NA" for k in DIM_VALUES.keys()}
    prefs: Dict[str, str] = {k: "NA" for k in DIM_VALUES.keys()}
    user_text = ""

    if mode == "Pick with filters":
        with st.expander("✅ Hard constraints (filters)", expanded=True):
            st.caption("These are *must-have*. Tools that don't satisfy these will be hidden.")
            hard["task"] = st.selectbox("Task (hard)", DIM_VALUES["task"], index=DIM_VALUES["task"].index(DEFAULTS["task"]))
            hard["access"] = st.selectbox("Model access (hard)", DIM_VALUES["access"], index=DIM_VALUES["access"].index(DEFAULTS["access"]))
            hard["arch"] = st.selectbox("Architecture (hard)", DIM_VALUES["arch"], index=DIM_VALUES["arch"].index(DEFAULTS["arch"]))
            hard["scope"] = st.selectbox("Explanation scope (hard)", DIM_VALUES["scope"], index=DIM_VALUES["scope"].index(DEFAULTS["scope"]))

        with st.expander("⭐ Ranking preference (accessibility)", expanded=True):
            st.caption("This does not hide tools. It only changes ordering.")
            prefs["accessibility"] = st.selectbox(
                "Audience / accessibility (ranking only)",
                DIM_VALUES["accessibility"],
                index=DIM_VALUES["accessibility"].index(DEFAULTS["accessibility"]),
            )

        st.info("Tip: If a tool appears but doesn't match your format/fidelity, it’s because those are preferences (ranking), not filters.")

    else:
        user_text = st.text_area(
            "Describe it in words",
            placeholder=(
                "Example: I need white-box mechanistic interpretability for a transformer, "
                "focusing on attention heads and circuits; both local and global insights; "
                "prefer an interactive UI."
            ),
            height=160,
        )

        add_hard = st.checkbox("Add hard constraints too", value=False)

        if add_hard:
            with st.expander("✅ Hard constraints (optional)", expanded=True):
                hard["task"] = st.selectbox("Task (hard)", DIM_VALUES["task"], index=0)
                hard["access"] = st.selectbox("Model access (hard)", DIM_VALUES["access"], index=0)
                hard["arch"] = st.selectbox("Architecture (hard)", DIM_VALUES["arch"], index=0)
                hard["scope"] = st.selectbox("Explanation scope (hard)", DIM_VALUES["scope"], index=0)

        with st.expander("⭐ Ranking preference (accessibility)", expanded=True):
            prefs["accessibility"] = st.selectbox(
                "Audience / accessibility (ranking only)",
                DIM_VALUES["accessibility"],
                index=DIM_VALUES["accessibility"].index(DEFAULTS["accessibility"]),
            )

        temperature = st.slider("Text model temperature", 0.2, 1.5, 0.7, 0.05)
        show_text_prefs = st.checkbox("Show predicted preferences", value=True)

# -------------------------
# Compute recommendations
# -------------------------
recommended: List[Dict[str, Any]] = []
excluded: List[Dict[str, Any]] = []

if mode == "Pick with filters":
    for m in methods:
        ok, why = feasible(hard, m)
        if not ok:
            excluded.append({"name": m.get("name", "NA"), "why": why, "notes": m.get("notes", "")})
            continue

        sc, matched, mismatched = score(prefs, m)
        recommended.append(
            {
                "name": m.get("name", "NA"),
                "plugin_id": m.get("plugin_id"),  # may be None (still selectable + comparable)
                "score": float(sc),
                "matched": matched,
                "mismatched": mismatched,
                "notes": m.get("notes", ""),
                "description": m.get("description", {}),
                "strengths": m.get("strengths", []),
                "limitations": m.get("limitations", []),
                "accessibility": m.get("accessibility", "NA"),
                "research_applications": m.get("research_applications", []),
                "meta": {
                    "scope": m.get("target_scope", "NA"),
                    "access": m.get("access_arch", {}).get("access", "NA"),
                    "arch": m.get("access_arch", {}).get("arch", "NA"),
                    "granularity": m.get("granularity", "NA"),
                    "format": m.get("format", "NA"),
                    "fidelity": m.get("fidelity", "NA"),
                    "accessibility": m.get("accessibility", "NA"),
                },
                "hard_used": {k: hard.get(k, "NA") for k in HARD_DIMS},
                "prefs_used": {k: prefs.get(k, "NA") for k in PREF_DIMS},
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
            ok, why = feasible(hard, m)
            if not ok:
                excluded.append({"name": m.get("name", "NA"), "why": why, "notes": m.get("notes", "")})
                continue
            filtered_methods.append(m)

        ranked, text_probs = rank_methods(
            methods=filtered_methods,
            user_text=user_text.strip(),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        for item in ranked:
            m = item["method"]
            sc, matched, mismatched = score(prefs, m)

            recommended.append(
                {
                    "name": item["name"],
                    "plugin_id": m.get("plugin_id"),  # may be None
                    "score": float(item["final_score"]),
                    "matched": ["🧠 text match"] + matched,
                    "mismatched": mismatched,
                    "notes": m.get("notes", ""),
                    "description": m.get("description", {}),
                    "strengths": m.get("strengths", []),
                    "limitations": m.get("limitations", []),
                    "research_applications": m.get("research_applications", []),
                    "meta": {
                        "scope": m.get("target_scope", "NA"),
                        "access": m.get("access_arch", {}).get("access", "NA"),
                        "arch": m.get("access_arch", {}).get("arch", "NA"),
                        "granularity": m.get("granularity", "NA"),
                        "format": m.get("format", "NA"),
                        "fidelity": m.get("fidelity", "NA"),
                        "accessibility": m.get("accessibility", "NA"),
                    },
                    "hard_used": {k: hard.get(k, "NA") for k in HARD_DIMS},
                    "prefs_used": {k: prefs.get(k, "NA") for k in PREF_DIMS},
                }
            )

# -------------------------
# Layout (3 columns)
# -------------------------
col_spacer, col_recs, col_run = st.columns([0.2, 1.4, 1.8], gap="large")

# Column 2: Recommendations
with col_recs:
    st.subheader(f"👇 {min(top_k, len(recommended))} tools match your request")

    with st.expander("🔎 Current selection (what filters vs what ranks)", expanded=False):
        st.markdown("**✅ Hard constraints (filters):**")
        st.json({k: hard.get(k, "NA") for k in HARD_DIMS}, expanded=False)
        st.markdown("**⭐ Preferences (ranking only):**")
        st.json({k: prefs.get(k, "NA") for k in PREF_DIMS}, expanded=False)

    for item in recommended[:top_k]:
        item_key = _compare_key(item)

        with st.container(border=True):
            st.markdown(f"### {item['name']}")
            st.write(f"**Preference score:** {item['score']:.2f}")

            if item.get("matched"):
                st.caption("✅ Matches preferences: " + ", ".join(item["matched"]))
            if item.get("mismatched"):
                st.caption("⚠️ Mismatches: " + "; ".join(item["mismatched"]))

            # NOTE: Select is ALWAYS available now (even if plugin_id is None)
            cA, cB = st.columns([1, 1], gap="medium")

            with cA:
                if st.button("Select", key=f"select__{item_key}"):
                    st.session_state["selected_item"] = item
                    st.session_state["selected_key"] = item_key
                    st.session_state["selected_plugin_id"] = item.get("plugin_id")  # may be None
                    st.session_state["last_outputs"] = None

                    # ensure anchor isn't in compare list
                    st.session_state["compare_keys"] = [k for k in st.session_state["compare_keys"] if k != item_key]
                    st.session_state["compare_items"].pop(item_key, None)

            # Add-to-compare should ONLY APPEAR after first selection (no disabled/transparent buttons)
            with cB:
                anchor_key = st.session_state.get("selected_key")
                if anchor_key and (item_key != anchor_key):
                    in_compare = item_key in st.session_state["compare_keys"]
                    if not in_compare:
                        if st.button("➕ Add to compare", key=f"cmp_add__{item_key}"):
                            if len(st.session_state["compare_keys"]) >= 2:
                                st.warning("You can compare up to 3 tools total (selected + 2).")
                            else:
                                st.session_state["compare_keys"].append(item_key)
                                st.session_state["compare_items"][item_key] = item
                    else:
                        if st.button("➖ Remove", key=f"cmp_rm__{item_key}"):
                            st.session_state["compare_keys"] = [k for k in st.session_state["compare_keys"] if k != item_key]
                            st.session_state["compare_items"].pop(item_key, None)
                            st.session_state["compare_outputs"].pop(item_key, None)

# Column 3: Selected method + Run + Result
with col_run:
    st.subheader("Selected tool")

    selected_item = st.session_state.get("selected_item")
    selected_plugin_id = st.session_state.get("selected_plugin_id")

    if not selected_item:
        st.info("Select a tool on the left.")
    else:
        # Always show the card (even if not runnable)
        render_selected_tool_card(selected_item)

        #st.markdown("#### ✅/⚠️ Preference fit for this tool")
        #m1, m2 = st.columns(2, gap="large")
        #with m1:
        #    st.markdown("**✅ Matches**")
        #    if selected_item.get("matched"):
        #        for x in selected_item["matched"]:
        #            st.write(f"- {x}")
        #    else:
        #        st.caption("No preference matches.")
        #with m2:
        #    st.markdown("**⚠️ Mismatches**")
        #    if selected_item.get("mismatched"):
        #        for x in selected_item["mismatched"]:
        #            st.write(f"- {x}")
        #    else:
        #        st.caption("No preference mismatches.")

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # If the selected tool has no runnable plugin, stop here (still comparable!)
        if not selected_plugin_id:
            #st.info("This tool is not runnable in the UI yet (no plugin connected). You can still compare its metadata/strengths/limitations.")
            st.info("This tool is not runnable in the UI.")
        else:
            plugin = PLUGINS.get(selected_plugin_id)
            if plugin is None:
                st.warning(f"Plugin id is set but no runnable plugin is registered for: {selected_plugin_id}")
            else:
                st.markdown(f"### {plugin.name}")
                inputs = render_plugin_form(plugin)

                if st.button("Run explanation", key="run_expl"):
                    try:
                        outputs = plugin.run(inputs)
                        st.session_state["last_outputs"] = outputs
                    except Exception as e:
                        st.error(f"Run failed: {e}")

                outputs = st.session_state.get("last_outputs")

                # ---- Captum renderers ----
                if outputs and outputs.get("plugin") in (
                    "captum_ig_classifier",
                    "captum_saliency_classifier",
                    "captum_deeplift_classifier",
                ) and outputs.get("attributions"):
                    render_captum_result(outputs, selected_item)

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
                    render_downloads(outputs, selected_item=selected_item)

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

                    anchor = outputs.get("anchor", None)
                    is_empty_anchor = (
                        anchor is None
                        or (isinstance(anchor, str) and anchor.strip() == "")
                        or (isinstance(anchor, (list, tuple)) and len(anchor) == 0)
                    )

                    if is_empty_anchor:
                        st.warning("No anchor found (try lowering threshold / increasing coverage_samples / increasing beam_size).")
                    else:
                        if isinstance(anchor, list):
                            st.markdown("**Anchor (rule):** " + " ∧ ".join([f"`{a}`" for a in anchor]))
                        else:
                            st.markdown(f"**Anchor (rule):** `{anchor}`")

                    precision = outputs.get("precision", None)
                    coverage = outputs.get("coverage", None)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Precision", f"{precision:.3f}" if isinstance(precision, (int, float)) else "NA")
                    with c2:
                        st.metric("Coverage", f"{coverage:.3f}" if isinstance(coverage, (int, float)) else "NA")

                    examples = outputs.get("examples", {}) or {}
                    if isinstance(examples, dict) and (examples.get("covered_true") or examples.get("covered_false")):
                        st.markdown("### Examples")
                        ex_cols = st.columns(2)

                        with ex_cols[0]:
                            st.markdown("**Where the anchor holds (covered_true)**")
                            ok_ex = examples.get("covered_true", []) or []
                            if ok_ex:
                                for i, ex in enumerate(ok_ex[:10]):
                                    st.write(f"{i+1}. {ex}")
                            else:
                                st.caption("No examples provided.")

                        with ex_cols[1]:
                            st.markdown("**Where it flips (covered_false)**")
                            bad_ex = examples.get("covered_false", []) or []
                            if bad_ex:
                                for i, ex in enumerate(bad_ex[:10]):
                                    st.write(f"{i+1}. {ex}")
                            else:
                                st.caption("No counterexamples provided.")
                    else:
                        st.caption("No example texts returned by the explainer (try increasing n_covered_ex).")

                    params = outputs.get("params", None)
                    if params:
                        with st.expander("Parameters", expanded=False):
                            st.json(params, expanded=False)

                    render_downloads(outputs, selected_item=selected_item)

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

                    toks = outputs.get("tokens", [])
                    if toks:
                        preview = " ".join([f"{i}:{t}" for i, t in enumerate(toks)])
                        st.caption("Tokenization (index:token)")
                        st.code(preview)

                    layers = outputs["layers"]
                    n_layers = len(layers)
                    top_k_ll = int(outputs.get("top_k", 10))

                    layer_idx = st.slider("Layer", 0, n_layers - 1, n_layers - 1)
                    layer_obj = layers[layer_idx]

                    st.markdown(f"### Top-{top_k_ll} tokens at layer {layer_idx}")
                    df = pd.DataFrame(layer_obj["top"])
                    st.dataframe(df, use_container_width=True)

                    fig = plt.figure()
                    plt.bar(range(len(df)), df["score"].tolist())
                    plt.xticks(range(len(df)), df["token"].tolist(), rotation=45, ha="right")
                    plt.ylabel(f"Score ({outputs.get('score_type','prob')})")
                    plt.title(f"Layer {layer_idx}: Top-{top_k_ll} tokens")
                    plt.tight_layout()
                    st.pyplot(fig)

                    tracked = outputs.get("tracked_token")
                    tracked_probs = outputs.get("tracked_probs")

                    figs = {
                        f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_layer_{layer_idx}_top_tokens.png": fig
                    }

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

                        figs[f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_tracked_token_across_layers.png"] = fig2

                    render_downloads(outputs, selected_item=selected_item, figs=figs)

                elif outputs and outputs.get("plugin") == "direct_logit_attribution" and outputs.get("components"):
                    st.subheader("Result")
                    with st.expander("ℹ️ How to read Direct Logit Attribution (DLA)", expanded=True):
                        st.write(
                            "- **DLA** decomposes a single **target logit** into contributions from transformer components.\n"
                            "- Each component output vector is projected onto the **unembedding direction** of the target token.\n"
                            "- Positive values push the model *toward* the target token; negative values push it *away*.\n"
                            "- This is a **linear diagnostic** view (not fully causal): it ignores softmax coupling and other nonlinear interactions."
                        )

                    st.write(f"**Model:** {outputs.get('model', 'NA')}")
                    st.write(f"**Architecture detected:** {outputs.get('arch_detected', 'NA')}")
                    st.write(f"**Text length (tokens):** {len(outputs.get('tokens', []))}")
                    st.write(f"**Position inspected:** {outputs.get('position', 'NA')} (0-based index)")

                    pred = outputs.get("predicted_next", {})
                    tgt = outputs.get("target", {})
                    st.write(f"**Predicted next token:** {pred.get('token','NA')}  (id={pred.get('id','NA')})")
                    st.write(f"**Target token:** {tgt.get('token','NA')}  (id={tgt.get('id','NA')}, mode={tgt.get('mode','NA')})")
                    st.write(f"**Total target logit:** {outputs.get('total_logit', 0.0):.4f}")

                    toks = outputs.get("tokens", [])
                    if toks:
                        preview = " ".join([f"{i}:{t}" for i, t in enumerate(toks)])
                        st.caption("Tokenization (index:token)")
                        st.code(preview)

                    comps = outputs["components"]
                    df = pd.DataFrame(comps)

                    sort_mode = st.selectbox("Sort components by", ["abs_contribution (desc)", "contribution (desc)", "layer (asc)"])
                    if sort_mode == "contribution (desc)":
                        df = df.sort_values("contribution", ascending=False)
                    elif sort_mode == "layer (asc)":
                        df = df.sort_values(["layer", "type"], ascending=True)
                    else:
                        df = df.sort_values("abs_contribution", ascending=False)

                    st.markdown(f"### Top-{outputs.get('top_n', len(df))} component contributions")
                    st.dataframe(df, use_container_width=True)

                    fig = plt.figure()
                    plt.bar(range(len(df)), df["contribution"].tolist())
                    plt.xticks(range(len(df)), df["component"].tolist(), rotation=60, ha="right")
                    plt.ylabel("Contribution to target logit")
                    plt.title("Direct Logit Attribution (component → target logit)")
                    plt.tight_layout()
                    st.pyplot(fig)

                    notes = outputs.get("notes", [])
                    if notes:
                        with st.expander("Notes / caveats", expanded=False):
                            for n in notes:
                                st.write(f"- {n}")

                    render_downloads(
                        outputs,
                        selected_item=selected_item,
                        figs={f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_dla_components.png": fig},
                    )

                elif outputs and outputs.get("plugin") == "sae_feature_explorer":
                    st.subheader("Result")
                    # (unchanged body from your original; kept short here on purpose)
                    st.json(outputs, expanded=False)
                    render_downloads(outputs, selected_item=selected_item)

                elif outputs and str(outputs.get("plugin", "")).startswith("inseq_") and outputs.get("out"):
                    import re

                    def inseq_html_dark_fix(html: str) -> str:
                        if not html:
                            return html
                        html = re.sub(r"<style.*?>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
                        css = """
                        <style>
                        :root { color-scheme: dark; }
                        html, body {
                            background: transparent !important;
                            color: #E5E7EB !important;
                            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
                        }
                        body, body * { color: inherit !important; }
                        table { background: transparent !important; }
                        td, th { border-color: rgba(255,255,255,0.15) !important; }
                        pre, code { background: rgba(255,255,255,0.06) !important; color: inherit !important; }
                        div, section, article { background: transparent !important; }
                        </style>
                        """
                        if re.search(r"</head>", html, flags=re.IGNORECASE):
                            html = re.sub(r"</head>", css + "</head>", html, flags=re.IGNORECASE)
                        else:
                            html = css + html
                        return html

                    st.subheader("Result")
                    st.write(f"**Model:** {outputs.get('model', 'NA')}")
                    st.write(f"**Device:** {outputs.get('device', 'NA')}")
                    st.write(f"**Text:** {outputs.get('text', '')}")
                    fixed = inseq_html_dark_fix(outputs["out"])
                    components.html(fixed, height=850, scrolling=True)
                    render_downloads(outputs, selected_item=selected_item)

                elif outputs and outputs.get("plugin") == "meta_transparency_graph":
                    st.subheader("Result")
                    st.json(outputs, expanded=False)
                    render_downloads(outputs, selected_item=selected_item)

                elif outputs and outputs.get("plugin") == "embedding_pca_layers" and outputs.get("projected"):
                    st.subheader("Result")
                    st.json(outputs, expanded=False)
                    render_downloads(outputs, selected_item=selected_item)

                elif outputs and outputs.get("plugin") == "linear_cka_layers" and outputs.get("cka_matrix"):
                    st.subheader("Result")
                    st.json(outputs, expanded=False)
                    render_downloads(outputs, selected_item=selected_item)

                elif outputs and outputs.get("plugin") == "cca_layers" and outputs.get("cca_matrix"):
                    st.subheader("Result")
                    st.json(outputs, expanded=False)
                    render_downloads(outputs, selected_item=selected_item)

                elif outputs and outputs.get("plugin") == "attention_rollout" and outputs.get("token_scores"):
                    st.subheader("Result")
                    st.json(outputs, expanded=False)
                    render_downloads(outputs, selected_item=selected_item)

    # -------------------------
    # Compare tools (NEW DESIGN)
    # -------------------------
    st.markdown("---")
    #st.subheader("Compare tools")

    anchor_item = st.session_state.get("selected_item")
    anchor_key = st.session_state.get("selected_key")
    cmp_keys = st.session_state.get("compare_keys", [])

    if not anchor_item or not anchor_key:
        st.info("Select a tool first. Then “Add to compare” will appear next to other tools.")
    else:
        cmp_keys = [k for k in cmp_keys if k != anchor_key][:2]
        other_items = []
        for k in cmp_keys:
            it = st.session_state["compare_items"].get(k)
            if it:
                other_items.append(it)

        if not other_items:
            st.info("Add up to 2 other tools from the left to compare.")
        else:
            render_compare_view(anchor_item, other_items)

            c1, c2 = st.columns([1, 1], gap="medium")
            with c1:
                if st.button("Clear compare list", key="cmp_clear"):
                    st.session_state["compare_keys"] = []
                    st.session_state["compare_items"] = {}
                    st.rerun()
            with c2:
                st.caption("Max 3 tools: selected + 2 comparisons.")