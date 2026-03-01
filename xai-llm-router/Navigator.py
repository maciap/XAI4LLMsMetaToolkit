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
from toolkits.captum_classifier import (
    CaptumIGClassifierAttribution,
    CaptumSaliencyClassifierAttribution,
    CaptumDeepLiftClassifierAttribution,
    CaptumInputXGradientClassifierAttribution,
    CaptumGradientShapClassifierAttribution,
    CaptumOcclusionClassifierAttribution,
    CaptumFeatureAblationClassifierAttribution,
    CaptumNoiseTunnelSaliencyClassifierAttribution,
    CaptumNoiseTunnelIGClassifierAttribution,
    CaptumNoiseTunnelInputXGradClassifierAttribution,
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

from toolkits.probing import ProbingBinaryExamples

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
    _pretty_task_label, 
    _pretty_arch_label
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


    plugin26 = CaptumInputXGradientClassifierAttribution()
    plugin27 = CaptumGradientShapClassifierAttribution()
    plugin28 = CaptumOcclusionClassifierAttribution()
    plugin29 = CaptumFeatureAblationClassifierAttribution()
    plugin30 = CaptumNoiseTunnelSaliencyClassifierAttribution()
    plugin31 = CaptumNoiseTunnelIGClassifierAttribution()
    plugin32 = CaptumNoiseTunnelInputXGradClassifierAttribution()


    plugin33 = ProbingBinaryExamples()

    return {
        plugin1.id: plugin1,
        plugin2.id: plugin2,
        plugin3.id: plugin3,
        plugin4.id: plugin4,
        plugin5.id: plugin5,
        plugin6.id: plugin6,

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


        plugin26.id: plugin26,
        plugin27.id: plugin27,
        plugin28.id: plugin28,
        plugin29.id: plugin29,
        plugin30.id: plugin30,
        plugin31.id: plugin31,
        plugin32.id: plugin32,

        plugin33.id: plugin33

    }


PLUGINS = get_plugins()

# -------------------------
# Config: dimension values
# -------------------------

DIM_VALUES = {
    "task": ["NA", "classification", "generation"],
    "access": ["NA", "black_box",  "white_box"],
    "arch": ["NA", "decoder", "encdec"],
    "scope": ["NA", "local", "global", "both"],
    "accessibility": ["NA", "experts", "mid experts", "non experts"],
}

DEFAULTS = {
    "task": "classification",
    "access": "white_box",
    "arch": "decoder",
    "scope": "local",
    "accessibility": "non experts",
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
            "task": meta.get("task", "NA"),
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
    Virgil: Your LLM Explainability Navigator 🧭
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

        st.info("Tip: If a tool appears but doesn't match your level of expertise, it’s because that is just a preference.")

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
                "implementation": m.get("implementation"),
                "score": float(sc),
                "matched": matched,
                "mismatched": mismatched,
                "notes": m.get("notes", ""),
                "description": m.get("description", {}),
                "strengths": m.get("strengths", []),
                "limitations": m.get("limitations", []),
                "accessibility": m.get("accessibility", "NA"),
                "research_applications": m.get("research_applications", []),
                "task_input": m.get("task_input", []),
                "meta": {
                     "task": m.get("task_input", "NA"),
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
                    "implementation": m.get("implementation"), 
                    "score": float(item["final_score"]),
                    "matched": ["🧠 text match"] + matched,
                    "mismatched": mismatched,
                    "notes": m.get("notes", ""),
                    "description": m.get("description", {}),
                    "strengths": m.get("strengths", []),
                    "limitations": m.get("limitations", []),
                    "research_applications": m.get("research_applications", []),
                    "task_input": m.get("task_input", []),

                    "meta": {
                        "task": m.get("task_input", "NA"),
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
            tlabel = _pretty_task_label(item)
            alabel = _pretty_arch_label(item)

            # Show Task + Arch ONLY for allowed values
            if tlabel and tlabel.strip():
                st.caption(f"🎯 Task: {tlabel}")
            if alabel:
                st.caption(f"🏗️ Architecture: {alabel}")

            st.write(f"**Preference score:** {item['score']:.2f}")

            if item.get("matched"):
                st.caption("✅ Matches preferences: " + ", ".join(item["matched"]))
            if item.get("mismatched"):
                st.caption("⚠️ Mismatches: " + "; ".join(item["mismatched"]))

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
                    "captum_inputxgradient_classifier",
                    "captum_gradientshap_classifier",
                    "captum_occlusion_classifier",
                    "captum_featureablation_classifier",
                    "captum_noisetunnel_saliency_classifier",
                    "captum_noisetunnel_ig_classifier",
                    "captum_noisetunnel_inputxgrad_classifier",
                    ):
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
                    with st.expander("ℹ️ How to read Sparse Autoencoders (SAELens + Neuronpedia)", expanded=True):
                        st.write(
                            "- A **Sparse Autoencoder (SAE)** learns a set of directions (called **features**) in a model’s hidden activations.\n"
                            "- For each token position, the SAE **encodes** the model activation into a sparse vector of **feature activations**.\n"
                            "- Each row in **Top activating SAE features** is:\n"
                            "  - **feature_id**: the index of a learned feature (a latent direction)\n"
                            "  - **activation**: how strongly that feature is present at the selected token position\n\n"
                            "**Interpretation tips:**\n"
                            "- Higher **activation** ⇒ the feature is more strongly present for that token at this layer/hook.\n"
                            "- Features are **not labels** by default. To understand a feature, you usually inspect:\n"
                            "  1) which tokens/contexts make it fire (top examples), and\n"
                            "  2) which tokens in *your input* activate it.\n"
                            "- A single feature can sometimes be **polysemantic** (fires on multiple unrelated patterns), "
                            "especially if the SAE is small or sparsity is weak.\n\n"
                            "**What 'Position' means:**\n"
                            "- The position is a **token index** (0-based). `-1` means the **last token**.\n"
                            "- The activations shown are computed at the SAE’s hook point (e.g. `blocks.6.hook_resid_pre`).\n\n"
                            "**Per-token view (if enabled):**\n"
                            "- Shows the top features for *each* token position (useful for seeing where features fire across the sentence).\n\1"
                            "- We also embed Neuronpedia dashboards for selected features. These dashboards show corpus-level information (such as top activating examples, explanations, and activation statistics) which helps attach semantic meaning to a feature beyond this single input."
                        )

                    st.write(f"**Model:** {outputs.get('model')}")
                    st.write(f"**SAE:** {outputs.get('release')} / {outputs.get('sae_id')}")
                    st.write(f"**Position:** {outputs.get('position')}")

                    toks = outputs.get("tokens", [])
                    pos = outputs.get("position", 0)

                    if isinstance(toks, str):
                        toks = [toks]  # prevent char-by-char enumerate

                    if toks:
                        st.caption("Tokenization (index:token)")
                        st.code(" ".join([f"{i}:{t}" for i, t in enumerate(toks)]))

                        # nice: highlight selected position
                        if isinstance(pos, int) and 0 <= pos < len(toks):
                            st.caption("Selected token position")
                            st.markdown(" ".join([f"**[{t}]**" if i == pos else t for i, t in enumerate(toks)]))

                    st.markdown("### Top activating SAE features at this position")
                    df = pd.DataFrame(outputs.get("top_features", []))
                    st.dataframe(df, use_container_width=True)

                    # Optional: bar plot
                    figs = None
                    if not df.empty and "activation" in df.columns and "feature_id" in df.columns:
                        fig = plt.figure()
                        plt.bar(range(len(df)), df["activation"].tolist())
                        plt.xticks(range(len(df)), df["feature_id"].astype(str).tolist(), rotation=45, ha="right")
                        plt.ylabel("SAE feature activation")
                        plt.title("Top SAE features at selected token position")
                        plt.tight_layout()
                        st.pyplot(fig)
                        figs = {f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_top_features.png": fig}

                    if outputs.get("per_token"):
                        st.markdown("### Per-token top features (k=5)")
                        st.json(outputs.get("per_token_top", [])[:20])

                    # Downloads before embeds (embeds aren't downloadable anyway)
                    render_downloads(outputs, selected_item=selected_item, figs=figs)

                    # --- Neuronpedia integration (Level 1) -
                    # 
                    # --
                    print("neuronpedia") 
                    np = outputs.get("neuronpedia", {}) or {}
                    if np.get("enabled") and np.get("feature_urls"):
                        with st.expander("🧠 Neuronpedia feature dashboards", expanded=False):
                            st.caption(
                                "These dashboards are hosted on Neuronpedia and help interpret SAE features "
                                "(example contexts, explanations, and activation tests)."
                            )

                            max_n = min(10, len(np["feature_urls"]))
                            slider_key = f"np_show_n__{outputs.get('sae_id','na')}__pos{pos}"
                            show_n = st.slider("How many dashboards to embed", 1, max_n, min(3, max_n), key=slider_key)

                            for item in np["feature_urls"][:show_n]:
                                fid = item["feature_id"]
                                url = item["url"]
                                st.markdown(f"#### Feature {fid}")
                                components.iframe(url, height=560, scrolling=True)
                    else:
                        st.caption("Neuronpedia dashboards not available for this SAE release / id.")
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
                    tokens = outputs.get("tokens", [])
                    graph = outputs.get("graph_data")  # <-- dict now
                    model_info = outputs.get("model_info")

                    st.caption(
                        f"Model: {outputs.get('model','NA')} · "
                        f"layers={getattr(model_info,'n_layers','NA')} · "
                        f"focus_token={outputs.get('focus_token_index','NA')} · "
                        f"threshold={outputs.get('threshold','NA')}"
                    )

                    if not graph or not isinstance(graph, dict):
                        st.error("graph_data is missing or not a dict. Showing raw outputs:")
                        st.json(outputs, expanded=False)
                    elif not graph.get("edges"):
                        st.warning("Graph has no edges (try lowering threshold).")
                        st.json(graph, expanded=False)
                    else:
                        with st.expander("Tokenization (index:token)", expanded=False):
                            st.code(" ".join([f"{i}:{t}" for i, t in enumerate(tokens)]))

                        n_layers = int(getattr(model_info, "n_layers", 0) or 0)
                        render_meta_graph_svg(tokens=tokens, graph=graph, n_layers=n_layers, height_px=720)

                    render_downloads(outputs, selected_item=selected_item)

                elif outputs and outputs.get("plugin") == "embedding_pca_layers" and outputs.get("projected"):
                    st.subheader("Result")
                    # --- explainer ---
                    with st.expander("ℹ️ How to read this PCA view", expanded=True):
                        st.write(
                            "- We project each token’s vector into PCA space.\n"
                            "- **Single basis**: PCA is fit once (default: last layer) and reused → plots are comparable across layers.\n"
                            "- **Per-layer basis**: PCA is fit separately per layer → shows within-layer structure but axes are not comparable.\n"
                            "- Tokens are labeled by their tokenizer output; subword tokens may look like 'Ġword' (GPT-2) or '##ing' (BERT).\n"
                            "- In 3D, labels can be occluded; hover always shows token strings."
                        )

                    st.write(f"**Model:** {outputs.get('model','NA')}")
                    params = outputs.get("params", {}) or {}
                    st.caption(
                        f"basis_mode={params.get('basis_mode','NA')} · "
                        f"fit_on={params.get('single_basis_fit_on','NA')} · "
                        f"max_length={params.get('max_length','NA')} · "
                        f"drop_special_tokens={params.get('drop_special_tokens','NA')}"
                    )

                    # --- tokenization preview ---
                    toks = outputs.get("tokens", []) or []
                    if toks:
                        with st.expander("Tokenization (index:token)", expanded=False):
                            st.code(" ".join([f"{i}:{t}" for i, t in enumerate(toks)]))

                    projected = outputs["projected"]
                    max_layer = len(projected) - 1

                    # nicer default: show last layer
                    layer_idx = st.slider(
                        "Layer index (includes embeddings at 0)",
                        0,
                        max_layer,
                        max_layer,
                        key="pca_layers__layer_idx",
                    )

                    layer_obj = projected[layer_idx]
                    df = pd.DataFrame(layer_obj.get("rows", []))

                    # pca info differs depending on mode
                    pca_info = layer_obj.get("pca_info", {}) or {}
                    evr = pca_info.get("explained_variance_ratio", None)
                    if evr and isinstance(evr, (list, tuple)) and len(evr) >= 2:
                        st.caption(
                            f"PCA variance explained: PC1={float(evr[0]):.3f}, PC2={float(evr[1]):.3f} "
                            f"(method={pca_info.get('method','NA')}, fit_on={pca_info.get('fit_on','NA')})"
                        )
                    else:
                        st.caption(f"PCA: method={pca_info.get('method','NA')} · fit_on={pca_info.get('fit_on','NA')}")

                    if df.empty:
                        st.warning("No PCA rows returned.")
                        st.json(layer_obj, expanded=False)
                    else:
                        # show cols depending on pc3 availability
                        cols = ["i", "token", "token_id", "pc1", "pc2"] + (["pc3"] if "pc3" in df.columns else [])
                        st.dataframe(df[cols], use_container_width=True)

                        # --- plot controls ---
                        c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 1.2], gap="medium")
                        with c1:
                            show_labels_2d = st.checkbox("Label points with tokens (2D)", value=True, key="pca_layers__labels_2d")
                        with c2:
                            label_every_2d = st.slider("2D label every N tokens", 1, 8, 1, key="pca_layers__label_every_2d")
                        with c3:
                            point_size = st.slider("Point size", 10, 80, 35, key="pca_layers__ptsize")
                        with c4:
                            show_3d = st.checkbox("Show interactive 3D (drag)", value=True, key="pca_layers__show_3d")

                        # -------------------------
                        # 2D scatter (matplotlib)
                        # -------------------------
                        fig = plt.figure()
                        plt.scatter(df["pc1"].values, df["pc2"].values, s=int(point_size))
                        plt.xlabel("PC1")
                        plt.ylabel("PC2")
                        plt.title(f"Token representations in PCA space — layer {layer_idx}")

                        if show_labels_2d:
                            for _, r in df.iterrows():
                                if int(r["i"]) % int(label_every_2d) != 0:
                                    continue
                                plt.text(float(r["pc1"]), float(r["pc2"]), str(r["token"]), fontsize=8)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # -------------------------
                        # 3D scatter (plotly, draggable) + token strings (hover + optional visible labels)
                        # -------------------------
                        if show_3d:
                            if "pc3" not in df.columns:
                                st.info(
                                    "3D view requires `pc3` in the plugin outputs. "
                                    "Update the PCA plugin to return 3 components (pc1, pc2, pc3) for each layer."
                                )
                            else:
                                # extra UI for 3D labeling
                                d1, d2, d3 = st.columns([1.0, 1.0, 1.2], gap="medium")
                                with d1:
                                    show_3d_labels = st.checkbox("Show token labels in 3D", value=False, key="pca_layers__3d_labels")
                                with d2:
                                    label_every_3d = st.slider("3D label every N tokens", 1, 12, 3, key="pca_layers__label_every_3d")
                                with d3:
                                    marker_size_3d = st.slider("3D marker size", 2, 12, 5, key="pca_layers__marker_size_3d")

                                df3 = df.copy()
                                if show_3d_labels:
                                    df3["text_label"] = df3.apply(
                                        lambda r: str(r["token"]) if (int(r["i"]) % int(label_every_3d) == 0) else "",
                                        axis=1,
                                    )
                                else:
                                    df3["text_label"] = ""

                                # IMPORTANT: hover_name ensures token string shows on hover
                                fig3d = px.scatter_3d(
                                    df3,
                                    x="pc1",
                                    y="pc2",
                                    z="pc3",
                                    hover_name="token",
                                    hover_data={"i": True, "token_id": True, "pc1": ":.4f", "pc2": ":.4f", "pc3": ":.4f"},
                                )

                                # IMPORTANT: mode markers+text is what makes labels visible in 3D
                                fig3d.update_traces(
                                    mode="markers+text" if show_3d_labels else "markers",
                                    text=df3["text_label"],
                                    textposition="top center",
                                    marker=dict(size=int(marker_size_3d)),
                                    hovertemplate=(
                                        "<b>%{hovertext}</b><br>"
                                        "i=%{customdata[0]}<br>"
                                        "token_id=%{customdata[1]}<br>"
                                        "pc1=%{x:.4f}<br>"
                                        "pc2=%{y:.4f}<br>"
                                        "pc3=%{z:.4f}<extra></extra>"
                                    ),
                                )

                                fig3d.update_layout(
                                    height=720,
                                    title=f"Token representations in 3D PCA space — layer {layer_idx}",
                                    margin=dict(l=0, r=0, t=50, b=0),
                                )

                                st.plotly_chart(fig3d, use_container_width=True)

                        # ✅ Downloads (2D plot as PNG; JSON always available via render_downloads)
                        figs_to_download = {
                            f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_pca_layer_{layer_idx}_2d.png": fig
                        }
                        render_downloads(outputs, selected_item=selected_item, figs=figs_to_download)


                elif outputs and outputs.get("plugin") == "linear_cka_layers" and outputs.get("cka_matrix"):
                    st.subheader("Result")
                    with st.expander("ℹ️ How to read Linear CKA", expanded=True):
                        st.write(
                            "- **Linear CKA** measures similarity between two representation sets (here: token vectors) from different layers.\n"
                            "- Values are in **[0, 1]** (higher = more similar).\n"
                            "- We compute it using feature-centering and the linear CKA formula based on Frobenius norms.\n"
                            "- Layer labels: **emb** = embedding output, **Lk** = transformer block k."
                        )

                    st.write(f"**Model:** {outputs.get('model','NA')} (arch={outputs.get('arch_used','NA')})")
                    params = outputs.get("params", {}) or {}
                    st.caption(
                        f"token_subset={params.get('token_subset','NA')} · "
                        f"max_tokens_used={params.get('max_tokens_used','NA')} · "
                        f"max_length={params.get('max_length','NA')} · "
                        f"compute_on_cpu={params.get('compute_on_cpu','NA')}"
                    )

                    toks = outputs.get("tokens", []) or []
                    used = outputs.get("token_indices_used", []) or []
                    if toks and used:
                        with st.expander("Token indices used (index:token)", expanded=False):
                            st.code(" ".join([f"{i}:{toks[i]}" for i in used if 0 <= i < len(toks)]))

                    import numpy as np
                    import plotly.express as px
                    import matplotlib.pyplot as plt
                    import pandas as pd

                    M = np.array(outputs["cka_matrix"], dtype=float)
                    labels = outputs.get("layer_labels", [str(i) for i in range(M.shape[0])])

                    # Plotly interactive heatmap (VISIBLE)
                    fig = px.imshow(
                        M,
                        x=labels,
                        y=labels,
                        zmin=0.0,
                        zmax=1.0,
                        color_continuous_scale="viridis",
                        aspect="auto",
                        title="Linear CKA similarity across layers",
                    )
                    fig.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabular view
                    df = pd.DataFrame(M, index=labels, columns=labels)
                    with st.expander("Matrix values", expanded=False):
                        st.dataframe(df, use_container_width=True)

                    # --- Hidden matplotlib heatmap (for download only) ---
                    fig2 = plt.figure()
                    plt.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
                    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
                    plt.yticks(range(len(labels)), labels)
                    plt.title("Linear CKA similarity across layers")
                    plt.tight_layout()

                    # DO NOT call st.pyplot(fig2)  ← this removes the yellow display

                    render_downloads(
                        outputs,
                        selected_item=selected_item,
                        figs={
                            f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_cka_heatmap.png": fig2
                        },
                    )

                elif outputs and outputs.get("plugin") == "cca_layers" and outputs.get("cca_matrix"):
                    st.subheader("Result")
                    with st.expander("ℹ️ How to read CCA", expanded=True):
                        st.write(
                            "- **CCA** measures linear similarity between two representation sets (token vectors) from different layers.\n"
                            "- Values are in **[0, 1]** (higher = more similar).\n"
                            "- We compute it via Google's SVCCA `cca_core.get_cca_similarity` and return **mean canonical correlation**.\n"
                            "- Because SVCCA-CCA requires `neurons < tokens`, we SVD-reduce the neuron dimension to `tokens-1` when needed.\n"
                            "- Layer labels: **emb** = embedding output, **Lk** = transformer block k."
                        )

                    st.write(f"**Model:** {outputs.get('model','NA')} (arch={outputs.get('arch_used','NA')})")
                    params = outputs.get("params", {}) or {}
                    st.caption(
                        f"token_subset={params.get('token_subset','NA')} · "
                        f"max_tokens_used={params.get('max_tokens_used','NA')} · "
                        f"max_length={params.get('max_length','NA')} · "
                        f"compute_on_cpu={params.get('compute_on_cpu','NA')} · "
                        f"svd_reduce_to={params.get('svd_reduce_to','NA')} · "
                        f"epsilon={params.get('epsilon','NA')}"
                    )

                    toks = outputs.get("tokens", []) or []
                    used = outputs.get("token_indices_used", []) or []
                    if toks and used:
                        with st.expander("Token indices used (index:token)", expanded=False):
                            st.code(" ".join([f"{i}:{toks[i]}" for i in used if 0 <= i < len(toks)]))

                    import numpy as np
                    import plotly.express as px
                    import matplotlib.pyplot as plt
                    import pandas as pd

                    M = np.array(outputs["cca_matrix"], dtype=float)
                    labels = outputs.get("layer_labels", [str(i) for i in range(M.shape[0])])

                    # Visible interactive heatmap
                    fig = px.imshow(
                        M,
                        x=labels,
                        y=labels,
                        zmin=0.0,
                        zmax=1.0,
                        color_continuous_scale="viridis",
                        aspect="auto",
                        title="CCA similarity across layers (mean canonical correlation)",
                    )
                    fig.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    # Values table
                    df = pd.DataFrame(M, index=labels, columns=labels)
                    with st.expander("Matrix values", expanded=False):
                        st.dataframe(df, use_container_width=True)

                    # Download-only matplotlib heatmap
                    fig2 = plt.figure()
                    plt.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
                    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
                    plt.yticks(range(len(labels)), labels)
                    plt.title("CCA similarity across layers")
                    plt.tight_layout()

                    render_downloads(
                        outputs,
                        selected_item=selected_item,
                        figs={f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_cca_heatmap.png": fig2},
                    )



                elif outputs and outputs.get("plugin") == "attention_rollout" and outputs.get("token_scores"):
                    st.subheader("Result")
                    with st.expander("ℹ️ How to read Attention Rollout", expanded=True):
                        st.write(
                            "- Attention rollout multiplies attention matrices across layers (with residual connections) to estimate token-to-token influence.\n"
                            "- The scores below show which **source tokens** contribute most to the selected **target token** through attention pathways.\n"
                            "- Scores are normalized to [0,1] for display."
                        )

                    st.write(f"**Model:** {outputs.get('model','NA')}")
                    st.write(f"**Target token index:** {outputs.get('target_token_index','NA')}")

                    toks = outputs.get("tokens", [])
                    scores = outputs.get("token_scores", [])

                    # Highlight tokens (treat as nonnegative importance)
                    render_token_highlight(
                        tokens=toks,
                        scores=scores,          # all >= 0
                        title="🖍️ Highlighted text (attention rollout relevance)",
                        max_abs=1.0,
                    )

                    with st.expander("Top source tokens", expanded=False):
                        st.dataframe(pd.DataFrame(outputs.get("top_sources", [])), use_container_width=True)

                    render_downloads(outputs, selected_item=selected_item)


                elif outputs and outputs.get("plugin") == "probing_binary_examples":
                    st.subheader("Result")
                    with st.expander("ℹ️ How to read Probing results", expanded=True):
                        st.write(
                            "- A **probe** is a simple linear classifier trained on hidden representations from a specific model layer.\n"
                            "- If performance is high, it suggests that the probed layer encodes information that linearly separates the two classes.\n"
                            "- We extract hidden states from the selected layer, pool them into a single vector per example, "
                            "and train a linear classifier (e.g., logistic regression).\n"
                            "- Results are reported using **Stratified Cross-Validation**, so each fold trains and tests on different splits.\n\n"
                            "**Metrics explained:**\n"
                            "- **Accuracy**: overall proportion of correct predictions.\n"
                            "- **Balanced Accuracy**: accounts for class imbalance (recommended metric).\n"
                            "- **Macro F1**: harmonic mean of precision and recall, averaged across classes.\n"
                            "- The **confusion matrix** shows how many positives/negatives were correctly or incorrectly predicted.\n\n"
                            "⚠️ **Important:** This demo uses a small number of examples (e.g., 30 vs 30 by default). "
                            "While useful for experimentation, robust scientific conclusions require substantially larger datasets.\n"
                            "Small datasets may lead to unstable or over-optimistic estimates."
                        )


                    st.write(f"**Model:** {outputs.get('model','NA')}")
                    st.write(f"**Device:** {outputs.get('device','NA')}")
                    st.caption(f"n_pos={outputs.get('n_pos')} · n_neg={outputs.get('n_neg')} · total={outputs.get('total')}")

                    params = outputs.get("params", {}) or {}
                    with st.expander("Parameters", expanded=False):
                        st.json(params, expanded=False)

                    metrics = outputs.get("metrics", {}) or {}
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Accuracy (mean ± std)",
                            f"{metrics.get('accuracy_mean', 0.0):.3f}",
                            delta=f"± {metrics.get('accuracy_std', 0.0):.3f}",
                        )
                    with c2:
                        st.metric(
                            "Balanced Acc (mean ± std)",
                            f"{metrics.get('balanced_accuracy_mean', 0.0):.3f}",
                            delta=f"± {metrics.get('balanced_accuracy_std', 0.0):.3f}",
                        )
                    with c3:
                        st.metric(
                            "Macro F1 (mean ± std)",
                            f"{metrics.get('macro_f1_mean', 0.0):.3f}",
                            delta=f"± {metrics.get('macro_f1_std', 0.0):.3f}",
                        )

                    folds = outputs.get("folds", []) or []
                    if folds:
                        st.markdown("### Cross-validation folds")
                        st.dataframe(pd.DataFrame(folds), use_container_width=True)

                    cm = (outputs.get("confusion_matrix") or {})
                    mat = cm.get("matrix")
                    labels = cm.get("labels", ["neg(0)", "pos(1)"])
                    if mat:
                        st.markdown("### Confusion matrix (aggregated over CV predictions)")
                        df_cm = pd.DataFrame(mat, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])
                        st.dataframe(df_cm, use_container_width=True)

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