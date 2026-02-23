# ✅ UPDATED DESIGN: Hard constraints vs Preferences (ranking only)
# Paste this as your app.py (it’s your code with minimal, targeted edits).

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


from text_to_score import rank_methods
#from toolkits.captum_classifier import CaptumClassifierAttribution
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
from toolkits.inseq_proxy_http import InseqDecoderIG_HTTP, InseqEncDecIG_HTTP

from toolkits.meta_transparency import MetaTransparencyGraph  # adjust import path

import tempfile
import os
from pyvis.network import Network

from toolkits.PCAViz import EmbeddingPCALayers
import plotly.express as px
import plotly.graph_objects as go
from toolkits.linear_cka import LinearCKALayers

from toolkits.cca_layers import CCALayers

def captum_method_explainer_text(algo: str, params: Dict[str, Any]) -> str:
    algo = (algo or "").strip()

    if algo == "IntegratedGradients":
        n_steps = params.get("n_steps", "NA")
        return (
            "### ℹ️ How to read Integrated Gradients (IG)\n"
            "- **What it is**: IG measures how much each token changes the target class score when moving from a **baseline** input to your input.\n"
            "- **Baseline**: here it’s the model’s **[PAD] embedding** (or zeros if PAD doesn’t exist).\n"
            "- **Interpretation**: tokens with large **positive** values push the model **toward** the explained label; large **negative** values push it **away**.\n"
            "- **Stability knob**: `n_steps` controls the approximation quality. More steps → smoother but slower.\n"
            f"- **Your run**: `n_steps={n_steps}`.\n"
        )

    if algo == "Saliency":
        return (
            "### ℹ️ How to read Saliency\n"
            "- **What it is**: Saliency uses the **gradient of the target logit w.r.t. the input embeddings**.\n"
            "- **Interpretation**: high magnitude means the label score is **sensitive** to that token.\n"
            "- **Caveat**: gradients can be **noisy** and sometimes saturate; saliency is fast but less stable than IG.\n"
            "- **Tip**: if results look spiky, try IG (more stable) or average over multiple runs/seeds.\n"
        )

    if algo == "DeepLift":
        return (
            "### ℹ️ How to read DeepLift\n"
            "- **What it is**: DeepLift compares activations to a **baseline** and attributes differences back to inputs.\n"
            "- **Baseline**: here it’s the model’s **[PAD] embedding** (or zeros if PAD doesn’t exist).\n"
            "- **Interpretation**: positive pushes toward the explained label; negative pushes away.\n"
            "- **Why use it**: can produce **sharper** attributions than plain gradients when gradients saturate.\n"
        )

    # fallback
    return (
        "### ℹ️ How to read this attribution\n"
        "- Positive pushes toward the explained label; negative pushes away.\n"
        "- Larger magnitude = larger influence.\n"
    )


import html as _html

def render_token_highlight(
    tokens: List[str],
    scores: List[float],
    *,
    title: str = "🖍️ Token highlights",
    max_abs: float | None = None,
):
    """
    Renders tokens with background intensity based on signed attribution scores.
    Positive = blue tint, negative = red tint. Intensity ~ |score|.

    tokens: list of tokens (already merged if you do that)
    scores: list of floats in [-1,1] or any scale; will normalize by max_abs if provided/needed.
    """
    if not tokens or not scores or len(tokens) != len(scores):
        st.caption("No token highlight available.")
        return

    if max_abs is None:
        max_abs = max((abs(s) for s in scores), default=1e-9) or 1e-9

    # Build HTML spans
    spans = []
    for t, s in zip(tokens, scores):
        # skip special tokens if they slip in
        if t in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        # normalize into [0,1]
        a = min(1.0, abs(float(s)) / (max_abs + 1e-9))

        # Use RGBA so we can modulate alpha (intensity)
        # Positive => blue-ish, Negative => red-ish
        if s >= 0:
            bg = f"rgba(37, 99, 235, {0.08 + 0.55*a:.3f})"   # blue
        else:
            bg = f"rgba(220, 38, 38, {0.08 + 0.55*a:.3f})"   # red

        # Border helps visibility for small alpha
        style = (
            "display:inline-block;"
            "padding:2px 4px;"
            "margin:2px 2px;"
            "border-radius:6px;"
            "border:1px solid rgba(0,0,0,0.06);"
            f"background:{bg};"
            "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;"
            "font-size:0.92rem;"
            "line-height:1.35;"
        )

        spans.append(f"<span style='{style}' title='{float(s):.4f}'>{_html.escape(t)}</span>")

    st.markdown(f"#### {title}")
    st.markdown(
        "<div style='padding:10px; border:1px solid #e5e7eb; border-radius:12px; background:#fff;'>"
        + "".join(spans)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.caption("Blue = pushes toward explained label. Red = pushes away. Hover a token to see the score.")


def render_captum_result(outputs: Dict[str, Any], selected_item: Dict[str, Any] | None):
    pred = outputs.get("predicted", {}) or {}
    tgt = outputs.get("target", {}) or {}
    params = outputs.get("params", {}) or {}

    algo = outputs.get("algorithm", "NA")
    model_name = outputs.get("model", "NA")

    st.subheader("Result")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Algorithm:** {algo}")
    st.write(f"**Prediction:** {pred.get('label', pred.get('idx', 'NA'))}")
    st.write(f"**Label being explained:** {tgt.get('label', tgt.get('idx', 'NA'))}")

    # Method-specific explanation (✅ differs per method)
    with st.expander("ℹ️ How to read this explanation", expanded=True):
        st.markdown(captum_method_explainer_text(algo, params))

    # Optional: show IG steps inline too
    if algo == "IntegratedGradients" and "n_steps" in params:
        st.caption(f"Integrated Gradients steps: {params.get('n_steps')}")

    st.caption("Each bar shows how strongly a token contributes to the explained label (normalized).")

    df = pd.DataFrame(outputs["attributions"])
    df_plot = df[~df["token"].isin(["[CLS]", "[SEP]", "[PAD]"])].copy()

    st.dataframe(df_plot[["token", "attr_raw", "attr_norm"]], use_container_width=True)

    fig = plt.figure()
    plt.bar(range(len(df_plot)), df_plot["attr_norm"].tolist())
    plt.xticks(range(len(df_plot)), df_plot["token"].tolist(), rotation=45, ha="right")
    plt.ylabel("Attribution (normalized)")
    plt.title(f"{algo} token attributions")
    plt.tight_layout()
    st.pyplot(fig)

    show_highlight = st.checkbox("Show highlighted text", value=True, key="captum_show_highlight")
    if show_highlight:
        render_token_highlight(
            tokens=df_plot["token"].tolist(),
            scores=df_plot["attr_norm"].tolist(),  # already normalized to [-1, 1]
            title="🖍️ Highlighted text (by attribution)",
            max_abs=1.0,  # because attr_norm is normalized
        )

    render_downloads(
        outputs,
        selected_item=selected_item,
        figs={f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_attribution_plot.png": fig},
    )


def _to_node_id(n: Any) -> str:
    # pyvis node ids must be str/int; make it stable
    return str(n)

def _infer_edges_and_nodes(graph_obj: Any):
    """
    Best-effort conversion of Meta graph objects to (nodes, edges).
    Works for:
      - dict with keys like nodes/edges
      - list of edge dicts
    You may need to tweak this once you see your actual graph_data structure.
    """
    nodes = {}
    edges = []

    if isinstance(graph_obj, dict):
        # Common patterns
        if "nodes" in graph_obj and "edges" in graph_obj:
            for n in graph_obj["nodes"]:
                nid = _to_node_id(n.get("id", n.get("name", n)))
                nodes[nid] = n
            for e in graph_obj["edges"]:
                src = _to_node_id(e.get("source", e.get("src", e.get("from"))))
                dst = _to_node_id(e.get("target", e.get("dst", e.get("to"))))
                w = e.get("weight", e.get("value", 1.0))
                edges.append((src, dst, float(w)))
                if src not in nodes: nodes[src] = {"id": src, "label": src}
                if dst not in nodes: nodes[dst] = {"id": dst, "label": dst}
            return nodes, edges

        # If it's already an edge list dict
        if "edges" in graph_obj and isinstance(graph_obj["edges"], list):
            for e in graph_obj["edges"]:
                if isinstance(e, dict):
                    src = _to_node_id(e.get("source", e.get("src", e.get("from"))))
                    dst = _to_node_id(e.get("target", e.get("dst", e.get("to"))))
                    w = e.get("weight", e.get("value", 1.0))
                    edges.append((src, dst, float(w)))
                    nodes.setdefault(src, {"id": src, "label": src})
                    nodes.setdefault(dst, {"id": dst, "label": dst})
            return nodes, edges

    # If graph_obj is list of dict edges
    if isinstance(graph_obj, list):
        for e in graph_obj:
            if isinstance(e, dict):
                src = _to_node_id(e.get("source", e.get("src", e.get("from"))))
                dst = _to_node_id(e.get("target", e.get("dst", e.get("to"))))
                w = e.get("weight", e.get("value", 1.0))
                edges.append((src, dst, float(w)))
                nodes.setdefault(src, {"id": src, "label": src})
                nodes.setdefault(dst, {"id": dst, "label": dst})
        if edges:
            return nodes, edges

    return nodes, edges


def render_meta_flow_pyvis(outputs: Dict[str, Any]):
    """
    Renders outputs from MetaTransparencyGraph using PyVis.
    - outputs["graph_data"] is expected to be a list (typically per-token).
    - We let the user choose which token graph to inspect.
    """
    tokens = outputs.get("tokens", [])
    graph_data = outputs.get("graph_data", [])

    if not graph_data:
        st.warning("No graph_data returned.")
        return

    st.caption("Choose which token-position graph to visualize (Meta returns per-token route graphs).")
    max_idx = len(graph_data) - 1
    default_idx = max_idx
    if isinstance(tokens, list) and tokens:
        default_idx = min(default_idx, len(tokens) - 1)

    idx = st.slider("Token index", 0, max_idx, default_idx, key="meta_flow_token_idx")

    # Show tokenization context
    if isinstance(tokens, list) and tokens:
        preview = " ".join([f"{i}:{t}" for i, t in enumerate(tokens)])
        with st.expander("Tokenization (index:token)", expanded=False):
            st.code(preview)

    g = graph_data[idx]

    nodes, edges = _infer_edges_and_nodes(g)

    if not edges:
        st.warning(
            "Could not infer edges from graph_data. "
            "Show the raw JSON below and we’ll adapt the converter to Meta’s exact structure."
        )
        st.json(g, expanded=False)
        return

    # Build interactive network
    net = Network(height="720px", width="100%", directed=True, notebook=False)

    # Add nodes with optional labels
    for nid, nobj in nodes.items():
        label = nobj.get("label") if isinstance(nobj, dict) else None
        if not label:
            label = str(nobj.get("id", nid)) if isinstance(nobj, dict) else str(nid)
        title = None
        if isinstance(nobj, dict):
            # hover tooltip: keep compact
            title = "<br/>".join([f"{k}: {nobj[k]}" for k in list(nobj.keys())[:8]])
        net.add_node(nid, label=label, title=title)

    # Add edges
    for (src, dst, w) in edges:
        # value influences thickness in pyvis
        net.add_edge(src, dst, value=max(0.1, abs(w)), title=f"weight={w:.4g}")

    # Basic layout tuning
    net.repulsion(node_distance=170, central_gravity=0.1, spring_length=140, spring_strength=0.04, damping=0.25)

    # Render to HTML and embed
    html = net.generate_html()

    # Make it more streamlit-friendly (avoid external fetches if possible)
    components.html(html, height=760, scrolling=True)



import html
import re
import streamlit.components.v1 as components

_NODE_RE = re.compile(r"^(X0|A|M|I)(\d+)?_(\d+)$")  # X0_3 OR A6_3 etc.

def _parse_node(node_id: str):
    m = _NODE_RE.match(node_id)
    if not m:
        return None
    typ = m.group(1)
    layer = -1 if typ == "X0" else int(m.group(2))
    tok = int(m.group(3))
    return typ, layer, tok

def render_meta_graph_svg(tokens: list[str], graph: dict, n_layers: int, height_px: int = 720):
    x_step, y_step = 34, 34
    left_pad, top_pad = 120, 25
    type_row = {"X0": 0, "A": 1, "M": 2, "I": 3}

    def xy(node_id: str):
        p = _parse_node(node_id)
        if not p:
            return None
        typ, layer, tok = p
        layer_row = layer + 1  # X0(-1)->0, L0->1 ...
        x = left_pad + tok * x_step
        y = top_pad + (layer_row * 4 + type_row[typ]) * y_step
        return x, y

    # positions
    pos = {}
    for nid in graph.get("nodes", []):
        nid = str(nid)
        pt = xy(nid)
        if pt:
            pos[nid] = pt

    edges = graph.get("edges", [])
    if not edges:
        st.warning("Graph has no edges (try lowering threshold).")
        return

    max_w = max((abs(float(e.get("weight", 0.0))) for e in edges), default=1.0)

    line_elems = []
    for e in edges:
        u, v = str(e.get("src")), str(e.get("dst"))
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        w = float(e.get("weight", 0.0))
        sw = 0.6 + 4.2 * (abs(w) / (max_w + 1e-9))
        op = 0.15 + 0.75 * (abs(w) / (max_w + 1e-9))
        stroke = "#2563eb" if w >= 0 else "#dc2626"
        line_elems.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}" stroke-opacity="{op:.2f}" />'
        )

    dot_elems = [f'<circle cx="{x}" cy="{y}" r="3.2" fill="#111827" fill-opacity="0.85" />'
                 for (x, y) in pos.values()]

    # token labels
    base_y = top_pad + ((n_layers + 1) * 4 + 4) * y_step
    tok_labels = []
    for i, t in enumerate(tokens):
        tx = left_pad + i * x_step
        tok_labels.append(
            f'<text x="{tx}" y="{base_y}" font-size="10" fill="#111827" text-anchor="middle">{html.escape(t)}</text>'
        )

    # y labels
    ylabels = []
    for layer in range(-1, n_layers):
        layer_row = layer + 1
        ylabels.append(
            f'<text x="10" y="{top_pad + (layer_row*4+0)*y_step + 4}" font-size="11" fill="#374151">'
            f'{"X0" if layer==-1 else "L"+str(layer)}</text>'
        )

    width = left_pad + max(1, len(tokens)) * x_step + 40
    height = max(height_px, int(base_y + 60))

    svg = (
        f'<svg width="{width}" height="{height}" style="background:white; border:1px solid #e5e7eb; border-radius:12px;">'
        + "".join(line_elems)
        + "".join(dot_elems)
        + "".join(ylabels)
        + "".join(tok_labels)
        + "</svg>"
    )

    components.html(svg, height=min(height, 900), scrolling=True)


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
    plugin7 = InseqDecoderIG_HTTP()
    plugin8 = InseqEncDecIG_HTTP()
    plugin9 = MetaTransparencyGraph()
    plugin10 = CaptumSaliencyClassifierAttribution()
    plugin11 = CaptumDeepLiftClassifierAttribution()
    plugin12 = EmbeddingPCALayers()
    plugin13 = LinearCKALayers()
    plugin14 = CCALayers()


    return {
        plugin1.id: plugin1,
        plugin2.id: plugin2,
        plugin3.id: plugin3,
        plugin4.id: plugin4,
        plugin5.id: plugin5,
        plugin6.id: plugin6,
        plugin7.id: plugin7,
        plugin8.id: plugin8,
        plugin9.id: plugin9,
        plugin10.id: plugin10,
        plugin11.id: plugin11, 
        plugin12.id: plugin12, 
        plugin13.id: plugin13, 
        plugin14.id: plugin14,

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
    "task": "NA",
    "access": "NA",
    "arch": "NA",
    "scope": "NA",
    "granularity": "NA",
    "goal": "NA",
    "fidelity": "NA",
    "format": "NA",
}

# Hard vs preference dims (UI clarity + logic)
HARD_DIMS = ["task", "access", "arch", "scope"]
PREF_DIMS = ["granularity", "goal", "fidelity", "format"]


# -------------------------
# Download helpers (NEW)
# -------------------------
def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.read()


def _make_prefix(selected_item: Dict[str, Any] | None, plugin_name: str) -> str:
    # short + stable filename prefix
    method = (selected_item or {}).get("name") or plugin_name or "result"
    method = re.sub(r"[^a-zA-Z0-9_\-]+", "_", method).strip("_")
    return method or "result"


def render_downloads(
    outputs: Dict[str, Any],
    selected_item: Dict[str, Any] | None = None,
    figs: Dict[str, Any] | None = None,
):
    """
    Shows an expander with download buttons:
      - raw outputs JSON (always)
      - plugin-specific CSV / HTML when present
      - optional ZIP containing JSON + CSV/HTML + any figures as PNG
    `figs`: mapping filename -> matplotlib figure
    """
    if not outputs:
        return

    plugin = outputs.get("plugin", "unknown")
    stamp = _now_stamp()
    prefix = _make_prefix(selected_item, plugin)

    extra_files: Dict[str, bytes] = {}
    # always include raw JSON in the ZIP
    extra_files[f"{prefix}_{plugin}_{stamp}.json"] = _to_json_bytes(outputs)

    with st.expander("⬇️ Download results", expanded=False):
        # --- Raw JSON (always)
        st.download_button(
            "Download raw outputs (JSON)",
            data=_to_json_bytes(outputs),
            file_name=f"{prefix}_{plugin}_{stamp}.json",
            mime="application/json",
            use_container_width=True,
        )

        # --- Captum attributions -> CSV
        if outputs.get("attributions"):
            df_attr = pd.DataFrame(outputs["attributions"])
            csv_bytes = df_attr.to_csv(index=False).encode("utf-8")
            fn = f"{prefix}_{plugin}_{stamp}_attributions.csv"
            extra_files[fn] = csv_bytes

            st.download_button(
                "Download attributions (CSV)",
                data=csv_bytes,
                file_name=fn,
                mime="text/csv",
                use_container_width=True,
            )

        # --- DLA components -> CSV
        if outputs.get("plugin") == "direct_logit_attribution" and outputs.get("components"):
            df_comps = pd.DataFrame(outputs["components"])
            csv_bytes = df_comps.to_csv(index=False).encode("utf-8")
            fn = f"{prefix}_{plugin}_{stamp}_components.csv"
            extra_files[fn] = csv_bytes

            st.download_button(
                "Download component contributions (CSV)",
                data=csv_bytes,
                file_name=fn,
                mime="text/csv",
                use_container_width=True,
            )

        # --- Logit lens layers -> flattened CSV
        if outputs.get("plugin") == "logit_lens" and outputs.get("layers"):
            rows = []
            for layer_i, layer_obj in enumerate(outputs["layers"]):
                for item in layer_obj.get("top", []):
                    rows.append(
                        {
                            "layer": layer_i,
                            "token": item.get("token"),
                            "score": item.get("score"),
                            "rank": item.get("rank"),
                        }
                    )
            if rows:
                df_ll = pd.DataFrame(rows)
                csv_bytes = df_ll.to_csv(index=False).encode("utf-8")
                fn = f"{prefix}_{plugin}_{stamp}_top_tokens.csv"
                extra_files[fn] = csv_bytes

                st.download_button(
                    "Download logit-lens top tokens (CSV)",
                    data=csv_bytes,
                    file_name=fn,
                    mime="text/csv",
                    use_container_width=True,
                )

        # --- BertViz HTML
        if outputs.get("plugin") == "bertviz_attention" and outputs.get("html"):
            html_bytes = outputs["html"].encode("utf-8")
            fn = f"{prefix}_{plugin}_{stamp}.html"
            extra_files[fn] = html_bytes

            st.download_button(
                "Download attention visualization (HTML)",
                data=html_bytes,
                file_name=fn,
                mime="text/html",
                use_container_width=True,
            )

        # --- Figures -> PNG (also added to ZIP)
        if figs:
            for fname, fig in figs.items():
                try:
                    png = _fig_to_png_bytes(fig)
                    extra_files[fname] = png
                    st.download_button(
                        f"Download plot: {fname}",
                        data=png,
                        file_name=fname,
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception:
                    pass

        # --- Everything ZIP
        if extra_files:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                for fname, b in extra_files.items():
                    z.writestr(fname, b)
            zbuf.seek(0)

            st.download_button(
                "Download everything (ZIP)",
                data=zbuf.read(),
                file_name=f"{prefix}_{plugin}_{stamp}.zip",
                mime="application/zip",
                use_container_width=True,
            )


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


def feasible(hard: Dict[str, str], m: Dict[str, Any]) -> Tuple[bool, str]:
    """Hard constraints only: exclude tools that cannot work in the requested setting."""
    # Task
    if hard["task"] != "NA":
        tasks = norm_list(m.get("task_input", []))
        if hard["task"] not in tasks and "general_NLP" not in tasks:
            return False, f"task mismatch (needs {tasks or 'NA'})"

    # Access
    if hard["access"] != "NA":
        acc = m.get("access_arch", {}).get("access", "NA")
        accs = norm_list(acc)
        if hard["access"] not in accs and "mixed" not in accs and "NA" not in accs:
            return False, f"access mismatch (method {accs})"

    # Architecture
    if hard["arch"] != "NA":
        arch = m.get("access_arch", {}).get("arch", "NA")
        archs = norm_list(arch)
        if hard["arch"] not in archs and "transformer_general" not in archs and "NA" not in archs:
            return False, f"arch mismatch (method {archs})"

    # Scope
    if hard["scope"] != "NA":
        sc = m.get("target_scope", "NA")
        if sc not in ["NA", "both"] and hard["scope"] != sc:
            return False, f"scope mismatch (method {sc})"

    return True, ""


def score(prefs: Dict[str, str], m: Dict[str, Any]) -> Tuple[int, List[str], List[str]]:
    """
    Preferences-only scoring:
    - returns (score, matched_reasons, mismatched_reasons)
    - mismatches are surfaced in UI to make it obvious these are not filters.
    """
    s = 0
    matched: List[str] = []
    mismatched: List[str] = []

    # Granularity
    if prefs["granularity"] != "NA":
        grans = norm_list(m.get("granularity", []))
        if prefs["granularity"] in grans:
            s += 2
            matched.append("🔍 granularity match")
        else:
            mismatched.append(f"🔍 granularity mismatch (wants {prefs['granularity']}, has {grans or 'NA'})")

    # Goal
    if prefs["goal"] != "NA":
        goals = norm_list(m.get("user_goal_audience", []))
        if prefs["goal"] in goals:
            s += 2
            matched.append("🎯 goal match")
        else:
            mismatched.append(f"🎯 goal mismatch (wants {prefs['goal']}, has {goals or 'NA'})")

    # Format
    if prefs["format"] != "NA":
        fmts = norm_list(m.get("format", []))
        if prefs["format"] in fmts:
            s += 2
            matched.append("🖥️ format match")
        else:
            mismatched.append(f"🖥️ format mismatch (wants {prefs['format']}, has {fmts or 'NA'})")

    # Fidelity
    if prefs["fidelity"] != "NA":
        f = m.get("fidelity", "NA")
        if f == prefs["fidelity"]:
            s += 2
            matched.append("🧪 fidelity match")
        elif f == "mixed":
            s += 1
            matched.append("🧪 fidelity partially supported (tool is mixed)")
            mismatched.append(f"🧪 fidelity preference not exact (wants {prefs['fidelity']}, tool is mixed)")
        else:
            mismatched.append(f"🧪 fidelity mismatch (wants {prefs['fidelity']}, has {f})")

    return s, matched, mismatched



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

    SAE_DEFAULTS = {
        "model_name": "gpt2-small",
        "release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "dtype": "float32",  # safest on CPU
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "text": "The capital of France is",
        "position_index": -1,
        "top_k": 10,
        "per_token": False,
    }

    plugin_id = getattr(plugin, "id", "")
    default_map = SAE_DEFAULTS if plugin_id == "sae_feature_explorer" else {}

    def _k(field_key: str) -> str:
        return f"{plugin_id}__{field_key}"

    for f in plugin.spec():
        f_default = getattr(f, "default", None)  # ✅ SAFE across plugins
        f_help = getattr(f, "help", "")

        # --- TEXTAREA ---
        if f.type == "textarea":
            default_val = default_map.get(f.key, f_default if f_default is not None else "")
            vals[f.key] = st.text_area(
                f.label,
                value=str(default_val),
                help=f_help,
                key=_k(f.key),
            )

        # --- TEXT ---
        elif f.type == "text":
            default_val = default_map.get(f.key, f_default if f_default is not None else "")
            vals[f.key] = st.text_input(
                f.label,
                value=str(default_val),
                help=f_help,
                key=_k(f.key),
            )

        # --- SELECT ---
        elif f.type == "select":
            options = getattr(f, "options", None) or []
            default_val = default_map.get(f.key, f_default if f_default is not None else (options[0] if options else ""))

            index = 0
            if options and default_val in options:
                index = options.index(default_val)

            vals[f.key] = st.selectbox(
                f.label,
                options,
                index=index,
                help=f_help,
                key=_k(f.key),
            )

        # --- NUMBER ---
        elif f.type == "number":
            # Priority order:
            # 1) FieldSpec.default if present
            # 2) plugin-specific defaults (SAE_DEFAULTS)
            # 3) anchor defaults
            # 4) heuristics fallback
            if f_default is not None:
                default = float(f_default)
            elif f.key in default_map:
                default = float(default_map[f.key])
            elif f.key in ANCHOR_DEFAULTS:
                default = float(ANCHOR_DEFAULTS[f.key])
            else:
                if f.key == "n_steps":
                    default = 50
                elif f.key == "max_length":
                    default = 128
                elif f.key == "top_k":
                    default = 10
                elif f.key == "position_index":
                    default = -1
                else:
                    default = 0

            step = 1.0
            if f.key in ("threshold", "delta", "tau"):
                step = 0.05

            vals[f.key] = st.number_input(
                f.label,
                value=float(default),
                step=float(step),
                help=f_help,
                key=_k(f.key),
            )

        # --- CHECKBOX ---
        elif f.type == "checkbox":
            default_val = default_map.get(f.key, bool(f_default) if f_default is not None else False)
            vals[f.key] = st.checkbox(
                f.label,
                value=bool(default_val),
                help=f_help,
                key=_k(f.key),
            )

        else:
            st.warning(f"Unknown field type: {getattr(f, 'type', 'NA')} (field {getattr(f, 'key', 'NA')})")

    return vals















# -------------------------
# Selected tool card
# -------------------------
def _safe(s: str) -> str:
    return re.sub(r"[<>]", "", s or "")

def _chip(text: str):
    st.markdown(
        f"""
        <span class="xai-chip">{_safe(text)}</span>
        """,
        unsafe_allow_html=True,
    )
    

def render_selected_tool_card(selected_item: Dict[str, Any]):
    name = selected_item.get("name", "Selected tool")
    notes = selected_item.get("notes", "")
    meta = selected_item.get("meta", {}) or {}
    desc = selected_item.get("description", {}) or {}

    overview = desc.get("overview") or desc.get("summary") or ""
    funcs = desc.get("main_functionalities", []) or []
    strengths = selected_item.get("strengths", []) or []
    limitations = selected_item.get("limitations", []) or []
    apps = selected_item.get("research_applications", []) or []

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;">
              <div>
                <div style="font-size:1.45rem; font-weight:700; margin-bottom:0.25rem;">
                 🛠️ {_safe(name)}
                </div>
                <div style="color: var(--text-color); font-size:0.98rem; line-height:1.35;">
                  {_safe(overview) if overview else _safe(notes)}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

        chip_map = [
            ("Scope", meta.get("scope")),
            ("Access", meta.get("access")),
            ("Architecture", meta.get("arch")),
            ("Granularity", meta.get("granularity")),
            ("Format", meta.get("format")),
            ("Fidelity", meta.get("fidelity")),
        ]

        any_chip = False
        for k, v in chip_map:
            if not v or v == "NA":
                continue
            any_chip = True
            if isinstance(v, list):
                v = ", ".join([str(x) for x in v if x and x != "NA"])
            _chip(f"{k}: {_safe(str(v))}")
        if not any_chip:
            st.caption("")

        left, right = st.columns([1.2, 1.0], gap="large")

        with left:
            st.markdown("#### ⚙️ Main functionalities")
            if funcs:
                for x in funcs:
                    st.write(f"- {x}")
            else:
                st.caption("No main functionalities provided for this method yet.")
            if notes and overview:
                with st.expander("Extra notes", expanded=False):
                    st.write(notes)

        with right:
            st.markdown("#### 📊 Strengths vs limitations")
            s_col, l_col = st.columns(2, gap="medium")

            with s_col:
                st.markdown("**✅ Strengths**")
                if strengths:
                    for x in strengths:
                        st.markdown(f"<span style='color:var(--success-color)'>• {x}</span>", unsafe_allow_html=True)
                else:
                    st.caption("—")

            with l_col:
                st.markdown("**⚠️ Limitations**")
                if limitations:
                    for x in limitations:
                        st.write(f"- {x}")
                else:
                    st.caption("—")

        if apps:
            with st.expander("📚 Further Reading", expanded=False):
                for a in apps:
                    title = a.get("used_in", "Untitled")
                    year = a.get("year", "")
                    typ = a.get("type", "")
                    source = a.get("source", "")
                    url = a.get("url", "")
                    note = a.get("note", "")

                    header_bits = " · ".join([str(x) for x in [year, typ, source] if x])

                    if url:
                        st.markdown(f"**[{title}]({url})**")
                    else:
                        st.markdown(f"**{title}**")

                    if header_bits:
                        st.caption(header_bits)
                    if note:
                        st.write(note)

                    st.markdown("---")
        else:
            with st.expander("📚 Further Reading", expanded=False):
                st.caption("No further reading for this method.")


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="XAI Router for LLMs", layout="wide")

# Top image
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
  max-width: 1250px;
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
# Sidebar: hard constraints vs preferences (NEW)
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

        with st.expander("⭐ Preferences (ranking only)", expanded=True):
            st.caption("These do *not* hide tools. They only affect ordering and the 'matches/mismatches' info.")
            prefs["granularity"] = st.selectbox("Granularity (preference)", DIM_VALUES["granularity"], index=DIM_VALUES["granularity"].index(DEFAULTS["granularity"]))
            prefs["goal"] = st.selectbox("Goal / audience (preference)", DIM_VALUES["goal"], index=DIM_VALUES["goal"].index(DEFAULTS["goal"]))
            prefs["fidelity"] = st.selectbox("Fidelity (preference)", DIM_VALUES["fidelity"], index=DIM_VALUES["fidelity"].index(DEFAULTS["fidelity"]))
            prefs["format"] = st.selectbox("Explanation format (preference)", DIM_VALUES["format"], index=DIM_VALUES["format"].index(DEFAULTS["format"]))

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

        with st.expander("⭐ Preferences (optional, ranking only)", expanded=True):
            prefs["granularity"] = st.selectbox("Granularity (preference)", DIM_VALUES["granularity"], index=0)
            prefs["goal"] = st.selectbox("Goal / audience (preference)", DIM_VALUES["goal"], index=0)
            prefs["fidelity"] = st.selectbox("Fidelity (preference)", DIM_VALUES["fidelity"], index=0)
            prefs["format"] = st.selectbox("Explanation format (preference)", DIM_VALUES["format"], index=0)

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
                "plugin_id": m.get("plugin_id"),
                "score": float(sc),
                "matched": matched,
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
            dim_values_map=DIM_VALUES,
            base_scores=None,
            temperature=temperature,
            text_weight=1.0,
            soft_scale=10.0,
        )

        for item in ranked:
            m = item["method"]

            sc, matched, mismatched = score(prefs, m)

            recommended.append(
                {
                    "name": item["name"],
                    "plugin_id": m.get("plugin_id"),
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
                    },
                    "hard_used": {k: hard.get(k, "NA") for k in HARD_DIMS},
                    "prefs_used": {k: prefs.get(k, "NA") for k in PREF_DIMS},
                }
            )

# -------------------------
# Layout (3 columns)
# -------------------------
col_spacer, col_recs, col_run = st.columns([0.15, 1.2, 1.3], gap="large")

# Column 2: Recommendations
with col_recs:
    st.subheader(f"👇 {min(top_k, len(recommended))} tools match your request")

    with st.expander("🔎 Current selection (what filters vs what ranks)", expanded=False):
        st.markdown("**✅ Hard constraints (filters):**")
        st.json({k: hard.get(k, "NA") for k in HARD_DIMS}, expanded=False)
        st.markdown("**⭐ Preferences (ranking only):**")
        st.json({k: prefs.get(k, "NA") for k in PREF_DIMS}, expanded=False)

    for item in recommended[:top_k]:
        with st.container(border=True):
            st.markdown(f"### {item['name']}")
            st.write(f"**Preference score:** {item['score']:.2f}")

            if item.get("matched"):
                st.caption("✅ Matches preferences: " + ", ".join(item["matched"]))
            if item.get("mismatched"):
                st.caption("⚠️ Mismatches: " + "; ".join(item["mismatched"]))

            has_plugin = bool(item.get("plugin_id"))
            if st.button("Select", key=f"select_{item['name']}", disabled=not has_plugin):
                st.session_state["selected_item"] = item
                st.session_state["selected_method"] = item["name"]
                st.session_state["selected_plugin_id"] = item.get("plugin_id")
                st.session_state["last_outputs"] = None

# Column 3: Selected method + Run + Result
with col_run:
    st.subheader("Selected tool")

    selected_plugin_id = st.session_state.get("selected_plugin_id")
    selected_item = st.session_state.get("selected_item")

    if not selected_plugin_id:
        st.info("Select a runnable method on the left to configure and run it.")
    else:
        plugin = PLUGINS.get(selected_plugin_id)
        if plugin is None:
            st.error(f"No runnable plugin registered for: {selected_plugin_id}")
        else:
            if selected_item:
                render_selected_tool_card(selected_item)

                st.markdown("#### ✅/⚠️ Preference fit for this tool")
                m1, m2 = st.columns(2, gap="large")
                with m1:
                    st.markdown("**✅ Matches**")
                    if selected_item.get("matched"):
                        for x in selected_item["matched"]:
                            st.write(f"- {x}")
                    else:
                        st.caption("No preference matches.")

                with m2:
                    st.markdown("**⚠️ Mismatches**")
                    if selected_item.get("mismatched"):
                        for x in selected_item["mismatched"]:
                            st.write(f"- {x}")
                    else:
                        st.caption("No preference mismatches.")

                st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

            st.markdown(f"### {plugin.name}")
            inputs = render_plugin_form(plugin)

            if st.button("Run explanation", key="run_expl"):
                try:
                    outputs = plugin.run(inputs)
                    st.session_state["last_outputs"] = outputs
                except Exception as e:
                    st.error(f"Run failed: {e}")

            outputs = st.session_state.get("last_outputs")

            # ---- Captum renderers (method-specific explanation, shared plot/table) ----
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

                # ✅ Downloads (NEW)
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

                # ---- Anchor rule ----
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

                # ---- Metrics ----
                precision = outputs.get("precision", None)
                coverage = outputs.get("coverage", None)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Precision", f"{precision:.3f}" if isinstance(precision, (int, float)) else "NA")
                with c2:
                    st.metric("Coverage", f"{coverage:.3f}" if isinstance(coverage, (int, float)) else "NA")

                # ---- Examples (UPDATED KEYS) ----
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

                # ---- Params ----
                params = outputs.get("params", None)
                if params:
                    with st.expander("Parameters", expanded=False):
                        st.json(params, expanded=False)

                # ✅ Downloads
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

                # ✅ Downloads (NEW)
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

                # ✅ Downloads (NEW)
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



            elif outputs and outputs.get("plugin") in ("inseq_decoder_ig", "inseq_encdec_ig") and outputs.get("out"):
                st.subheader("Result")
                with st.expander("ℹ️ What you are seeing", expanded=True):
                    st.write(
                        "- Integrated Gradients attribution visualization produced by Inseq.\n"
                        "- The visualization is returned as HTML and embedded here.\n"
                        "- If it looks empty, try smaller max_new_tokens / fewer steps."
                    )

                st.write(f"**Model:** {outputs.get('model', 'NA')}")
                st.write(f"**Device:** {outputs.get('device', 'NA')}")
                st.write(f"**Text:** {outputs.get('text', '')}")

                # Render HTML returned by Inseq
                components.html(outputs["out"], height=850, scrolling=True)

                # Downloads (JSON always; we’ll add optional HTML download below)
                render_downloads(outputs, selected_item=selected_item)

            # Locate your "selected_plugin_id" logic in app.py
            #if selected_plugin_id == "meta_transparency_graph":
            #    if st.button("Generate Flow Graph"):
            #        # Execute the toolkit run()
            #        with st.spinner("Calculating information routes..."):
            #            params = render_plugin_form(plugin) # Using your existing form helper
            #            results = plugin.run(params)
            #            st.session_state["last_outputs"] = results

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