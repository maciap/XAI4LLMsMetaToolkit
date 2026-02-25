
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


def _acc_rank_order(pref: str) -> List[str]:
    """
    Preference -> ordering from best to worst.
    """
    pref = (pref or "NA").strip()
    if pref == "experts":
        return ["experts", "mid experts", "non experts"]
    if pref == "non experts":
        return ["non experts", "mid experts", "experts"]
    if pref == "mid experts":
        return ["mid experts", "non experts", "experts"]
    return []  # NA => no ranking

def score(prefs: Dict[str, str], m: Dict[str, Any]) -> Tuple[int, List[str], List[str]]:
    """
    Accessibility-only scoring:
    - higher score = higher rank
    - if prefs["accessibility"] == "NA" -> score 0 for everything (no ranking)
    """
    pref = prefs.get("accessibility", "NA")
    order = _acc_rank_order(pref)

    tool_acc = (m.get("accessibility", "NA") or "NA").strip()

    if not order or pref == "NA":
        return 0, [], []  # no ranking requested

    # If tool has unknown accessibility, put it last
    if tool_acc not in ("experts", "mid experts", "non experts"):
        return -1, [], [f"🎓 accessibility unknown (tool has {tool_acc})"]

    # Map: best -> 2, middle -> 1, worst -> 0
    # (or any scale you prefer)
    if tool_acc == order[0]:
        return 2, [f"🎓 accessibility match ({tool_acc})"], []
    if tool_acc == order[1]:
        return 1, [f"🎓 accessibility mid ({tool_acc})"], []
    return 0, [], [f"🎓 accessibility lower priority (tool has {tool_acc})"]



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
            ("Accessibility", meta.get("accessibility")),  
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
