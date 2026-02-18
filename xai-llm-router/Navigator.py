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
from toolkits.captum_classifier import CaptumClassifierAttribution
from toolkits.bertviz_attention import BertVizAttention
from toolkits.logit_lens import LogitLens
from toolkits.alibi_anchors_text import AlibiAnchorsText
from toolkits.direct_logit_attribution import DirectLogitAttribution


@st.cache_resource
def get_plugins():
    plugin1 = CaptumClassifierAttribution()
    plugin2 = BertVizAttention()
    plugin3 = LogitLens()
    plugin4 = AlibiAnchorsText()
    plugin5 = DirectLogitAttribution()

    return {
        plugin1.id: plugin1,
        plugin2.id: plugin2,
        plugin3.id: plugin3,
        plugin4.id: plugin4,
        plugin5.id: plugin5,
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

    # Plugin-specific defaults (extend as you add more plugins)
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

    # ✅ FIX: stable, unique widget keys (prevents Streamlit “sticking” across reruns/tools)
    def _k(field_key: str) -> str:
        return f"{plugin_id}__{field_key}"

    for f in plugin.spec():
        # --- TEXTAREA ---
        if f.type == "textarea":
            vals[f.key] = st.text_area(
                f.label,
                value=str(default_map.get(f.key, "")),
                help=getattr(f, "help", ""),
                key=_k(f.key),
            )

        # --- TEXT ---
        elif f.type == "text":
            vals[f.key] = st.text_input(
                f.label,
                value=str(default_map.get(f.key, "")),
                help=getattr(f, "help", ""),
                key=_k(f.key),
            )

        # --- SELECT ---
        elif f.type == "select":
            options = f.options or []
            default_val = default_map.get(f.key, options[0] if options else "")

            # Choose default index safely
            index = 0
            if options and default_val in options:
                index = options.index(default_val)

            vals[f.key] = st.selectbox(
                f.label,
                options,
                index=index,
                help=getattr(f, "help", ""),
                key=_k(f.key),
            )

        # --- NUMBER ---
        elif f.type == "number":
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

                if f.key in ANCHOR_DEFAULTS:
                    default = ANCHOR_DEFAULTS[f.key]

            step = 1.0
            if f.key in ("threshold", "delta", "tau"):
                step = 0.05

            vals[f.key] = st.number_input(
                f.label,
                value=float(default),
                step=float(step),
                help=getattr(f, "help", ""),
                key=_k(f.key),
            )

        # --- CHECKBOX ---
        elif f.type == "checkbox":
            vals[f.key] = st.checkbox(
                f.label,
                value=bool(default_map.get(f.key, False)),
                help=getattr(f, "help", ""),
                key=_k(f.key),
            )

        else:
            st.warning(f"Unknown field type: {f.type} (field {f.key})")

    return vals














# -------------------------
# Selected tool card
# -------------------------
def _safe(s: str) -> str:
    return re.sub(r"[<>]", "", s or "")


def _chip(text: str):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:0.22rem 0.60rem;
            margin:0 0.35rem 0.35rem 0;
            border-radius:999px;
            border:1px solid #e6e8ec;
            background:#ffffff;
            color:#374151;
            font-size:0.85rem;
            line-height:1.2;">
            {text}
        </span>
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
                <div style="color:#6b7280; font-size:0.98rem; line-height:1.35;">
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
                        st.markdown(f"<span style='color:#065f46'>• {x}</span>", unsafe_allow_html=True)
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
st.set_page_config(page_title="XAI Navigator for LLMs", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3 { font-weight: 600; letter-spacing: -0.2px; }

/* base containers */
div[data-testid="stContainer"] > div {
  border-radius: 14px !important;
  background-color: #f7f8fa !important;
  border: 1px solid #e6e8ec !important;
  padding: 1.2rem !important;
  box-shadow: 0 1px 0 rgba(16,24,40,0.02);
}

div[data-testid="stExpander"] > details {
  border-radius: 14px !important;
  border: 1px solid #e6e8ec !important;
  background-color: #f7f8fa !important;
}

h4 { margin-top: 0.4rem; margin-bottom: 0.4rem; }
</style>
""",
    unsafe_allow_html=True,
)

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
    color: #6b7280;
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

            # ---- Your existing output rendering logic + download hooks (NEW) ----
            if outputs and outputs.get("attributions"):
                st.subheader("Result")
                with st.expander("ℹ️ How to read this explanation", expanded=True):
                    st.write(
                        "- **Model**: a text classification model that assigns one label to your sentence.\n"
                        "- **Prediction**: the label the model believes is most likely.\n"
                        "- **Label being explained**: the label whose score we attribute back to the input words.\n"
                        "- **Word importance**: tokens with larger absolute scores influence the label more.\n\n"
                        "A **positive** attribution pushes the model *toward* the explained label; "
                        "a **negative** attribution pushes it *away*."
                    )

                pred = outputs.get("predicted", {})
                tgt = outputs.get("target", {})
                params = outputs.get("params", {})

                st.write(f"**Model:** {outputs.get('model', 'NA')}")
                st.write(f"**Algorithm:** {outputs.get('algorithm', 'NA')}")
                st.write(f"**Prediction:** {pred.get('label', pred.get('idx', 'NA'))}")
                st.write(f"**Label being explained:** {tgt.get('label', tgt.get('idx', 'NA'))}")

                if outputs.get("algorithm") == "IntegratedGradients":
                    st.caption(f"Integrated Gradients steps: {params.get('n_steps', 'NA')}")

                label_map = outputs.get("label_map")
                if label_map:
                    st.markdown("**Label meanings (index → name)**")
                    st.json(label_map, expanded=False)

                st.caption("Each bar shows how strongly a token contributes to the explained label (normalized).")

                df = pd.DataFrame(outputs["attributions"])
                df_plot = df[~df["token"].isin(["[CLS]", "[SEP]", "[PAD]"])].copy()

                st.dataframe(df_plot[["token", "attr_raw", "attr_norm"]], use_container_width=True)

                fig = plt.figure()
                plt.bar(range(len(df_plot)), df_plot["attr_norm"].tolist())
                plt.xticks(range(len(df_plot)), df_plot["token"].tolist(), rotation=45, ha="right")
                plt.ylabel("Attribution (normalized)")
                plt.tight_layout()
                st.pyplot(fig)

                # ✅ Downloads (NEW)
                render_downloads(
                    outputs,
                    selected_item=selected_item,
                    figs={f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_attribution_plot.png": fig},
                )

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
