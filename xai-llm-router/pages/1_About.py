import streamlit as st

st.set_page_config(page_title="About", layout="wide")

# --- Minimal CSS for nicer typography + cards ---
st.markdown(
    """
    <style>
      .about-quote {
        padding: 1.25rem 1rem;
        font-size: 1.1rem;
        font-style: italic;
        color: #555;
        border-left: 4px solid rgba(49, 130, 206, 0.35);
        background: rgba(0,0,0,0.02);
        border-radius: 10px;
      }
      .about-cite {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.25rem;
      }
      .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        background: rgba(49, 130, 206, 0.12);
        margin-right: 0.35rem;
        margin-top: 0.35rem;
      }
      .card {
        padding: 1rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.5);
      }
      .small-muted { color: #666; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("About LLM Explainability Navigator 🧭")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown(
        """
        <div class="about-quote">
        “Midway upon the journey of our life,<br>
        I found myself within a forest dark,<br>
        For the straightforward pathway had been lost.”
        </div>
        <div class="about-cite">
        — Dante Alighieri, <em>The Divine Comedy</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
The landscape of explainability (XAI) for large language models is **fragmented** and **fast-moving**.
New methods, studies, and taxonomies appear rapidly, but *navigating* them is difficult, especially for newcomers.

Like Dante at the beginning of the *Divine Comedy*, one can feel lost in a dark forest not for lack of paths,
but for lack of orientation. Virgil does not create the road; he acts as a guide through it.

This app acts as a **guide** in the forest of LLM explainability methods.
        """
    )

    st.markdown("")

    # Quick “what you get” pills
    st.markdown(
        """
        <span class="pill">Method discovery</span>
        <span class="pill">Constraint-based filtering</span>
        <span class="pill">Side-by-side comparison</span>
        <span class="pill">Runnable demos (when available)</span>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.image("images/logo_app.png", use_container_width=True)
    st.markdown(
        "<div class='small-muted'>A structured navigator for selecting and comparing LLM explainability tools.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- Main sections as tabs for scanability ---
tab1, tab2, tab3, tab4 = st.tabs(["What it does", "How it works", "Who it's for", "Notes & limitations"])

with tab1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### ✅ What you can do")
        st.markdown(
            """
- **Identify feasible tools** for your setting (model type, access level, compute, etc.).
- **Understand key characteristics** (assumptions, strengths, limitations, outputs).
- **Compare alternatives** side-by-side to make trade-offs explicit.
- **Explore tools interactively** when runnable components are available (custom inputs/parameters).
            """
        )

    with c2:
        st.markdown("#### 🎯 Why it matters")
        st.markdown(
            """
Explainability method selection is often *ad hoc* (defaulting to familiar tools).
This navigator helps make the choice more **systematic**, **transparent**, and **defensible*, both in research and practice.
            """
        )
        st.info(
            "Think of it as explaining *how to choose an explanation method*, not only how to explain a model.",
            icon="🧭",
        )

with tab2:
    st.markdown("#### 🧩 The core idea")
    st.markdown(
        """
This app connects **user intent** → **tool constraints** → **method recommendations and comparisons**.

A typical workflow:
1. Specify your setting and goals (what you need, what you can access).
2. Filter out methods that are **not applicable** given your constraints.
3. Rank or shortlist methods based on preferences (e.g., faithfulness, usability, output type).
4. Compare finalists, then (optionally) run a method if supported.
        """
    )

    st.markdown("#### 🧠 Design philosophy")
    st.markdown(
        """
- **Clarity over complexity:** keep the decision process understandable.
- **Separation of constraints vs preferences:** feasibility first, ranking second.
        """
    )

with tab3:
    st.markdown("#### 👥 Who this is for")
    st.markdown(
        """
This app aims to support **both**:
- **Users with limited expertise** in LLMs/XAI, by lowering the entry barrier and providing guided navigation.
- **Experienced practitioners**, by offering a structured way to search, justify, and compare tools in their toolbox.
        """
    )

    st.markdown("#### 🎓 Educational & training use")
    st.markdown(
        """
It can also be useful for:
- **Teaching** XAI for LLMs (conceptual map + trade-offs).
- **Workshops / onboarding** in labs or companies.
- **Methodological discussions** in interdisciplinary teams (ML + domain experts).
 """
    )

with tab4:
    st.markdown("#### ⚠️ Notes & limitations")
    st.markdown(
        """
- The field evolves quickly; the app needs to be regularly updated.
- “Best method” depends on goals and constraints; the app supports decision-making but does not replace domain judgment.
  """
    )

    with st.expander("Suggested citation / credit", expanded=False):
        st.markdown(
            """
If you’re using the Navigator in a paper, consider citing it.
            """
        )

    with st.expander("Contact / links", expanded=False):
        st.markdown(
            """
- GitHub repository: *https://github.com/maciap/XAI4LLMsMetaToolkit*
            """
        )

st.markdown("---")
st.caption("LLM Explainability Navigator 🧭 — About page")