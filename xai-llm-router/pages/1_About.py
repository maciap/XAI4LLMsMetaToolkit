import streamlit as st




st.set_page_config(page_title="About", layout="wide")

st.title("About LLM Explainability Navigator 🧭")

col_left, col_right = st.columns([2, 1])  # left wider for text

with col_left:
    st.markdown("""
    <div style='padding: 1.5rem 1rem; font-size: 1.2rem; font-style: italic; color: #555;'>
    “Midway upon the journey of our life,<br>
    I found myself within a forest dark,<br>
    For the straightforward pathway had been lost.”
    </div>

    <div style='font-size: 0.9rem; color: #888; margin-top: -0.5rem;'>
    — Dante Alighieri, <em>The Divine Comedy</em>
    </div>
    """, unsafe_allow_html=True)

with col_left:
    st.image("images/logo_app.png", width=220)



st.markdown("""
### Why this app exists
            
The landscape of explainability (XAI) for large language models is fragmented and fast-moving.  
New methods, taxonomies, and surveys appear constantly, but navigating them is difficult, especially for newcomers.

Like Dante at the beginning of the *Divine Comedy*, one can easily feel lost in a dark forest —  
not for lack of paths, but for lack of orientation.  
In that journey, Virgil does not create the road; he acts as a guide through it.
            
This tool acts as a **guide** in the forest of explainability for large language models.

It helps you identify which explainability methods are feasible for your setting and, when supported, lets you run them directly.
---

### Who this is primarily for

- Non-expert LLM users exploring interpretability tools  

---

### What this tool helps you do

- Discover relevant explainability toolkits  
- Compare methods across scope, access, granularity, and goals  
- Run selected tools when plugins are available  

---

### Design Philosophy

Instead of proposing yet another taxonomy,  
this app connects:

- **User intent**
- **Method metadata**
- **Runnable toolkits**

The goal is clarity, not complexity.
""")
