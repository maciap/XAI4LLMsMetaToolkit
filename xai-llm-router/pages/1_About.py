import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About LLM Explainability Navigator 🧭")

st.markdown("""
### Why this app exists

The landscape of explainability for large language models is fragmented and fast-moving.  
New methods, taxonomies, and surveys appear constantly — but navigating them is difficult, especially for newcomers.

This tool acts as a **compass**.

It helps you identify which explainability methods are feasible for your setting and, when supported, lets you run them directly.

---

### Who this is for

- Non-expert LLM users exploring interpretability tools  
- ML engineers working with real systems  
- Researchers and practitioners studying model behavior  

---

### What this tool helps you do

- Translate your needs into structured constraints  
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
