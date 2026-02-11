# XAI4LLMsMetaToolkit


🚀 A tool designed to help practitioners with XAI for LLMs. 

💻 Tested on Windows. 


### Developer Instructions

* Activate environment: 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

* Run:

```python
streamlit run app.py
```

#### Currently supported plugins 
* Attribution (Captum)
* Logit lens (mechanistic interpretability)
* BertViz (visualization of attention patterns)
* Anchors (Alibi) - to revise 


#### TODO 

* Update and adjust current entries methods.json (containing all the methods that can be recommended)
* Add more runnable tools 
