# XAI4LLMsMetaToolkit


🚀 A tool designed to help practitioners with XAI for LLMs. 

💻 Tested on Windows. 


### Developer Instructions

* Activate environment: 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt (requirements_full.txt actually contains all the libraries Martino uses) 
```



* Run:

```python
streamlit run app.py
```

In Windows, Martino runs: 
```bash
.venv\Scripts\Activate.ps1
python -m streamlit run Navigator.py
```


#### Currently supported plugins 
* Attribution (Captum)
* Logit lens (mechanistic interpretability)
* Direct Logit Attribution (mechanistic interpretability)
* BertViz (visualization of attention patterns)
* Anchors (Alibi)


#### TODO 

* Update and adjust current entries methods.json (containing all the methods that can be recommended)
* Add more runnable tools 
