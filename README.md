# XAI4LLMsMetaToolkit


🚀 A tool designed to help practitioners with XAI for LLMs. 

💻 Tested on Windows. 


### Developer Instructions

* Activate environment: 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_new.txt (requirements_new.txt actually contains all the libraries Martino uses) 
```

and 

```bash
pip install -r xai-inseq-requirements.txt (for inseq envioronment) 
```


* Run:
```python
streamlit run app.py
```

In Windows (power shell), to the app, Martino runs: 
```bash
.\run_all.ps1
```
And we need two envioronments to handle conflicting dependenices. Martino uses a conda envioronment with "xai-inseq-requirements.txt" and a virtual envioronment with "requirements_new.txt". 



#### Currently supported plugins 
* Attribution (Captum)
* Logit lens (mechanistic interpretability)
* Direct Logit Attribution (mechanistic interpretability)
* BertViz (visualization of attention patterns)
* Anchors (Alibi)
* Sparse Autoencoders (The coolest)
* Integrated Gradients for Text Generation (Seq2Seq) in Encoder Decoder architecture (Inseq)
* Integrated Gradients for Text Generation (Seq2Seq) in Decoder architecture (Inseq)



#### TODO 

* Update and adjust current entries methods.json (containing all the methods that can be recommended)
* Add more runnable tools
* Understand how to deploy and make the code package that can be contributed 
