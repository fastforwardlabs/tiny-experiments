# neural spellcheck

Streamlit app implementing some sort of neural spellcheck type thing. It's not really spellcheck.
It uses the [huggingface/transformers](https://github.com/huggingface/transformers) library to call BERT.

Install with

```bash
pip install -r requirements.txt
```

and run the app with

```bash
streamlit run app.py
```

The first time you run the app, it needs to download the BERT model weights, which are about half a GB and will take a few minutes. 
