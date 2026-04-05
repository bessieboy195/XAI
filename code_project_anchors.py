import os
os.environ["HF_HOME"] = r"C:\huggingface_cache"
from transformers import pipeline
from datasets import load_dataset
from alibi.explainers import AnchorText
import spacy
import numpy as np

ds = load_dataset("LabHC/bias_in_bios")
data = ds["train"].select(range(5)) 

texts = [item["hard_text"] for item in data]

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
nlp = spacy.load("en_core_web_sm")

def predict_fn(texts):
    outputs = classifier(texts)
    probs = []
    for o in outputs:
        if o['label'] == 'POSITIVE':
            probs.append([1 - o['score'], o['score']])
        else:
            probs.append([o['score'], 1 - o['score']])
    return np.array(probs)

explainer = AnchorText(nlp=nlp, predictor=predict_fn, seed=42)

save_folder = r"C:\Users\sebsl\OneDrive - Radboud Universiteit\second year courses\explainble AI"
os.makedirs(save_folder, exist_ok=True)
for i, text in enumerate(texts):
    explanation = explainer.explain(
    text,
    threshold=0.8,     
    tau=0.3,             
    n_samples=50,     
    max_anchor_size=2    
    )
    print(f"Text ({i}): {text[:100]}...") 
    print("Anchor:", explanation.anchor)
    print("Precision:", explanation.precision)
    print("Coverage:", explanation.coverage)
    print()
    output_path = os.path.join(save_folder, f"anchor_explanation_{i}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Text:\n{text}\n\n")
        f.write(f"Anchor:\n{explanation.anchor}\n\n")
        f.write(f"Precision: {explanation.precision}\n")
        f.write(f"Coverage: {explanation.coverage}\n")
    print("Saved to:", os.path.abspath(output_path))