from transformers import pipeline
from lime.lime_text import LimeTextExplainer

# Load model
classifier = pipeline("text-classification")

# LIME needs a function that returns probabilities
import numpy as np

def predict_proba(texts):
    results = classifier(texts)
    
    probs = []
    for r in results:
        if r['label'] == 'POSITIVE':
            probs.append([1 - r['score'], r['score']])
        else:
            probs.append([r['score'], 1 - r['score']])
    
    return np.array(probs)   

explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])

texts = [
    "She is a nurse with many years of experience.",
    "He is a nurse with many years of experience.",
    "She is a software engineer at a big company.",
    "He is a software engineer at a big company."
]

for text in texts:
    exp = explainer.explain_instance(text, predict_proba, num_features=6)
    print(f"Text: {text}")
    print(exp.as_list())
    exp.save_to_file(f"lime_explanation_{text[:10]}.html")  

