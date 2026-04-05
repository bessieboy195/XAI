from transformers import pipeline
from lime.lime_text import LimeTextExplainer
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import numpy as np
import os

ds = load_dataset("LabHC/bias_in_bios")
data = ds["train"].select(range(5)) 

texts = [item["hard_text"] for item in data]

classifier = pipeline("text-classification", model="distilbert-base-uncased")

def predict_proba(texts):
    results = classifier(texts)
    
    probs = []
    for r in results:
        if r['label'] == 'POSITIVE':
            probs.append([1 - r['score'], r['score']])
        else:
            probs.append([r['score'], 1 - r['score']])
    
    return np.array(probs)


def plot_lime_explanation(exp, title="LIME Explanation"):
    data = exp.as_list()
    filtered = [(w, s) for w, s in data if w.lower() not in ENGLISH_STOP_WORDS]

    words = [str(w) for w, _ in filtered]
    weights = [w for _, w in filtered]


    plt.figure()
    plt.barh(words, weights)
    plt.title(title)
    plt.xlabel("Feature Importance")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.show()

explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])



for i, text in enumerate(texts):

    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    print(f"Text: {text}")
    print(exp.as_list())

    filename = r"C:\Users\sebsl\OneDrive - Radboud Universiteit\second year courses\explainble AI\lime_explanation_{}.html".format(i)
    exp.save_to_file(filename)
    print("Saved:", filename)

    print("Saved to:", os.path.abspath(filename))
    plot_lime_explanation(exp, title=f"LIME Explanation {i}")