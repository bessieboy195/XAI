from transformers import pipeline
from datasets import load_dataset
import shap

ds = load_dataset("LabHC/bias_in_bios")
data = ds["train"].select(range(5))  

texts = [item["hard_text"] for item in data]

classifier = pipeline("text-classification")

def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    texts = [str(t) for t in texts]

    outputs = classifier(texts)
    
    probs = []
    for o in outputs:
        if o["label"] == "POSITIVE":
            probs.append([1 - o["score"], o["score"]])
        else:
            probs.append([o["score"], 1 - o["score"]])
    
    return probs

explainer = shap.Explainer(predict_proba, shap.maskers.Text())

shap_values = explainer(texts)

shap.plots.text(shap_values[0])

for i in range(len(texts)):
    output_path = r"C:\Users\sebsl\OneDrive - Radboud Universiteit\second year courses\explainble AI\shap_explanation_{}.html".format(i)
    
    html = shap.plots.text(shap_values[i], display=False)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print("Saved SHAP explanation to:", output_path)