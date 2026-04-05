from transformers import pipeline
from datasets import load_dataset

ds = load_dataset("LabHC/bias_in_bios")

data = ds["train"].select(range(10))

classifier = pipeline("text-classification")

texts = [item["hard_text"] for item in data]

results = classifier(texts)

for text, result in zip(texts, results):
    print("TEXT:", text[:100])  
    print("PREDICTION:", result)
    print()