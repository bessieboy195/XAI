from transformers import pipeline

classifier = pipeline("text-classification")

texts = [
    "She is a nurse who works at a hospital.",
    "He is a software engineer at Google."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(text)
    print(result)
    print()