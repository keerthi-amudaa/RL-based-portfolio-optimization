from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

@torch.no_grad()
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    model.eval()
    return tokenizer, model

def predict_sentiment(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    probs = probs.detach().cpu().numpy()

    labels = ["Negative", "Neutral", "Positive"]
    sentiments = [labels[np.argmax(p)] for p in probs]

    return sentiments, probs
