import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch import nn
import os

# CustomDistilBERTClassifier
class CustomDistilBERTClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super(CustomDistilBERTClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        logits = self.classifier(hidden_state)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}

model_path = "./models/custom_model/finetuned_distilbert_3class/"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

model = CustomDistilBERTClassifier(model_path, num_labels=3)
model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs['logits']
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


test_sentences = [
    "The US economy grew by 2.5% in the last quarter.",
    "China's exports dropped significantly in March.",
    "The Federal Reserve maintained interest rates unchanged.",
    "Tariff tensions between China and the US continue to escalate.",
    "Chinese tech companies reported better-than-expected earnings.",
    "Investors are cautious ahead of the US inflation data release.",
    "The Biden administration plans to increase tariffs on Chinese goods.",
    "The yuan strengthened slightly against the dollar today.",
    "Talks between US and China officials ended without major agreements.",
    "US-China trade relations remain uncertain."
]

expected_labels = [2, 0, 1, 0, 2, 1, 0, 2, 0, 1]

correct = 0
for text, expected in zip(test_sentences, expected_labels):
    pred = predict(text)
    print(f"Text: {text}\nPredicted: {pred}, Expected: {expected}\n")
    if pred == expected:
        correct += 1

print(f"Accuracy: {correct}/10 = {correct*10}%")

