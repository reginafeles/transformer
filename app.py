from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import re
import csv


class TransformerClassifier(torch.nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = torch.nn.Linear(d_model, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.permute(1, 0, 2)
        encoded = self.encoder(emb)
        cls = encoded[0]
        return self.classifier(cls)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("model/transformer_classifier.pt", map_location=device)
vocab = checkpoint['vocab']
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {i: name for name, i in class_to_idx.items()}

model = TransformerClassifier(len(vocab), len(class_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def encode(text, max_len=32):
    tokens = ['<CLS>'] + tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    encoded = encode(text)
    x = torch.tensor(encoded).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return jsonify({
        'label': idx_to_class[pred_idx],
        'confidence': round(confidence, 3),
        'probs': {idx_to_class[i]: round(p.item(), 3) for i, p in enumerate(probs[0])}
    })
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    text = data.get('text')
    predicted_label = data.get('predicted_label')
    correct = data.get('correct')

    with open('feedback.txt', 'a') as f:
        f.write(f'Text: {text}\nPredicted Label: {predicted_label}\nCorrect: {correct}\n\n')

    return jsonify({'status': 'Feedback received'})

@app.route('/correction', methods=['POST'])
def correction():
    data = request.get_json()
    text = data.get('text')
    predicted_label = data.get('predicted_label')
    true_label = data.get('true_label')

    with open('corrections.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Text', 'Predicted Label', 'True Label'])
        writer.writerow([text, predicted_label, true_label])

    return jsonify({'status': 'Correction received'})
