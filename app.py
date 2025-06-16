from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import csv

from model.load import load_model, encode

model, vocab, class_to_idx, idx_to_class= load_model()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    encoded = encode(text, vocab)
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

    return jsonify({'status': 'received'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
