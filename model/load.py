import torch
import re


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


def load_model(model_path="model/transformer_classifier.pt"):
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    model = TransformerClassifier(len(vocab), len(class_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab, class_to_idx, idx_to_class


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def encode(text, vocab, max_len=32):
    tokens = ['<CLS>'] + tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids
