import torch

from config import (
    MODEL_PATH,
    VOCAB_PATH,
    LABELS,
    MAX_LEN,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
    DEVICE,
    THRESHOLD
)

from LSTM_Model import ToxicLSTM

from preprocessing_text import (
    clean_text,
    tokenize,
    load_vocab,
    tokens_to_sequence,
    pad_sequence
)


class ToxicClassifier:

    def __init__(self):

        print("Loading vocabulary...")

        self.vocab = load_vocab(VOCAB_PATH)

        print(f"Vocabulary size: {len(self.vocab)}")

        print("Loading toxicity model...")

        self.model = ToxicLSTM(
            vocab_size=len(self.vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_classes=NUM_CLASSES
        )

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )
        )

        self.model.to(DEVICE)
        self.model.eval()

        print("Toxicity model loaded.\n")


    def preprocess(self, text):

        text = clean_text(text)

        tokens = tokenize(text)

        sequence = tokens_to_sequence(
            tokens,
            self.vocab
        )

        sequence = pad_sequence(
            sequence,
            MAX_LEN
        )

        return sequence


    def predict(self, text):

        if not text or not text.strip():

            return {
                label: {
                    "probability": 0.0,
                    "prediction": False
                }
                for label in LABELS
            }

        sequence = self.preprocess(text)

        x = torch.tensor(
            [sequence],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            logits = self.model(x)

            probabilities = torch.sigmoid(logits)

        probabilities = probabilities.squeeze(0).cpu().numpy()

        results = {}

        for i, label in enumerate(LABELS):

            probability = float(probabilities[i])

            prediction = probability >= THRESHOLD

            results[label] = {
                "probability": probability,
                "prediction": prediction
            }

        return results