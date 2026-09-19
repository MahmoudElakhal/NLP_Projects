from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_lstm_model.pth"
VOCAB_PATH = MODEL_DIR / "vocab.pkl"


# LABELS
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]



# TEXT SETTINGS

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

MAX_VOCAB_SIZE = 30000
MAX_LEN = 150




# MODEL SETTINGS
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.3

NUM_CLASSES = len(LABELS)


# TRAINING SETTINGS

BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

EPOCHS = 10
PATIENCE = 3

GRADIENT_CLIP = 5.0

THRESHOLD = 0.5


# REPRODUCIBILITY
SEED = 42


# DEVICE
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("MODEL_DIR exists:", MODEL_DIR.exists())
print("VOCAB_PATH:", VOCAB_PATH)
print("VOCAB exists:", VOCAB_PATH.exists())
print("MODEL exists:", MODEL_PATH.exists())