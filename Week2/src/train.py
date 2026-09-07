import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from dataset_torch import ToxicCommentDataset
from LSTM_Model import ToxicLSTM

from config import *

# SPLIT DATA

def split_data(X, y):
    # DEV_TRAIN_SIZE = 10000
    # DEV_VAL_SIZE = 2000
    # DEV_TEST_SIZE = 2000
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.20,random_state=SEED)

    X_val, X_test, y_val, y_test = train_test_split( X_temp,y_temp,test_size=0.50,random_state=SEED)

    # X_train = X_train[:DEV_TRAIN_SIZE]
    # y_train = y_train[:DEV_TRAIN_SIZE]

    # X_val = X_val[:DEV_VAL_SIZE]
    # y_val = y_val[:DEV_VAL_SIZE]

    # X_test = X_test[:DEV_TEST_SIZE]
    # y_test = y_test[:DEV_TEST_SIZE]
    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# CLASS WEIGHTS


# def calculate_pos_weights(y):

    # positive = y.sum(axis=0)

    # negative = len(y) - positive

    # weights = negative / positive

    # return torch.tensor( weights,dtype=torch.float32 ).to(DEVICE)

def calculate_pos_weights(y_train, cap=20.0):

    # Make sure y_train is numeric
    y_train = np.asarray(
        y_train,
        dtype=np.float32
    )

    # Make sure cap is numeric
    cap = float(cap)

    # Number of positive samples for each label
    positive = y_train.sum(axis=0)

    # Number of negative samples for each label
    negative = y_train.shape[0] - positive

    # Avoid division by zero
    positive = np.maximum(
        positive,
        1.0
    )

    # Calculate positive class weights
    weights = negative / positive

    # Cap extremely large weights
    weights = np.minimum(
        weights,
        cap
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )

# TRAIN ONE EPOCH

def train_one_epoch(model,loader,criterion,optimizer):

    model.train()

    total_loss = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion( logits,y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_( model.parameters(),GRADIENT_CLIP)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# VALIDATION

def validate( model,loader,criterion):

    model.eval()

    total_loss = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)

            loss = criterion( logits,y_batch)

            total_loss += loss.item()

            probabilities = torch.sigmoid(logits )

            all_probs.append(probabilities.cpu().numpy())

            all_targets.append( y_batch.cpu().numpy())

    return (
        total_loss / len(loader),
        np.vstack(all_targets),
        np.vstack(all_probs)
    )


# TRAIN MODEL

def train_model(X_train,y_train, X_val,y_val):

    train_dataset = ToxicCommentDataset( X_train, y_train)

    val_dataset = ToxicCommentDataset( X_val,y_val)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True
    )

    val_loader = DataLoader( val_dataset,batch_size=BATCH_SIZE,shuffle=False)

    model = ToxicLSTM(vocab_size=int( X_train.max() + 1),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    pos_weight = calculate_pos_weights(y_train ).to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight )

    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY )

    history = {"train_loss": [],"val_loss": [],"train_f1": [],"val_f1": []}

    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(model,train_loader,criterion, optimizer)

        val_loss, val_true, val_probs = (validate( model,val_loader,criterion))

        val_pred = (val_probs >= 0.5).astype(int)

        from sklearn.metrics import f1_score

        val_f1 = f1_score(
            val_true,
            val_pred,
            average="macro",
            zero_division=0
        )

        history["train_loss"].append(train_loss )

        history["val_loss"].append(val_loss)

        history["val_f1"].append( val_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print("Early stopping.")

                break

    return model, history