import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from config import LABELS


# ============================================================
# BASIC METRICS
# ============================================================

def evaluate_model(
    y_true,
    y_probs,
    threshold=0.5
):

    y_pred = (
        y_probs >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    roc_auc = []

    pr_auc = []

    for i in range(len(LABELS)):

        roc_auc.append(
            roc_auc_score(
                y_true[:, i],
                y_probs[:, i]
            )
        )

        pr_auc.append(
            average_precision_score(
                y_true[:, i],
                y_probs[:, i]
            )
        )

    results = pd.DataFrame({
        "Label": LABELS,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    print("\nPer-label results:")
    print(results)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            zero_division=0
        )
    )

    return results


# ROC CURVES

def plot_roc_curves(
    y_true,
    y_probs
):

    plt.figure(figsize=(10, 6))

    for i, label in enumerate(LABELS):

        fpr, tpr, _ = roc_curve(
            y_true[:, i],
            y_probs[:, i]
        )

        auc = roc_auc_score(
            y_true[:, i],
            y_probs[:, i]
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{label} AUC={auc:.3f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves")

    plt.legend()

    plt.show()



# PR CURVES
def plot_pr_curves(
    y_true,
    y_probs
):

    plt.figure(figsize=(10, 6))

    for i, label in enumerate(LABELS):

        precision, recall, _ = (
            precision_recall_curve(
                y_true[:, i],
                y_probs[:, i]
            )
        )

        ap = average_precision_score(
            y_true[:, i],
            y_probs[:, i]
        )

        plt.plot(
            recall,
            precision,
            label=f"{label} AP={ap:.3f}"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall Curves"
    )

    plt.legend()

    plt.show()



# F1 PLOT

def plot_f1(results):

    plt.figure(figsize=(10, 5))

    plt.bar(
        results["Label"],
        results["F1"]
    )

    plt.xlabel("Toxicity Category")
    plt.ylabel("F1 Score")

    plt.title(
        "F1 Score per Category"
    )

    plt.xticks(rotation=30)

    plt.show()



# TRAINING CURVES
def plot_training_history(history):

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training vs Validation Loss"
    )

    plt.legend()

    plt.show()

    if "train_f1" in history:

        plt.figure(figsize=(10, 5))

        plt.plot(
            history["train_f1"],
            label="Training F1"
        )

        plt.plot(
            history["val_f1"],
            label="Validation F1"
        )

        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")

        plt.title(
            "Training vs Validation F1"
        )

        plt.legend()

        plt.show()


def find_optimal_thresholds(
    y_true,
    y_probs,
    labels,
    start=0.1,
    end=0.9,
    step=0.05
):

    thresholds = np.arange(
        start,
        end + step,
        step
    )

    best_thresholds = {}

    results = []

    for i, label in enumerate(labels):

        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds:

            predictions = (
                y_probs[:, i] >= threshold
            ).astype(int)

            score = f1_score(
                y_true[:, i],
                predictions,
                zero_division=0
            )

            if score > best_f1:

                best_f1 = score
                best_threshold = threshold

        best_thresholds[label] = (
            best_threshold
        )

        results.append({
            "Label": label,
            "Best Threshold": best_threshold,
            "Best F1": best_f1
        })

    return (
        best_thresholds,
        pd.DataFrame(results)
    )