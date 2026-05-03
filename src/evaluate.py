"""
evaluate.py
-----------
Model evaluation: classification report, confusion matrix,
ROC curve, and training history plots.
All plot functions return matplotlib Figure objects
so they can be rendered in Streamlit via st.pyplot().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score
)


# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#E63946",   # steel red
    "secondary": "#457B9D",   # cool blue
    "accent":    "#F4A261",   # warm amber
    "bg":        "#1D1D2B",   # dark navy
    "surface":   "#2A2A3D",
    "text":      "#EAEAEA",
}


def _dark_fig(figsize=(8, 5)):
    fig = plt.figure(figsize=figsize, facecolor=PALETTE["bg"])
    return fig


def predict(model, generator):
    """Run prediction on a generator; returns (probs, preds, true_labels)."""
    generator.reset()
    probs = model.predict(generator, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    true_labels = generator.classes
    return probs, preds, true_labels


def get_metrics(true_labels, preds, probs):
    """Return a dict of scalar metrics."""
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(true_labels, preds,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy":  report["accuracy"],
        "precision": report["1"]["precision"],
        "recall":    report["1"]["recall"],
        "f1":        report["1"]["f1-score"],
        "auc":       roc_auc,
    }


def plot_training_history(history, model_name="Model"):
    """Accuracy + Loss curves side by side."""
    epochs = range(1, len(history["accuracy"]) + 1)

    fig = _dark_fig(figsize=(12, 4))
    fig.suptitle(f"{model_name} — Training History",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(1, 2, 1, facecolor=PALETTE["surface"])
    ax1.plot(epochs, history["accuracy"],    color=PALETTE["primary"],
             marker="o", lw=2, label="Train")
    ax1.plot(epochs, history["val_accuracy"], color=PALETTE["secondary"],
             marker="o", lw=2, label="Validation")
    best = np.argmax(history["val_accuracy"])
    ax1.scatter(best + 1, history["val_accuracy"][best],
                s=120, color=PALETTE["accent"], zorder=5,
                label=f"Best epoch {best+1}")
    ax1.set_title("Accuracy", color=PALETTE["text"])
    ax1.set_xlabel("Epoch", color=PALETTE["text"])
    ax1.tick_params(colors=PALETTE["text"])
    ax1.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    ax1.grid(True, alpha=0.2)
    for spine in ax1.spines.values():
        spine.set_edgecolor(PALETTE["surface"])

    ax2 = fig.add_subplot(1, 2, 2, facecolor=PALETTE["surface"])
    ax2.plot(epochs, history["loss"],     color=PALETTE["primary"],
             marker="o", lw=2, label="Train")
    ax2.plot(epochs, history["val_loss"], color=PALETTE["secondary"],
             marker="o", lw=2, label="Validation")
    ax2.set_title("Loss", color=PALETTE["text"])
    ax2.set_xlabel("Epoch", color=PALETTE["text"])
    ax2.tick_params(colors=PALETTE["text"])
    ax2.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    ax2.grid(True, alpha=0.2)
    for spine in ax2.spines.values():
        spine.set_edgecolor(PALETTE["surface"])

    fig.tight_layout()
    return fig


def plot_confusion_matrix(true_labels, preds, model_name="Model"):
    """Styled confusion matrix heatmap."""
    cm = confusion_matrix(true_labels, preds)
    fig = _dark_fig(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1, facecolor=PALETTE["surface"])

    cmap = sns.light_palette(PALETTE["primary"], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=["No Defect", "Defect"],
                yticklabels=["No Defect", "Defect"],
                linewidths=0.5, linecolor=PALETTE["bg"])

    ax.set_title(f"Confusion Matrix — {model_name}",
                 color=PALETTE["text"], fontweight="bold")
    ax.set_xlabel("Predicted", color=PALETTE["text"])
    ax.set_ylabel("Actual", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    fig.tight_layout()
    return fig


def plot_roc_comparison(models_data: dict):
    """
    Overlay ROC curves for multiple models.
    models_data: { model_name: (true_labels, probs) }
    """
    fig = _dark_fig(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1, facecolor=PALETTE["surface"])

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    for (name, (true_labels, probs)), color in zip(models_data.items(), colors):
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
    ax.set_title("ROC Curve Comparison", color=PALETTE["text"], fontweight="bold")
    ax.set_xlabel("False Positive Rate", color=PALETTE["text"])
    ax.set_ylabel("True Positive Rate", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["surface"])
    fig.tight_layout()
    return fig


def print_report(true_labels, preds, model_name="Model"):
    """Print a classification report to console."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(true_labels, preds,
                                target_names=["No Defect", "Defect"],
                                digits=4))
