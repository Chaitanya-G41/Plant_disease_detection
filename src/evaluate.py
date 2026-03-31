"""
Evaluation utilities — confusion matrix, per-class F1, guava-specific metrics.
All outputs are saved as files for inclusion in the research paper.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        with autocast():
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_model(
    model,
    val_loader,
    class_names: list,
    save_dir: str,
    stage: int = 1,
    device: str = 'cuda',
    guava_only: bool = False
):
    """
    Full evaluation: accuracy, per-class F1, confusion matrix plot.
    Saves all outputs to save_dir for the paper.
    """
    os.makedirs(save_dir, exist_ok=True)
    preds, labels, probs = get_predictions(model, val_loader, device)

    # Filter to guava classes only if requested (Stage 2 evaluation)
    if guava_only:
        guava_idx = [i for i, c in enumerate(class_names) if c.startswith('Guava_')]
        mask = np.isin(labels, guava_idx)
        preds  = preds[mask]
        labels = labels[mask]
        # Remap to 0-based indices for guava only
        label_map = {old: new for new, old in enumerate(guava_idx)}
        preds  = np.array([label_map[p] if p in label_map else -1 for p in preds])
        labels = np.array([label_map[l] for l in labels])
        class_names = [class_names[i] for i in guava_idx]
        # Remove -1 predictions (model predicted non-guava class for guava image)
        valid = preds != -1
        preds, labels = preds[valid], labels[valid]

    # Overall accuracy
    accuracy = 100. * (preds == labels).sum() / len(labels)

    # Per-class metrics
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    macro_f1     = f1_score(labels, preds, average='macro',    zero_division=0)
    weighted_f1  = f1_score(labels, preds, average='weighted', zero_division=0)
    macro_prec   = precision_score(labels, preds, average='macro',    zero_division=0)
    macro_recall = recall_score(labels, preds,    average='macro',    zero_division=0)

    print(f"\n{'='*60}")
    print(f"Stage {stage} Evaluation {'(Guava only)' if guava_only else '(All classes)'}")
    print(f"{'='*60}")
    print(f"Overall Accuracy  : {accuracy:.2f}%")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print(f"Macro Precision   : {macro_prec:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"\nPer-class F1:")
    for cls in class_names:
        if cls in report:
            print(f"  {cls:<45} F1: {report[cls]['f1-score']:.4f}")

    # Save metrics to JSON
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_recall,
        'per_class': report
    }
    tag = f'stage{stage}_{"guava" if guava_only else "all"}'
    with open(os.path.join(save_dir, f'{tag}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    plot_confusion_matrix(
        labels, preds, class_names,
        save_path=os.path.join(save_dir, f'{tag}_confusion_matrix.png'),
        title=f'Stage {stage} Confusion Matrix'
    )

    return metrics


def plot_confusion_matrix(labels, preds, class_names, save_path, title=''):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize per row

    fig_size = max(10, len(class_names) // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_training_history(history_path: str, save_dir: str, stage: int):
    """Plot loss and accuracy curves from saved history JSON."""
    with open(history_path) as f:
        history = json.load(f)

    epochs     = [h['epoch']      for h in history]
    train_acc  = [h['train_acc']  for h in history]
    val_acc    = [h['val_acc']    for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss']   for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_acc,  label='Train', marker='o', markersize=3)
    ax1.plot(epochs, val_acc,    label='Val',   marker='o', markersize=3)
    ax1.set_title(f'Stage {stage} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_loss, label='Train', marker='o', markersize=3)
    ax2.plot(epochs, val_loss,   label='Val',   marker='o', markersize=3)
    ax2.set_title(f'Stage {stage} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, f'stage{stage}_training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {out}")