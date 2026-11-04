"""Evaluation metrics, confusion matrix plotting, and 6-panel benchmark visualization."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Evaluate a model on a DataLoader and return comprehensive metrics."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=list(class_names),
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": report_dict["macro avg"]["precision"],
        "recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "report_text": report_text,
        "report_dict": report_dict,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_experimental_comparison(
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Generate the 6-panel experimental comparison chart across all models."""
    warnings.filterwarnings("ignore", category=UserWarning)

    plot_df = results_df.copy()
    plot_df["Precision (%)"] = plot_df["Precision"] * 100
    plot_df["Recall (%)"] = plot_df["Recall"] * 100
    plot_df["Macro F1 (%)"] = plot_df["Macro F1"] * 100

    baseline_acc = plot_df.loc[
        plot_df["Model"] == "Custom CNN", "Test Acc (%)"
    ].iloc[0]
    plot_df["Improvement (%)"] = (
        (plot_df["Test Acc (%)"] - baseline_acc) / baseline_acc
    ) * 100

    models_names = plot_df["Model"]
    x = np.arange(len(models_names))

    test_acc = plot_df["Test Acc (%)"]
    macro_f1 = plot_df["Macro F1 (%)"]
    precision = plot_df["Precision (%)"]
    recall = plot_df["Recall (%)"]
    params = plot_df["Parameters (M)"]

    scratch_acc = plot_df.loc[
        plot_df["Model"] == "Custom CNN", "Test Acc (%)"
    ].mean()
    pretrained_cnn_acc = plot_df.loc[
        plot_df["Model"].isin(["ResNet18", "DenseNet121"]), "Test Acc (%)"
    ].mean()
    transformer_acc = plot_df.loc[
        plot_df["Model"] == "DINOv3 ViT-S/16", "Test Acc (%)"
    ].mean()

    group_names = [
        "CNN \n(From Scratch)",
        "CNN \n(Transfer Learning)",
        "ViT \n(Transfer Learning)",
    ]
    group_values = [scratch_acc, pretrained_cnn_acc, transformer_acc]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Experimental Results: Brain Tumor MRI Model Comparison",
        fontsize=18,
        fontweight="bold",
    )

    width = 0.35

    # 1) Test Accuracy vs Macro F1
    bars1 = axes[0, 0].bar(x - width / 2, test_acc, width, label="Test Accuracy")
    bars2 = axes[0, 0].bar(x + width / 2, macro_f1, width, label="Macro F1")
    axes[0, 0].set_title("Test Accuracy vs Macro F1")
    axes[0, 0].set_ylabel("Score (%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models_names, rotation=15)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].legend()
    axes[0, 0].bar_label(bars1, fmt="%.2f", padding=3)
    axes[0, 0].bar_label(bars2, fmt="%.2f", padding=3)

    # 2) Model Size Comparison
    bars = axes[0, 1].bar(models_names, params)
    axes[0, 1].set_title("Model Size Comparison")
    axes[0, 1].set_ylabel("Model Size (M parameters)")
    axes[0, 1].set_xticklabels(models_names, rotation=15)
    axes[0, 1].bar_label(bars, fmt="%.2f", padding=3)

    # 3) Precision vs Recall
    bars1 = axes[0, 2].bar(x - width / 2, precision, width, label="Precision")
    bars2 = axes[0, 2].bar(x + width / 2, recall, width, label="Recall")
    axes[0, 2].set_title("Precision vs Recall")
    axes[0, 2].set_ylabel("Score (%)")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models_names, rotation=15)
    axes[0, 2].set_ylim(0, 105)
    axes[0, 2].legend()
    axes[0, 2].bar_label(bars1, fmt="%.2f", padding=3)
    axes[0, 2].bar_label(bars2, fmt="%.2f", padding=3)

    # 4) Relative Improvement over Custom CNN
    improvement_df = plot_df[plot_df["Model"] != "Custom CNN"].reset_index(drop=True)
    x_improvement = np.arange(len(improvement_df))
    bars = axes[1, 0].bar(x_improvement, improvement_df["Improvement (%)"])
    axes[1, 0].set_title("Relative Performance Improvement")
    axes[1, 0].set_ylabel("Improvement over Custom CNN (%)")
    axes[1, 0].set_xticks(x_improvement)
    axes[1, 0].set_xticklabels(improvement_df["Model"], rotation=15)
    axes[1, 0].bar_label(bars, fmt="%.2f%%", padding=3)

    # 5) Comparison by Model Family
    bars = axes[1, 1].bar(group_names, group_values)
    axes[1, 1].set_title("Comparison by Model Family")
    axes[1, 1].set_ylabel("Average Test Accuracy (%)")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].bar_label(bars, fmt="%.2f", padding=3)

    # 6) Feature Extraction vs Fine-Tuning
    ax = axes[1, 2]
    x_comparison = np.arange(len(comparison_df))
    bars_fe = ax.bar(
        x_comparison - width / 2,
        comparison_df["FE Test Acc (%)"],
        width,
        label="Feature Extraction",
    )
    bars_ft = ax.bar(
        x_comparison + width / 2,
        comparison_df["FT Test Acc (%)"],
        width,
        label="Partial Fine-Tuning",
    )

    fe_labels = [
        f"{acc:.2f}%\n({p:.4f}M)"
        for acc, p in zip(
            comparison_df["FE Test Acc (%)"], comparison_df["FE Params (M)"]
        )
    ]
    ft_labels = [
        f"{acc:.2f}%\n({p:.4f}M)"
        for acc, p in zip(
            comparison_df["FT Test Acc (%)"], comparison_df["FT Params (M)"]
        )
    ]

    ax.bar_label(bars_fe, labels=fe_labels, padding=3, fontsize=8)
    ax.bar_label(bars_ft, labels=ft_labels, padding=3, fontsize=8)
    ax.set_title("Feature Extraction vs Fine-Tuning")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x_comparison)
    ax.set_xticklabels(comparison_df["Model"], rotation=15)
    ax.set_ylim(0, 120)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.5,
        -0.25,
        "Labels: Accuracy (%) and Trainable Parameters (M)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    for sub_ax in axes.flat:
        sub_ax.grid(axis="y", alpha=0.3)
        sub_ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved experimental comparison figure -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
