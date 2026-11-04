"""CLI script for training Brain Tumor MRI classification models.

Supports training Custom CNN from scratch or running 3-phase progressive
fine-tuning for ResNet-18, DenseNet-121, and DINOv3 ViT-S/16.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn

from src.dataset import create_dataloaders
from src.engine import train_phase
from src.evaluate import evaluate_model, plot_confusion_matrix
from src.models import (
    SUPPORTED_MODELS,
    build_model,
    configure_fine_tuning_phase,
    count_params,
    count_trainable_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Brain Tumor MRI Classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Architecture to train.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to the Training ImageFolder directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to the Testing ImageFolder directory.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Training phase to run ('1', '2', '3', or 'all' for full progressive fine-tuning).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to load before starting training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints and evaluation plots.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max epochs per phase (defaults to 100 for custom_cnn, 30 for pretrained models).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for downloading gated DINOv3 weights (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--low-lr-phase3",
        action="store_true",
        help="Use the lower learning rate schedule for DINOv3 Phase 3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normalize = args.model != "custom_cnn"
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        normalize=normalize,
        num_workers=args.num_workers,
    )
    print(f"Classes ({len(classes)}): {classes}")

    model = build_model(
        model_name=args.model,
        num_classes=len(classes),
        pretrained=True,
    ).to(device)

    if args.resume_checkpoint:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        model.load_state_dict(
            torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    if args.model == "custom_cnn":
        phases = [1]
        default_epochs = 100
    else:
        phases = [1, 2, 3] if args.phase == "all" else [int(args.phase)]
        default_epochs = 30

    max_epochs = args.epochs or default_epochs
    best_val_acc = 0.0
    last_ckpt_path = None

    for phase in phases:
        print(f"\n{'=' * 60}")
        print(f"Starting {args.model.upper()} - Phase {phase}")
        print(f"{'=' * 60}")

        if last_ckpt_path is not None and Path(last_ckpt_path).is_file():
            print(f"Reloading best checkpoint from previous phase: {last_ckpt_path}")
            model.load_state_dict(
                torch.load(last_ckpt_path, map_location=device, weights_only=True)
            )

        optimizer = configure_fine_tuning_phase(
            model=model,
            model_name=args.model,
            phase=phase,
            low_lr=(args.low_lr_phase3 and phase == 3),
        )

        total_params = count_params(model)
        trainable_params = count_trainable_params(model)
        print(
            f"Total parameters: {total_params:,} | "
            f"Trainable parameters: {trainable_params:,}"
        )

        ckpt_path = output_dir / f"best_{args.model}_phase{phase}.pth"
        best_val_acc, _ = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            save_path=ckpt_path,
            max_epochs=max_epochs,
            patience=args.patience,
            initial_best_val_acc=best_val_acc,
        )

        if ckpt_path.is_file():
            last_ckpt_path = ckpt_path

    # Final evaluation on Test Set using the best checkpoint
    if last_ckpt_path is not None and Path(last_ckpt_path).is_file():
        print(f"\nLoading best checkpoint for final test evaluation: {last_ckpt_path}")
        model.load_state_dict(
            torch.load(last_ckpt_path, map_location=device, weights_only=True)
        )

    test_metrics = evaluate_model(model, test_loader, device, classes)
    print(f"\nFinal Test Evaluation - {args.model.upper()}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(test_metrics["report_text"])

    cm_path = output_dir / f"{args.model}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=test_metrics["labels"],
        y_pred=test_metrics["preds"],
        class_names=classes,
        title=f"{args.model.upper()} - Confusion Matrix",
        save_path=cm_path,
    )


if __name__ == "__main__":
    main()
