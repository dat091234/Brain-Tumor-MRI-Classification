"""CLI script for evaluating single or multiple Brain Tumor MRI checkpoints."""

import argparse
import os
from pathlib import Path
import pandas as pd
import torch

from src.dataset import create_dataloaders
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_experimental_comparison,
)
from src.models import (
    SUPPORTED_MODELS,
    build_model,
    count_params,
    count_unique_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Brain Tumor MRI Classification models on the test set."
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
        "--model",
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help="Evaluate a single model architecture (use with --checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path when evaluating a single model.",
    )
    # Multi-model benchmark arguments
    parser.add_argument(
        "--custom-cnn-ckpt",
        type=str,
        default=None,
        help="Best checkpoint path for Custom CNN.",
    )
    parser.add_argument(
        "--resnet18-ckpt",
        type=str,
        default=None,
        help="Best fine-tuned checkpoint path for ResNet18.",
    )
    parser.add_argument(
        "--densenet121-ckpt",
        type=str,
        default=None,
        help="Best fine-tuned checkpoint path for DenseNet121.",
    )
    parser.add_argument(
        "--dinov3-ckpt",
        type=str,
        default=None,
        help="Best fine-tuned checkpoint path for DINOv3 ViT-S/16.",
    )
    # Optional Phase 1 (Feature Extraction) checkpoints for full 6-panel chart
    parser.add_argument(
        "--resnet18-fe-ckpt",
        type=str,
        default=None,
        help="Optional Phase 1 (head-only) checkpoint for ResNet18.",
    )
    parser.add_argument(
        "--densenet121-fe-ckpt",
        type=str,
        default=None,
        help="Optional Phase 1 (head-only) checkpoint for DenseNet121.",
    )
    parser.add_argument(
        "--dinov3-fe-ckpt",
        type=str,
        default=None,
        help="Optional Phase 1 (head-only) checkpoint for DINOv3.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save evaluation reports and plots.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for loading DINOv3 architecture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader_unnorm, classes = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        normalize=False,
    )
    _, _, test_loader_norm, _ = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        normalize=True,
    )

    # Single-model evaluation mode
    if args.model and args.checkpoint:
        model = build_model(
            model_name=args.model,
            num_classes=len(classes),
            pretrained=False,
        ).to(device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        loader = test_loader_unnorm if args.model == "custom_cnn" else test_loader_norm
        metrics = evaluate_model(model, loader, device, classes)

        print(f"\nEvaluation Results - {args.model.upper()}")
        print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro F1:      {metrics['macro_f1']:.4f}")
        print(metrics["report_text"])

        plot_confusion_matrix(
            y_true=metrics["labels"],
            y_pred=metrics["preds"],
            class_names=classes,
            title=f"{args.model.upper()} - Confusion Matrix",
            save_path=output_dir / f"{args.model}_cm.png",
        )
        return

    # Multi-model benchmark mode
    benchmark_specs = [
        ("Custom CNN", "custom_cnn", args.custom_cnn_ckpt, test_loader_unnorm),
        ("ResNet18", "resnet18", args.resnet18_ckpt, test_loader_norm),
        ("DenseNet121", "densenet121", args.densenet121_ckpt, test_loader_norm),
        ("DINOv3 ViT-S/16", "dinov3", args.dinov3_ckpt, test_loader_norm),
    ]

    results = []
    built_models = {}

    for display_name, model_key, ckpt_path, loader in benchmark_specs:
        if not ckpt_path:
            continue
        print(f"Evaluating {display_name}...")
        model = build_model(
            model_name=model_key,
            num_classes=len(classes),
            pretrained=False,
        ).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        built_models[display_name] = model

        metrics = evaluate_model(model, loader, device, classes)
        num_params = count_params(model)

        results.append(
            {
                "Model": display_name,
                "Test Acc (%)": metrics["accuracy"] * 100,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Macro F1": metrics["macro_f1"],
                "Parameters (M)": num_params / 1e6,
            }
        )

        plot_confusion_matrix(
            y_true=metrics["labels"],
            y_pred=metrics["preds"],
            class_names=classes,
            title=f"{display_name} - Confusion Matrix",
            save_path=output_dir / f"{model_key}_cm.png",
        )

    if not results:
        raise ValueError(
            "Provide either --model and --checkpoint, or benchmark checkpoints "
            "(--custom-cnn-ckpt, --resnet18-ckpt, --densenet121-ckpt, --dinov3-ckpt)."
        )

    results_df = pd.DataFrame(results)
    print("\n=== Benchmark Summary ===")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(output_dir / "benchmark_results.csv", index=False)

    # Optional Feature Extraction vs Fine-Tuning comparison
    fe_specs = [
        ("ResNet18", args.resnet18_fe_ckpt, "fc"),
        ("DenseNet121", args.densenet121_fe_ckpt, "classifier"),
        ("DINOv3 ViT-S/16", args.dinov3_fe_ckpt, "classifier"),
    ]
    if all(ckpt is not None for _, ckpt, _ in fe_specs) and len(built_models) == 4:
        fe_results = []
        for display_name, fe_ckpt, head_attr in fe_specs:
            model = built_models[display_name]
            model.load_state_dict(
                torch.load(fe_ckpt, map_location=device, weights_only=True)
            )
            head_params = count_params(getattr(model, head_attr))
            metrics = evaluate_model(model, test_loader_norm, device, classes)
            fe_results.append(
                {
                    "Model": display_name,
                    "FE Test Acc (%)": metrics["accuracy"] * 100,
                    "FE Macro F1": metrics["macro_f1"],
                    "FE Params": head_params,
                    "FE Params (M)": head_params / 1e6,
                }
            )

        fe_results_df = pd.DataFrame(fe_results)
        resnet_m = built_models["ResNet18"]
        densenet_m = built_models["DenseNet121"]
        dinov3_m = built_models["DINOv3 ViT-S/16"]

        ft_trainable_params = {
            "ResNet18": count_unique_params(
                [resnet_m.layer3, resnet_m.layer4, resnet_m.fc]
            ),
            "DenseNet121": count_unique_params(
                [
                    densenet_m.features.denseblock3,
                    densenet_m.features.transition3,
                    densenet_m.features.denseblock4,
                    densenet_m.features.norm5,
                    densenet_m.classifier,
                ]
            ),
            "DINOv3 ViT-S/16": count_unique_params(
                [
                    dinov3_m.backbone.layer[-4:],
                    dinov3_m.backbone.norm,
                    dinov3_m.classifier,
                ]
            ),
        }

        ft_df = results_df[["Model", "Test Acc (%)", "Macro F1"]].rename(
            columns={"Test Acc (%)": "FT Test Acc (%)", "Macro F1": "FT Macro F1"}
        )
        comparison_df = fe_results_df.merge(ft_df, on="Model", how="inner")
        comparison_df["FT Params (M)"] = (
            comparison_df["Model"].map(ft_trainable_params) / 1e6
        )
        comparison_df["Accuracy Gain (pp)"] = (
            comparison_df["FT Test Acc (%)"] - comparison_df["FE Test Acc (%)"]
        )

        plot_experimental_comparison(
            results_df=results_df,
            comparison_df=comparison_df,
            save_path=output_dir / "experimental_results_comparison.png",
        )


if __name__ == "__main__":
    main()
