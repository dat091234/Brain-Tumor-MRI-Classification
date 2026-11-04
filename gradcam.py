"""CLI script for generating Grad-CAM visualizations across architectures."""

import argparse
import os
from pathlib import Path
import random
import torch

from src.dataset import create_datasets
from src.gradcam import compare_gradcam_across_models
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cross-architecture Grad-CAM visualizations."
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
        "--custom-cnn-ckpt",
        type=str,
        required=True,
        help="Checkpoint path for Custom CNN.",
    )
    parser.add_argument(
        "--resnet18-ckpt",
        type=str,
        required=True,
        help="Checkpoint path for ResNet18.",
    )
    parser.add_argument(
        "--densenet121-ckpt",
        type=str,
        required=True,
        help="Checkpoint path for DenseNet121.",
    )
    parser.add_argument(
        "--dinov3-ckpt",
        type=str,
        required=True,
        help="Checkpoint path for DINOv3 ViT-S/16.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific test dataset sample indices to visualize (if omitted, picks one random sample per class).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when selecting one sample per class (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gradcam",
        help="Directory to save Grad-CAM comparison figures.",
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

    _, _, test_dataset_unnorm, classes = create_datasets(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        normalize=False,
    )
    _, _, test_dataset_norm, _ = create_datasets(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        normalize=True,
    )

    ckpt_map = {
        "custom_cnn": args.custom_cnn_ckpt,
        "resnet18": args.resnet18_ckpt,
        "densenet121": args.densenet121_ckpt,
        "dinov3": args.dinov3_ckpt,
    }

    models_dict = {}
    for key, ckpt_path in ckpt_map.items():
        model = build_model(
            model_name=key,
            num_classes=len(classes),
            pretrained=False,
        ).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        models_dict[key] = model

    if args.indices:
        target_indices = [(f"idx_{idx}", idx) for idx in args.indices]
    else:
        random.seed(args.seed)
        target_indices = []
        for class_idx, class_name in enumerate(classes):
            class_sample_indices = [
                idx
                for idx, (_, label) in enumerate(test_dataset_unnorm.samples)
                if label == class_idx
            ]
            selected_idx = random.choice(class_sample_indices)
            target_indices.append((f"class_{class_name}_idx{selected_idx}", selected_idx))

    for tag, idx in target_indices:
        print(f"Generating Grad-CAM comparison for {tag} (index={idx})...")
        compare_gradcam_across_models(
            models_dict=models_dict,
            dataset_unnorm=test_dataset_unnorm,
            dataset_norm=test_dataset_norm,
            idx=idx,
            device=device,
            save_path=output_dir / f"{tag}.png",
        )


if __name__ == "__main__":
    main()
