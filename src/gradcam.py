"""Grad-CAM visual explanations for CNNs and DINOv3 Vision Transformer."""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder

from src.dataset import IMAGENET_MEAN, IMAGENET_STD


def reshape_transform_dino(
    tensor: torch.Tensor,
    height: int,
    width: int,
    num_register_tokens: int,
) -> torch.Tensor:
    """Reshape DINOv3 sequence tokens [B, N, C] into 2D spatial feature maps [B, C, H, W]."""
    num_prefix_tokens = 1 + num_register_tokens
    patch_tokens = tensor[:, num_prefix_tokens:, :]

    assert patch_tokens.shape[1] == height * width, (
        f"Expected {height * width} patch tokens, got {patch_tokens.shape[1]}"
    )

    patch_tokens = patch_tokens.reshape(
        tensor.shape[0], height, width, tensor.shape[-1]
    )
    return patch_tokens.permute(0, 3, 1, 2).contiguous()


def get_target_layer_and_reshape(
    model: nn.Module,
    model_name: str,
    image_size: int = 224,
) -> Tuple[nn.Module, Optional[Callable]]:
    """Return the Grad-CAM target layer and optional reshape_transform for a model."""
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        return model.features[7], None

    if model_name == "resnet18":
        return model.layer4[-1], None

    if model_name == "densenet121":
        return model.features.denseblock4, None

    if model_name == "dinov3":
        patch_size = model.backbone.config.patch_size
        num_register_tokens = model.backbone.config.num_register_tokens
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size

        height = image_size // patch_h
        width = image_size // patch_w
        reshape_fn = partial(
            reshape_transform_dino,
            height=height,
            width=width,
            num_register_tokens=num_register_tokens,
        )
        return model.backbone.layer[-1].norm1, reshape_fn

    raise ValueError(f"Unsupported model_name '{model_name}' for Grad-CAM.")


def get_gradcam_result(
    model: nn.Module,
    target_layer: nn.Module,
    dataset: ImageFolder,
    idx: int,
    device: torch.device,
    reshape_transform: Optional[Callable] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute prediction and Grad-CAM overlay for a single dataset sample."""
    model = model.to(device).eval()

    image, true_label = dataset[idx]
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_label = probs.argmax(dim=1).item()
        confidence = probs[0, pred_label].item()

    with torch.enable_grad():
        with GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        ) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(pred_label)],
            )[0]

    rgb_image = image.permute(1, 2, 0).cpu().numpy()
    if mean is not None and std is not None:
        rgb_image = rgb_image * np.array(std) + np.array(mean)

    rgb_image = np.clip(rgb_image, 0, 1).astype(np.float32)
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    return {
        "image": rgb_image,
        "heatmap": grayscale_cam,
        "overlay": overlay,
        "true": dataset.classes[true_label],
        "pred": dataset.classes[pred_label],
        "pred_idx": pred_label,
        "confidence": confidence,
    }


def compare_gradcam_across_models(
    models_dict: Dict[str, nn.Module],
    dataset_unnorm: ImageFolder,
    dataset_norm: ImageFolder,
    idx: int,
    device: torch.device,
    image_size: int = 224,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a 5-column side-by-side Grad-CAM comparison across all architectures."""
    display_names = {
        "custom_cnn": "Custom CNN",
        "resnet18": "ResNet18",
        "densenet121": "DenseNet121",
        "dinov3": "DINOv3",
    }

    results = []
    names = []
    original_img = None
    true_class = None

    for key in ["custom_cnn", "resnet18", "densenet121", "dinov3"]:
        if key not in models_dict:
            continue
        model = models_dict[key]
        target_layer, reshape_fn = get_target_layer_and_reshape(
            model, key, image_size=image_size
        )
        is_unnorm = key == "custom_cnn"
        res = get_gradcam_result(
            model=model,
            target_layer=target_layer,
            dataset=dataset_unnorm if is_unnorm else dataset_norm,
            idx=idx,
            device=device,
            reshape_transform=reshape_fn,
            mean=None if is_unnorm else IMAGENET_MEAN,
            std=None if is_unnorm else IMAGENET_STD,
        )
        if original_img is None:
            original_img = res["image"]
            true_class = res["true"]
        results.append(res)
        names.append(display_names.get(key, key))

    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4))
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original MRI\nTrue: {true_class}")

    for i, (name, result) in enumerate(zip(names, results), start=1):
        axes[i].imshow(result["overlay"])
        axes[i].set_title(
            f"{name}\nPred: {result['pred']} ({result['confidence']:.2%})"
        )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Grad-CAM comparison -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
