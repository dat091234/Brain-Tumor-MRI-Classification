"""Model architectures and progressive fine-tuning phase configurations."""

import os
from typing import Iterable, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoConfig, AutoModel

DINOV3_DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
SUPPORTED_MODELS = ("custom_cnn", "resnet18", "densenet121", "dinov3")


class CustomCNN(nn.Module):
    """Lightweight 3-block convolutional neural network trained from scratch."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Convolution Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolution Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolution Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class DINOv3Classifier(nn.Module):
    """Vision Transformer classifier built on top of DINOv3 ViT-S/16."""

    def __init__(self, backbone: nn.Module, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval mode when all backbone weights are frozen
        if mode and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self


def build_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dinov3_model_id: str = DINOV3_DEFAULT_MODEL_ID,
) -> nn.Module:
    """Instantiate one of the four supported architectures.

    Args:
        model_name: One of 'custom_cnn', 'resnet18', 'densenet121', or 'dinov3'.
        num_classes: Number of output classes (default: 4).
        pretrained: Whether to load pretrained backbone weights.
        dinov3_model_id: Hugging Face model ID for DINOv3.
    """
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        return resnet

    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)
        densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
        return densenet

    if model_name == "dinov3":
        token = os.environ.get("HF_TOKEN")
        if pretrained:
            backbone = AutoModel.from_pretrained(dinov3_model_id, token=token)
        else:
            config = AutoConfig.from_pretrained(dinov3_model_id, token=token)
            backbone = AutoModel.from_config(config)
        return DINOv3Classifier(backbone=backbone, num_classes=num_classes)

    raise ValueError(
        f"Unsupported model '{model_name}'. Choose from {SUPPORTED_MODELS}."
    )


def configure_fine_tuning_phase(
    model: nn.Module,
    model_name: str,
    phase: int = 1,
    low_lr: bool = False,
) -> torch.optim.Optimizer:
    """Freeze/unfreeze model layers and return the optimizer for a specific training phase.

    Phases for pretrained models:
        - Phase 1: Linear probing / feature extraction (only classification head).
        - Phase 2: Partial fine-tuning of top block(s) + classification head.
        - Phase 3: Deeper partial fine-tuning + classification head with decayed LR.
    """
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        for param in model.parameters():
            param.requires_grad = True
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    if model_name == "resnet18":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        if phase == 1:
            return torch.optim.Adam(model.fc.parameters(), lr=1e-3)
        elif phase == 2:
            for param in model.layer4.parameters():
                param.requires_grad = True
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
            )
        elif phase == 3:
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
            )
        raise ValueError(f"Invalid phase {phase} for resnet18 (expected 1, 2, or 3).")

    if model_name == "densenet121":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        if phase == 1:
            return torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        elif phase == 2:
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
            )
        elif phase == 3:
            for param in model.features.denseblock3.parameters():
                param.requires_grad = True
            for param in model.features.transition3.parameters():
                param.requires_grad = True
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
            )
        raise ValueError(f"Invalid phase {phase} for densenet121 (expected 1, 2, or 3).")

    if model_name == "dinov3":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        if phase == 1:
            return torch.optim.AdamW(
                model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
            )
        elif phase == 2:
            for param in model.backbone.layer[-2:].parameters():
                param.requires_grad = True
            for param in model.backbone.norm.parameters():
                param.requires_grad = True
            return torch.optim.AdamW(
                [
                    {"params": model.backbone.layer[-2:].parameters(), "lr": 1e-5},
                    {"params": model.backbone.norm.parameters(), "lr": 1e-5},
                    {"params": model.classifier.parameters(), "lr": 1e-4},
                ],
                weight_decay=1e-4,
            )
        elif phase == 3:
            for param in model.backbone.layer[-4:].parameters():
                param.requires_grad = True
            for param in model.backbone.norm.parameters():
                param.requires_grad = True

            if low_lr:
                param_groups = [
                    {"params": model.backbone.layer[8:10].parameters(), "lr": 1e-6},
                    {"params": model.backbone.layer[10:12].parameters(), "lr": 2e-6},
                    {"params": model.backbone.norm.parameters(), "lr": 2e-6},
                    {"params": model.classifier.parameters(), "lr": 2e-5},
                ]
            else:
                param_groups = [
                    {"params": model.backbone.layer[8:10].parameters(), "lr": 3e-6},
                    {"params": model.backbone.layer[10:12].parameters(), "lr": 5e-6},
                    {"params": model.backbone.norm.parameters(), "lr": 5e-6},
                    {"params": model.classifier.parameters(), "lr": 3e-5},
                ]
            return torch.optim.AdamW(param_groups, weight_decay=1e-4)
        raise ValueError(f"Invalid phase {phase} for dinov3 (expected 1, 2, or 3).")

    raise ValueError(f"Unsupported model '{model_name}'.")


def count_params(module: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: nn.Module) -> int:
    """Count trainable parameters (`requires_grad=True`) in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_unique_params(modules: Iterable[nn.Module]) -> int:
    """Count unique parameters across a list of submodules."""
    seen = set()
    total = 0
    for module in modules:
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total
