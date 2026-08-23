import torch
import torch.nn as nn
from torchvision import models

from configs.classes import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, freeze_backbone=True, unfreeze_layer4=False):
    """ImageNet 사전학습 ResNet18의 FC를 교체한다.

    Baseline은 freeze_backbone=True (classifier만 학습),
    Fine-tuning 단계에서는 unfreeze_layer4=True로 상위 블록을 연다.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, path, extra=None):
    payload = {"state_dict": model.state_dict(), "num_classes": model.fc.out_features}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device=None):
    device = device or get_device()
    payload = torch.load(path, map_location=device)
    model = build_model(payload["num_classes"], freeze_backbone=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
