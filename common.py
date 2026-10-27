"""Shared data, model and device helpers for DeepFER."""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MEAN, STD = 0.5, 0.5


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_split(root, split):
    """Return [(path, label_idx)] for root/split/<class>/*.jpg."""
    items = []
    for idx, name in enumerate(CLASSES):
        d = os.path.join(root, split, name)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, f), idx))
    return items


def train_transform(size):
    # rotation, scaling and flipping, as specified in the project's objectives
    return T.Compose([
        T.Grayscale(1),
        T.Resize(size),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize([MEAN], [STD]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_transform(size):
    return T.Compose([T.Grayscale(1), T.Resize(size), T.ToTensor(), T.Normalize([MEAN], [STD])])


class FERDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = self.cache.get(path)
        if img is None:
            img = self.cache[path] = Image.open(path).convert("L")
        return self.transform(img), y


class DeepFERNet(nn.Module):
    """Custom CNN: 4 conv blocks (conv-BN-ReLU x2, pool, dropout) + GAP + MLP head. Input 1x48x48."""

    def __init__(self, n_classes=7):
        super().__init__()

        def block(i, o, p):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(p))

        self.features = nn.Sequential(block(1, 64, 0.1), block(64, 128, 0.2), block(128, 256, 0.3), block(256, 512, 0.3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256),
                                  nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ResNetTL(nn.Module):
    """Transfer learning: ImageNet ResNet-18, grayscale replicated to 3 channels, new 7-way head."""

    def __init__(self, n_classes=7, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.resnet18(weights=w)
        self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.net(x.repeat(1, 3, 1, 1))

    def set_backbone_frozen(self, frozen):
        for n, p in self.net.named_parameters():
            if not n.startswith("fc"):
                p.requires_grad = not frozen


# name -> (constructor, input size)
MODELS = {"cnn": (DeepFERNet, 48), "resnet18": (ResNetTL, 96)}


def build(name, **kw):
    ctor, size = MODELS[name]
    return ctor(**kw), size


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    model, size = build(ck["model"], **({"pretrained": False} if ck["model"] == "resnet18" else {}))
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), size
