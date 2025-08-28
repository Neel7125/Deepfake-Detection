import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchvision.models import Inception_V3_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "../data/dataset"

model_config = {
    "architecture": "VGG16",
    "backbone_frozen": True,
    "num_classes": 2,
    "optimizer": "Adam",
    "initial_lr": 0.001,
    "batch_size": 32,
    "num_epochs": 20,
    "transforms": {
        # "resize": (229, 229),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
}

transform = transforms.Compose(
[
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=model_config["transforms"]["normalize_mean"],
        std=model_config["transforms"]["normalize_std"],
    ),
]
)

train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "train"), transform=transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"), transform=transform
)
test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "test"), transform=transform
)

print(train_dataset.classes)