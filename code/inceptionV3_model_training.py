# Standard library imports
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


def get_run_directory():
    """Create a unique directory name for this training run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"deepfake_inceptionV3_{timestamp}"


def create_file_structure(base_dir="experiments"):
    """Create organized file structure for the experiment"""
    run_dir = os.path.join(base_dir, get_run_directory())

    dirs = {
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "plots": os.path.join(run_dir, "plots"),
        "metrics": os.path.join(run_dir, "metrics"),
        "logs": os.path.join(run_dir, "logs"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def get_file_paths(dirs):
    """Generate organized file paths for all outputs"""
    files = {
        "best_model": os.path.join(dirs["checkpoints"], "best_model.pth"),
        "final_model": os.path.join(dirs["checkpoints"], "final_model.pth"),
        "training_history": os.path.join(dirs["plots"], "training_history.png"),
        "learning_rate": os.path.join(dirs["plots"], "learning_rate.png"),
        "val_confusion_matrix": os.path.join(dirs["plots"], "val_confusion_matrix.png"),
        "val_roc_curve": os.path.join(dirs["plots"], "val_roc_curve.png"),
        "val_pr_curve": os.path.join(dirs["plots"], "val_precision_recall.png"),
        "test_confusion_matrix": os.path.join(
            dirs["plots"], "test_confusion_matrix.png"
        ),
        "test_roc_curve": os.path.join(dirs["plots"], "test_roc_curve.png"),
        "test_pr_curve": os.path.join(dirs["plots"], "test_precision_recall.png"),
        "training_history_json": os.path.join(dirs["metrics"], "training_history.json"),
        "validation_results": os.path.join(dirs["metrics"], "validation_metrics.json"),
        "test_results": os.path.join(dirs["metrics"], "test_metrics.json"),
        "full_results": os.path.join(dirs["metrics"], "complete_results.json"),
        "training_log": os.path.join(dirs["logs"], "training.log"),
        "evaluation_log": os.path.join(dirs["logs"], "evaluation.log"),
        "model_config": os.path.join(dirs["logs"], "model_config.json"),
    }
    return files


def check_gpu():
    """Check and print GPU setup information"""
    logging.info("\n=== GPU Setup ===")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU device count: {torch.cuda.device_count()}")
        logging.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
        logging.info(f"Current GPU device: {torch.cuda.current_device()}")
    logging.info("================\n")


class DeepFakeDetector(nn.Module):
    def __init__(self, freeze_backbone=True, num_classes=2):
        super(DeepFakeDetector, self).__init__()

        # Load the InceptionV3 model with pretrained weights
        self.inceptionv3 = models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1
        )

        num_ftrs = (
            self.inceptionv3.fc.in_features
        )  # The number of features from the last layer
        dropout_rate = 0.25

        # Replace the classifier with custom layers
        self.inceptionv3.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.Dropout(dropout_rate),
            nn.Linear(
                256, num_classes
            ),  # Output layer with 2 logits (for binary classification)
        )

        self._initialize_weights()

    def forward(self, x):
        # Forward pass through the InceptionV3 model
        return self.inceptionv3(x)

    def _initialize_weights(self):
        """Custom weight initialization for added layers."""
        for m in self.inceptionv3.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curve(precision, recall, avg_precision, output_path):
    plt.figure(figsize=(10, 8))
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"PR curve (AP = {avg_precision:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_progress(history, output_path):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["val_f1"], label="F1 Score")
    plt.plot(history["val_precision"], label="Precision")
    plt.plot(history["val_recall"], label="Recall")
    plt.title("Metrics vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Training", file=sys.stdout, leave=True)

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Handle InceptionOutputs
        if isinstance(outputs, models.InceptionOutputs):
            outputs = outputs.logits  # Extract the logits

        # Compute loss
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / (batch_idx + 1)
        progress_bar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.2f}%",
                "gpu_mem": f"{torch.cuda.memory_allocated()/1024**2:.1f}MB",
            }
        )

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    return epoch_loss, epoch_acc, precision, recall, f1


def evaluate_model(
    model, dataloader, criterion, device, classes, file_paths, phase="test"
):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    logging.info(f"\nPerforming {phase} set evaluation...")
    progress_bar = tqdm(dataloader, desc=f"Evaluating", file=sys.stdout, leave=True)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader)
    test_acc = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)

    if phase == "test":
        plot_confusion_matrix(cm, classes, file_paths["test_confusion_matrix"])
        plot_roc_curve(fpr, tpr, roc_auc, file_paths["test_roc_curve"])
        plot_precision_recall_curve(
            precision_curve, recall_curve, avg_precision, file_paths["test_pr_curve"]
        )
    else:
        plot_confusion_matrix(cm, classes, file_paths["val_confusion_matrix"])
        plot_roc_curve(fpr, tpr, roc_auc, file_paths["val_roc_curve"])
        plot_precision_recall_curve(
            precision_curve, recall_curve, avg_precision, file_paths["val_pr_curve"]
        )

    results = {
        "loss": test_loss,
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": cm.tolist(),
    }

    results_path = (
        file_paths["test_results"]
        if phase == "test"
        else file_paths["validation_results"]
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"\n{phase.capitalize()} Set Evaluation Results:")
    logging.info(f"Loss: {test_loss:.4f}")
    logging.info(f"Accuracy: {test_acc:.2f}%")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info(f"Average Precision: {avg_precision:.4f}")

    return results


def validate(model, dataloader, criterion, device):
    """Validate the model on the validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(all_labels, all_preds)

    return val_loss, val_acc, precision, recall, f1, roc_auc, cm, fpr, tpr


def main():
    # Set for deterministic behavior
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Create directory structure
    dirs = create_file_structure()
    file_paths = get_file_paths(dirs)

    # Set up logging
    logging.basicConfig(
        filename=file_paths["training_log"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Add console handler for logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    # Model configuration
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

    with open(file_paths["model_config"], "w") as f:
        json.dump(model_config, f, indent=4)

    # Check GPU setup
    check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Dataset paths
    data_dir = "../data/dataset"

    # Hyperparameters
    batch_size = model_config["batch_size"]
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["initial_lr"]

    # Data transforms with normalization
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

    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "val"), transform=transform
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "test"), transform=transform
        )

        logging.info(f"\nDataset Statistics:")
        logging.info(f"Training images: {len(train_dataset)}")
        logging.info(f"Validation images: {len(val_dataset)}")
        logging.info(f"Test images: {len(test_dataset)}")
        logging.info(f"Classes: {train_dataset.classes}\n")

    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Initialize model
    model = DeepFakeDetector(freeze_backbone=True)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    try:
        logging.info("\nStarting training...\n")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logging.info(f"\nEpoch [{epoch+1}/{num_epochs}]")

            if device.type == "cuda":
                logging.info(
                    f"GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB"
                )

            # Train
            train_loss, train_acc, train_precision, train_recall, train_f1 = (
                train_epoch(model, train_loader, criterion, optimizer, device)
            )

            # Validate
            (
                val_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
                val_auc,
                cm,
                fpr,
                tpr,
            ) = validate(model, val_loader, criterion, device)

            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Update history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_f1"].append(val_f1)
            history["val_auc"].append(val_auc)

            # Save training history
            with open(file_paths["training_history_json"], "w") as f:
                json.dump(history, f, indent=4)

            # Plot current progress
            plot_training_progress(history, file_paths["training_history"])

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Log epoch summary
            logging.info(f"\nEpoch Summary:")
            logging.info(f"Time: {epoch_time:.2f}s")
            logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logging.info(f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
            logging.info(f"Learning Rate: {current_lr:.2e}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "val_auc": val_auc,
                    },
                    file_paths["best_model"],
                )
                logging.info(
                    f"Saved new best model with validation accuracy: {val_acc:.2f}%"
                )

            # Memory cleanup
            torch.cuda.empty_cache()

        # Save the final model
        logging.info("\nSaving final model...")
        torch.save(
            {
                "epoch": num_epochs - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
            },
            file_paths["final_model"],
        )
        logging.info(f"Final model saved to {file_paths['final_model']}.")

        total_time = time.time() - start_time
        logging.info(f"\nTraining completed in {total_time/60:.2f} minutes!")
        logging.info("Starting model evaluation...")

        # Load best model for evaluation
        checkpoint = torch.load(file_paths["best_model"])
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on validation and test sets
        logging.info("\nEvaluating on validation set...")
        val_results = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            train_dataset.classes,
            file_paths,
            phase="val",
        )

        logging.info("\nEvaluating on test set...")
        test_results = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            train_dataset.classes,
            file_paths,
            phase="test",
        )

        # Save combined results
        combined_results = {
            "validation": val_results,
            "test": test_results,
            "training_history": history,
            "training_time_minutes": total_time / 60,
            "best_validation_accuracy": best_val_acc,
        }

        with open(file_paths["full_results"], "w") as f:
            json.dump(combined_results, f, indent=4)

        logging.info(f"\nAll results saved in: {dirs['metrics']}")

    except Exception as e:
        logging.error(f"\nError during training or evaluation: {str(e)}")
        logging.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
