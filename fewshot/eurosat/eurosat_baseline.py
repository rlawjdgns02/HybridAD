"""
EuroSAT Baseline: Pretrained ResNet + Fine-tuning (Classical)
Reference: https://arxiv.org/pdf/1709.00029 (EuroSAT paper)
Expected accuracy: ~96% with ResNet-50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Config
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data"

# Hyperparameters (from EuroSAT paper)
BATCH_SIZE = 64
EPOCHS_HEAD = 10      # Last layer만 학습
EPOCHS_FINETUNE = 20  # 전체 fine-tuning
LR_HEAD = 0.01        # Last layer 학습 시
LR_FINETUNE = 0.001   # Fine-tuning 시
WEIGHT_DECAY = 1e-4

# EuroSAT classes
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
NUM_CLASSES = 10

# =============================================================================
# Data
# =============================================================================
def get_transforms():
    """ImageNet pretrained 모델용 transform (64->224 resize)"""
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 64->224 (성능 향상)
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def load_eurosat(train_transform, test_transform, train_ratio=0.8):
    """EuroSAT 데이터셋 로드 (80/20 split)"""
    # 전체 데이터셋 (transform 없이 로드 후 split)
    full_dataset = datasets.EuroSAT(
        root=DATA_ROOT,
        download=True,
        transform=None
    )

    # 80/20 split
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Transform 적용을 위한 wrapper
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = TransformDataset(train_dataset, train_transform)
    test_dataset = TransformDataset(test_dataset, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    return train_loader, test_loader


# =============================================================================
# Model
# =============================================================================
def create_model(backbone='resnet18', pretrained=True, freeze_backbone=True):
    """
    Pretrained ResNet + 새로운 classifier head

    Args:
        backbone: 'resnet18' or 'resnet50'
        pretrained: ImageNet pretrained 사용 여부
        freeze_backbone: backbone freeze 여부
    """
    if backbone == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = 512
    elif backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    model.fc = nn.Linear(feature_dim, NUM_CLASSES)

    return model.to(DEVICE)


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy * 100, all_preds, all_labels


def train_model(model, train_loader, test_loader, epochs, lr, phase_name="Training"):
    """모델 학습"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0

    print(f"\n=== {phase_name} (lr={lr}, epochs={epochs}) ===")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    return history, best_acc


# =============================================================================
# Visualization
# =============================================================================
def plot_history(history, save_path="results/fewshot/classical_baseline"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['test_loss'], label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['test_acc'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"EuroSAT Baseline Training")
    print("=" * 50)

    # Data
    train_transform, test_transform = get_transforms()
    train_loader, test_loader = load_eurosat(train_transform, test_transform)

    # Model
    model = create_model(backbone='resnet18', pretrained=True, freeze_backbone=True)
    print(f"\nModel: ResNet18 (pretrained, backbone frozen)")

    # Phase 1: Train head only
    history1, best_acc1 = train_model(
        model, train_loader, test_loader,
        epochs=EPOCHS_HEAD,
        lr=LR_HEAD,
        phase_name="Phase 1: Head Only"
    )

    # Phase 2: Fine-tune entire network
    for param in model.parameters():
        param.requires_grad = True

    history2, best_acc2 = train_model(
        model, train_loader, test_loader,
        epochs=EPOCHS_FINETUNE,
        lr=LR_FINETUNE,
        phase_name="Phase 2: Fine-tuning"
    )

    # Combine histories
    full_history = {
        'train_loss': history1['train_loss'] + history2['train_loss'],
        'train_acc': history1['train_acc'] + history2['train_acc'],
        'test_loss': history1['test_loss'] + history2['test_loss'],
        'test_acc': history1['test_acc'] + history2['test_acc'],
    }

    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)

    criterion = nn.CrossEntropyLoss()
    _, final_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    print(f"Best Accuracy (Phase 1): {best_acc1:.2f}%")
    print(f"Best Accuracy (Phase 2): {best_acc2:.2f}%")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASSES))

    # Plot
    plot_history(full_history)

    # Save model
    torch.save(model.state_dict(), "eurosat_baseline.pth")
    print("Model saved: eurosat_baseline.pth")

    return model, full_history


if __name__ == "__main__":
    main()
