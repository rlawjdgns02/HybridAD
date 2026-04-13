"""
EuroSAT Classical Few-shot: Pretrained ResNet + MLP Fine-tuning
k-shot per class 학습
처음부터 전체 fine-tuning (Phase 1 없음)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Config
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data"

# Few-shot config
K_SHOT = 10  # class당 샘플 수

# Hyperparameters
BATCH_SIZE = 16  # few-shot이라 작게
EPOCHS = 30  # 전체 fine-tuning
LR = 0.001
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
    """ImageNet pretrained 모델용 transform"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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


def sample_k_shot(dataset, k_shot, seed=42):
    """Class당 k개씩 샘플링"""
    np.random.seed(seed)

    # 클래스별 인덱스 수집
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    # 클래스당 k개 샘플링
    selected_indices = []
    for label in range(NUM_CLASSES):
        indices = class_indices[label]
        if len(indices) < k_shot:
            raise ValueError(f"Class {label} has only {len(indices)} samples, need {k_shot}")
        selected = np.random.choice(indices, k_shot, replace=False)
        selected_indices.extend(selected)

    return selected_indices


def load_eurosat_fewshot(train_transform, test_transform, k_shot=K_SHOT):
    """Few-shot EuroSAT 데이터셋 로드"""

    # 전체 데이터셋 로드
    full_dataset = datasets.EuroSAT(root=DATA_ROOT, download=True, transform=None)

    # 80/20 split (baseline과 동일하게)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)

    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Train subset에서 k-shot 샘플링
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, label = self.dataset[real_idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    # Train set 만들기 (k-shot 샘플링용)
    train_subset = TransformDataset(full_dataset, train_indices, None)

    # k-shot 인덱스 샘플링
    kshot_indices = sample_k_shot(train_subset, k_shot)

    # 실제 전체 데이터셋 인덱스로 변환
    kshot_real_indices = [train_indices[i] for i in kshot_indices]

    # 최종 데이터셋
    train_dataset = TransformDataset(full_dataset, kshot_real_indices, train_transform)
    test_dataset = TransformDataset(full_dataset, test_indices, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"K-shot: {k_shot} per class")
    print(f"Train: {len(train_dataset)} ({k_shot} x {NUM_CLASSES} classes)")
    print(f"Test: {len(test_dataset)}")

    return train_loader, test_loader


# =============================================================================
# Model
# =============================================================================
def create_model(backbone='resnet18', pretrained=True, freeze_backbone=True):
    """Pretrained ResNet + MLP classifier"""

    if backbone == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = 512
    elif backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # MLP classifier
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

    for images, labels in loader:
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
        for images, labels in loader:
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
def plot_history(history, k_shot, save_path=None):
    if save_path is None:
        save_path = f"results/fewshot/classical_{k_shot}/eurosat_classical_{k_shot}shot_history.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['test_loss'], label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss ({k_shot}-shot)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['test_acc'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Accuracy ({k_shot}-shot)')
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
    print(f"EuroSAT Classical Few-shot ({K_SHOT}-shot)")
    print("=" * 50)

    # Data
    train_transform, test_transform = get_transforms()
    train_loader, test_loader = load_eurosat_fewshot(train_transform, test_transform, K_SHOT)

    # Model - 처음부터 전체 학습 (freeze 안 함)
    model = create_model(backbone='resnet18', pretrained=True, freeze_backbone=False)
    print(f"\nModel: ResNet18 + MLP (pretrained, full fine-tuning)")

    # Fine-tuning (단일 phase)
    history, best_acc = train_model(
        model, train_loader, test_loader,
        epochs=EPOCHS,
        lr=LR,
        phase_name="Fine-tuning"
    )

    # Final evaluation
    print("\n" + "=" * 50)
    print(f"Final Evaluation ({K_SHOT}-shot)")
    print("=" * 50)

    criterion = nn.CrossEntropyLoss()
    _, final_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    print(f"Best Accuracy: {best_acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASSES))

    # Plot
    plot_history(history, K_SHOT)

    # Save model
    save_path = f"results/fewshot/classical_{K_SHOT}/eurosat_classical_{K_SHOT}shot.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")

    return model, history, final_acc


if __name__ == "__main__":
    main()
