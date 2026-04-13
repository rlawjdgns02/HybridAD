"""
EuroSAT Quantum Few-shot (논문 스타일 VQC)
- ResNet18 freeze
- Linear → AngleEmbedding → StronglyEntanglingLayers
- Minimal post-processing
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import pennylane as qml
from pennylane import numpy as np
from collections import defaultdict
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Config
# =============================================================================
n_qubits = 4
n_layers = 6
batch_size = 8
num_epochs = 30
lr = 0.01

K_SHOT = 20
NUM_CLASSES = 10
DATA_ROOT = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = qml.device("default.qubit", wires=n_qubits)

# =============================================================================
# Quantum Circuit (논문 스타일)
# =============================================================================
@qml.qnode(dev, interface="torch")
def quantum_net(inputs, weights):
    # Angle Encoding
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

    # Variational Layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# =============================================================================
# VQC Classifier
# =============================================================================
class VQCNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature → qubit dimension
        self.pre_net = nn.Linear(512, n_qubits)

        # Quantum weights
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3)
        )

        # Output classifier
        self.post_net = nn.Linear(n_qubits, NUM_CLASSES)

    def forward(self, x):
        x = self.pre_net(x)

        # scale to [-π, π]
        x = torch.tanh(x) * np.pi

        q_out = []
        for elem in x:
            out = quantum_net(elem, self.q_weights)
            q_out.append(torch.stack(out).float())

        q_out = torch.stack(q_out)

        return self.post_net(q_out)


# =============================================================================
# Data
# =============================================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_eurosat_fewshot(k_shot):
    dataset = datasets.EuroSAT(root=DATA_ROOT, download=True)

    targets = dataset.targets if hasattr(dataset, 'targets') else [dataset[i][1] for i in range(len(dataset))]

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    train_size = int(0.8 * len(indices))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    class_indices = defaultdict(list)
    for i, idx in enumerate(train_idx):
        class_indices[targets[idx]].append(i)

    selected = []
    for c in range(NUM_CLASSES):
        selected.extend(np.random.choice(class_indices[c], k_shot, replace=False))

    train_real = [train_idx[i] for i in selected]

    class Subset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            return transform(img), label

    train_set = Subset(dataset, train_real)
    test_set = Subset(dataset, test_idx[:1000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# Model
# =============================================================================
def build_model():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Identity()
    model = model.to(device)

    vqc = VQCNet().to(device)

    return model, vqc


# =============================================================================
# Training
# =============================================================================
def train(model, vqc, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vqc.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_acc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        model.eval()
        vqc.train()

        total, correct, loss_sum = 0, 0, 0

        for imgs, labels in tqdm(train_loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                feats = model(imgs)

            outputs = vqc(feats)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        print(f"Train Acc: {100*correct/total:.2f}")

        # Eval
        vqc.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = model(imgs)
                outputs = vqc(feats)

                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        acc = 100 * correct / total
        print(f"Val Acc: {acc:.2f}")

        if acc > best_acc:
            best_acc = acc

        scheduler.step()

    print(f"\nBest Acc: {best_acc:.2f}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    train_loader, test_loader = load_eurosat_fewshot(K_SHOT)

    backbone, vqc = build_model()

    print("Start Training...")
    train(backbone, vqc, train_loader, test_loader)