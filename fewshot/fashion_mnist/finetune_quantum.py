import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import os

import pennylane as qml

# =========================
# CONFIG
# =========================
N_SHOT = 5  # 클래스당 샘플 수 (1, 3, 5, 10, ...)
SEED = 42
EPOCHS = 50

random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# 1. DATA (FashionMNIST 8, 9 classes only)
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

def filter_finetune_classes(dataset):
    """클래스 8, 9만 추출"""
    idx = [i for i, (_, y) in enumerate(dataset) if y in [8, 9]]
    return Subset(dataset, idx)

def filter_fewshot(dataset, n_shot, seed=42):
    """클래스당 n_shot개의 샘플만 추출"""
    random.seed(seed)
    class_indices = {8: [], 9: []}

    for i, (_, y) in enumerate(dataset):
        if y in [8, 9]:
            class_indices[y].append(i)

    selected = []
    for cls in [8, 9]:
        selected += random.sample(class_indices[cls], min(n_shot, len(class_indices[cls])))

    return Subset(dataset, selected)

# Few-shot train, full test
train_dataset = filter_fewshot(train_dataset, N_SHOT, SEED)
test_dataset = filter_finetune_classes(test_dataset)

print(f"Fine-tune setting: {N_SHOT}-shot (train: {len(train_dataset)}, test: {len(test_dataset)})")

# Few-shot에서는 full batch 사용 (안정적인 gradient)
batch_size = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# =========================
# 2. CNN (pretrained, frozen)
# =========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 5 * 5, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.tanh(x)  # [-1,1]

# =========================
# 3. QUANTUM CIRCUIT (PennyLane)
# =========================
n_qubits = 4
n_layers = 3  # 레이어 깊이
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    # Angle embedding
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')

    # Multiple variational layers
    for layer in range(n_layers):
        # Entanglement (circular)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Variational rotations (RY, RZ)
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

    # 첫 2큐빗의 측정 확률 반환 → 2^2 = 4차원 출력 (fc 입력 차원과 일치)
    return qml.probs(wires=[0, 1])

# 3층 × 4큐빗 × 2회전 = 24 파라미터 (기존 4개에서 증가)
weight_shapes = {"weights": (n_layers, n_qubits, 2)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# =========================
# 4. HYBRID MODEL
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.qnn = qlayer
        self.fc = nn.Linear(4, 2) #2 classes

    def to(self, device):
        self.cnn = self.cnn.to(device)
        self.fc = self.fc.to(device)
        return self

    def forward(self, x):
        original_device = x.device
        x = self.cnn(x)           # (batch, 4)
        x = x.cpu()               # QNN은 CPU에서 실행
        x = self.qnn(x)           # (batch, 4)
        x = x.to(original_device)
        x = self.fc(x)            # (batch, 2)
        return x

# =========================
# 5. LOAD PRETRAINED & FREEZE CNN
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = HybridModel().to(device)

# Load pretrained CNN weights
model.cnn.load_state_dict(torch.load("cnn_pretrained.pth", map_location=device))
print("Loaded pretrained CNN weights")

# Freeze CNN
for param in model.cnn.parameters():
    param.requires_grad = False
print("CNN frozen - only training QNN + fc")

# =========================
# 6. TRAIN (QNN + fc only)
# =========================
trainable_params = list(model.qnn.parameters()) + list(model.fc.parameters())
opt = torch.optim.Adam(trainable_params, lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# History 기록
history = {
    "loss": [],
    "acc": []
}

# Early stopping 설정
import copy
best_acc = 0
best_state = None
best_epoch = 0
patience = 1  # 5 epoch 동안 개선 없으면 중단
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # 라벨 변환: 8 -> 0, 9 -> 1
        y = y - 8

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    history["loss"].append(avg_loss)

    # Epoch별 accuracy 계산
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y = y - 8
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    history["acc"].append(acc)

    # Best model 저장
    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        no_improve = 0
    else:
        no_improve += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Acc: {acc:.4f} (best: {best_acc:.4f} @ ep{best_epoch})")

    # Early stopping
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Best model 복원
model.load_state_dict(best_state)
print(f"Restored best model from epoch {best_epoch} (acc: {best_acc:.4f})")

# =========================
# 7. FINAL TEST & RESULTS
# =========================
print(f"\n[Quantum] {N_SHOT}-shot Best Accuracy: {best_acc:.4f} (epoch {best_epoch})")

# =========================
# 8. SAVE MODEL & PLOTS
# =========================
os.makedirs("results", exist_ok=True)

# 모델 저장
torch.save({
    "model_state_dict": model.state_dict(),
    "n_shot": N_SHOT,
    "best_acc": best_acc,
    "best_epoch": best_epoch,
    "history": history
}, f"results/quantum_{N_SHOT}shot.pth")
print(f"Model saved to results/quantum_{N_SHOT}shot.pth")

# Loss & Accuracy 그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["loss"], label="Loss", color="red")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title(f"Quantum {N_SHOT}-shot: Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["acc"], label="Accuracy", color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title(f"Quantum {N_SHOT}-shot: Accuracy (best: {best_acc:.4f} @ ep{best_epoch})")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"results/quantum_{N_SHOT}shot.png", dpi=150)
plt.show()
print(f"Plot saved to results/quantum_{N_SHOT}shot.png")
