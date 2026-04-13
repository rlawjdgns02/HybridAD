import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random

import pennylane as qml

# =========================
# CONFIG
# =========================
N_SHOT = 3  # None = 전체 데이터 사용, 숫자 = 클래스당 샘플 수 (예: 5, 10, 50)
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# 1. DATA (FashionMNIST 0 vs 1)
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

def filter_2class(dataset):
    idx = [i for i, (_, y) in enumerate(dataset) if y in [0, 1]]
    return Subset(dataset, idx)

def filter_fewshot(dataset, n_shot, seed=42):
    """클래스당 n_shot개의 샘플만 추출"""
    random.seed(seed)
    class_indices = {0: [], 1: []}

    for i, (_, y) in enumerate(dataset):
        if y in [0, 1]:
            class_indices[y].append(i)

    selected = []
    for cls in [0, 1]:
        selected += random.sample(class_indices[cls], min(n_shot, len(class_indices[cls])))

    return Subset(dataset, selected)

# Few-shot 또는 전체 데이터
if N_SHOT is not None:
    train_dataset = filter_fewshot(train_dataset, N_SHOT, SEED)
    print(f"Few-shot setting: {N_SHOT}-shot (총 {len(train_dataset)} 샘플)")
else:
    train_dataset = filter_2class(train_dataset)
    print(f"Full data setting: {len(train_dataset)} 샘플")

test_dataset = filter_2class(test_dataset)

# 배치 크기: few-shot일 때는 샘플 수에 맞게 조정
batch_size = min(len(train_dataset), 32) if N_SHOT else 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# =========================
# 2. CNN (4-dim embedding)
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
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    # Angle embedding
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')

    # Entanglement (ZZ-like)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Variational layer
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    # 모든 qubit의 expectation value 반환 (4차원 출력)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": n_qubits}

qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# =========================
# 4. HYBRID MODEL
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.qnn = qlayer  # QNN은 CPU에서만 동작
        self.fc = nn.Linear(n_qubits, 2)  # 4차원 입력

    def to(self, device):
        # qnn은 CPU에 유지 (lightning.qubit은 CPU만 지원)
        self.cnn = self.cnn.to(device)
        self.fc = self.fc.to(device)
        return self

    def forward(self, x):
        original_device = x.device
        x = self.cnn(x)                     # (batch, 4)
        # QNN은 CPU에서 실행
        x = x.cpu()
        x = self.qnn(x)                     # (batch, 4) - 4차원 출력
        x = x.to(original_device)
        x = self.fc(x)                      # (batch, 2)
        return x

# =========================
# 5. TRAIN
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridModel().to(device)

# CNN도 함께 학습 (랜덤 초기화 상태에서 freeze하면 학습 불가)
# few-shot setting을 원하면 먼저 CNN을 pretrain하거나,
# 적은 샘플로 전체 모델을 학습해야 함

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

# =========================
# 6. TEST
# =========================
model.eval()
correct, total = 0, 0
all_preds = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        all_preds.extend(pred.cpu().tolist())

        correct += (pred == y).sum().item()
        total += y.size(0)

# 예측 분포 확인
print(f"Pred 0: {all_preds.count(0)}, Pred 1: {all_preds.count(1)}")
print(f"Accuracy: {correct / total:.4f}")