import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# =========================
# CONFIG
# =========================
SEED = 42
EPOCHS = 20
BATCH_SIZE = 64

torch.manual_seed(SEED)

# =========================
# 1. DATA (FashionMNIST 0~7 classes)
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

def filter_pretrain_classes(dataset):
    """클래스 0~7만 추출 (8, 9는 fine-tuning용으로 남겨둠)"""
    idx = [i for i, (_, y) in enumerate(dataset) if y in range(8)]
    return Subset(dataset, idx)

train_dataset = filter_pretrain_classes(train_dataset)
test_dataset = filter_pretrain_classes(test_dataset)

print(f"Pretrain data: {len(train_dataset)} train, {len(test_dataset)} test (classes 0-7)")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =========================
# 2. CNN (4-dim embedding) + 8-class classifier
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

class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.classifier = nn.Linear(4, 8)  # 8 classes (0~7)

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

# =========================
# 3. TRAIN
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = PretrainModel().to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
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

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

# =========================
# 4. SAVE CNN WEIGHTS
# =========================
final_acc = correct / total
print(f"\nFinal Test Accuracy: {final_acc:.4f}")

if final_acc >= 0.85:
    torch.save(model.cnn.state_dict(), "cnn_pretrained.pth")
    print("CNN weights saved to cnn_pretrained.pth")
else:
    print(f"Warning: Accuracy {final_acc:.4f} < 0.85, but saving anyway...")
    torch.save(model.cnn.state_dict(), "cnn_pretrained.pth")
    print("CNN weights saved to cnn_pretrained.pth")
