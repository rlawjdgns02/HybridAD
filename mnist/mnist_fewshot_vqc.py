"""
mnist_fewshot_vqc.py
Feature Extractor + VQC for Few-shot Anomaly Detection (SVDD style)

수정사항:
  1. Center 고정 (collapse 방지)
  2. Weight decay 추가 (trivial mapping 방지)
  3. Center 초기화를 워밍업 이후로 변경
  4. VQC forward 배치 처리 개선
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# ── Config ────────────────────────────────────────────────────────────────────
NORMAL_CLASS  = 0
K_SHOT        = 10
N_QUBITS      = 4
N_LAYERS      = 3
EPOCHS        = 200
WARMUP_EPOCHS = 20      # center 초기화 전 워밍업
LR            = 0.01
WEIGHT_DECAY  = 1e-4
DATA_DIR      = "./data"
SAVE_DIR      = "./results/fewshot"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. CNN Feature Extractor ──────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self, n_qubits=N_QUBITS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32 * 7 * 7, n_qubits)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = torch.sigmoid(x) * np.pi  # [0, π] 범위
        return x


# ── 2. VQC ────────────────────────────────────────────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class VQC(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, dtype=torch.float32) * 0.1
        )

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            exp_vals = quantum_circuit(x[i], self.weights)
            outputs.append(torch.stack(exp_vals))
        return torch.stack(outputs)


# ── 3. Hybrid Model ───────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.feature_extractor = FeatureExtractor(n_qubits=n_qubits)
        self.vqc = VQC(n_qubits=n_qubits, n_layers=n_layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.vqc(features)
        return outputs


# ── 4. 데이터 샘플링 ──────────────────────────────────────────────────────────
def sample_fewshot_support(dataset, normal_class, k_shot, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    normal_indices = [
        idx for idx, (_, label) in enumerate(dataset)
        if label == normal_class
    ]
    sampled = random.sample(normal_indices, k_shot)
    return Subset(dataset, sampled)


def get_query_data(dataset, normal_class, support_set, max_per_class=100):
    support_idx_set = set(support_set.indices)
    query_indices = []
    normal_count = anomaly_count = 0

    for idx, (_, label) in enumerate(dataset):
        if idx in support_idx_set:
            continue
        if label == normal_class and normal_count < max_per_class:
            query_indices.append(idx)
            normal_count += 1
        elif label != normal_class and anomaly_count < max_per_class:
            query_indices.append(idx)
            anomaly_count += 1

    return Subset(dataset, query_indices)


# ── 5. 학습 ───────────────────────────────────────────────────────────────────
def train(model, train_x, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, lr=LR):
    """
    핵심 수정사항:
      - warmup_epochs 동안 임시 center로 학습 (모델 초기화)
      - warmup 이후 center 고정 → 이후 절대 갱신 안 함
      - weight_decay로 trivial mapping 방지
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    history = {"loss": [], "center_fixed_epoch": warmup_epochs}
    center = None

    for epoch in range(1, epochs + 1):
        model.train()
        outputs = model(train_x.to(DEVICE))

        if epoch <= warmup_epochs:
            # 워밍업: 임시 center로 학습 (모델 초기화 목적)
            temp_center = outputs.mean(dim=0, keepdim=True).detach()
            distances = torch.norm(outputs - temp_center, dim=1)
        else:
            # 워밍업 직후 center 고정
            if center is None:
                model.eval()
                with torch.no_grad():
                    init_outputs = model(train_x.to(DEVICE))
                    center = init_outputs.mean(dim=0).detach()
                model.train()
                print(f"\n[Center Fixed] epoch {epoch}, center: {center.cpu().numpy()}")

            distances = torch.norm(outputs - center, dim=1)

        loss = distances.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss: {loss.item():.5f}"
                  + (" [warmup]" if epoch <= warmup_epochs else ""))

    # 최종 center (고정된 것 사용)
    if center is None:
        model.eval()
        with torch.no_grad():
            center = model(train_x.to(DEVICE)).mean(dim=0).detach()

    return model, history, center


# ── 6. 평가 ───────────────────────────────────────────────────────────────────
def evaluate(model, test_x, test_y, center):
    model.eval()
    with torch.no_grad():
        outputs = model(test_x.to(DEVICE))
        distances = torch.norm(outputs - center.unsqueeze(0), dim=1)
        anomaly_scores = distances.cpu().numpy()

    auc = roc_auc_score(test_y.numpy(), anomaly_scores)

    normal_scores  = anomaly_scores[test_y.numpy() == 0]
    anomaly_only   = anomaly_scores[test_y.numpy() == 1]
    threshold      = np.percentile(normal_scores, 95)
    preds          = (anomaly_scores > threshold).astype(int)
    acc            = accuracy_score(test_y.numpy(), preds)
    cm             = confusion_matrix(test_y.numpy(), preds)

    print(f"\n[Evaluation]")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Threshold : {threshold:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"\n  Score Distribution:")
    print(f"    Normal  - mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
    print(f"    Anomaly - mean: {anomaly_only.mean():.4f},  std: {anomaly_only.std():.4f}")

    return {
        "auc": auc, "accuracy": acc, "threshold": threshold,
        "anomaly_scores": anomaly_scores,
    }


# ── 7. 시각화 ─────────────────────────────────────────────────────────────────
def plot_results(history, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss curve
    axes[0].plot(history["loss"])
    axes[0].axvline(
        x=history["center_fixed_epoch"],
        color="red", linestyle="--", label="center fixed"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Anomaly score distribution
    scores = results["anomaly_scores"]
    axes[1].hist(scores, bins=30, alpha=0.7, density=True)
    axes[1].axvline(
        x=results["threshold"], color="red",
        linestyle="--", label=f"threshold ({results['threshold']:.3f})"
    )
    axes[1].set_xlabel("Anomaly Score")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Anomaly Score Distribution (AUC={results['auc']:.4f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fewshot_vqc_results.png", dpi=150)
    plt.close()
    print(f"[Saved] {save_dir}/fewshot_vqc_results.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Few-shot: {K_SHOT}-shot (normal class = {NORMAL_CLASS})")
    print(f"VQC: {N_QUBITS} qubits, {N_LAYERS} layers")
    print(f"Warmup: {WARMUP_EPOCHS} epochs, then center fixed\n")

    # 데이터 로딩
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)

    # Support Set
    print(f"[Data] Sampling {K_SHOT}-shot support set...")
    support_set = sample_fewshot_support(train_full, NORMAL_CLASS, K_SHOT)

    # Query Set
    query_set = get_query_data(train_full, NORMAL_CLASS, support_set)
    print(f"  Support: {len(support_set)} / Query: {len(query_set)}")

    # 텐서 변환
    support_x = torch.stack([x for x, _ in support_set])
    query_x   = torch.stack([x for x, _ in query_set])
    query_y   = torch.tensor(
        [0 if y == NORMAL_CLASS else 1 for _, y in query_set],
        dtype=torch.float32
    )

    # 모델 생성
    model = HybridModel(n_qubits=N_QUBITS, n_layers=N_LAYERS).to(DEVICE)

    # 학습
    print("\n[Train] End-to-End (CNN + VQC) with fixed center...")
    model, history, center = train(model, support_x)
    print(f"\n  Final center: {center.cpu().numpy()}")

    # 평가
    print("\n[Evaluate]")
    results = evaluate(model, query_x, query_y, center)

    # 시각화
    plot_results(history, results, SAVE_DIR)
    print("\n[Done]")