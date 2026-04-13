"""
vqc_classifier.py
PennyLane 기반 Variational Quantum Classifier (VQC) - SVDD 스타일

파이프라인:
  Autoencoder latent → PCA (8차원) → VQC → One-Class Anomaly Detection

학습 방식 (Deep SVDD 스타일):
  - 정상 데이터의 VQC 출력을 center(0)로 당김
  - Loss = MSE(output, center)
  - Anomaly Score = |output - center| (정상: 작음, 이상: 큼)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────
import argparse
import matplotlib
parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
parser.add_argument("--visualize", action="store_true", help="Load saved model and visualize only")
parser.add_argument("--no-show", action="store_true", help="Don't display plots (for automation)")
args, _ = parser.parse_known_args()

if args.no_show:
    matplotlib.use('Agg')

CATEGORY     = args.category
N_QUBITS     = 8
N_LAYERS     = 4        # Variational layer 반복 횟수
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 0.005    # SVDD는 좀 더 작은 lr이 안정적
CENTER       = 0.0      # 정상 데이터가 모일 목표점
DATA_DIR     = f"./preprocessed/mvtec/{CATEGORY}"
SAVE_DIR     = f"./results/mvtec/{CATEGORY}"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ── 1. Quantum Circuit 정의 ──────────────────────────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    VQC 회로 구조:
    1. Angle Encoding: 입력 데이터를 RY 게이트로 인코딩
    2. Variational Layers: CNOT entanglement + RY/RZ rotations
    3. Measurement: 첫 번째 qubit의 기댓값 반환

    Args:
        inputs: (N_QUBITS,) - PCA 축소된 latent vector [0, π]
        weights: (N_LAYERS, N_QUBITS, 2) - 학습 가능한 파라미터

    Returns:
        float: 첫 번째 qubit의 Pauli-Z 기댓값 [-1, 1]
    """
    # 1. Angle Encoding (데이터 임베딩)
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # 2. Variational Layers
    for layer in range(N_LAYERS):
        # Entanglement: 순환 CNOT
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

        # Rotation gates (학습 파라미터)
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

    # 3. Measurement
    return qml.expval(qml.PauliZ(0))


# ── 2. VQC 모델 (PyTorch 연동) ────────────────────────────────────────────────
class VQCClassifier(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 학습 가능한 양자 회로 파라미터
        weight_shape = (n_layers, n_qubits, 2)  # (layers, qubits, 2 rotations)
        self.weights = nn.Parameter(
            torch.randn(weight_shape, dtype=torch.float32) * 0.1
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_qubits) - VQC 입력 데이터
        Returns:
            (batch_size,) - 각 샘플의 VQC 출력 [-1, 1]

        SVDD 방식: 변환 없이 raw expectation value 반환
        정상 데이터는 center(0)에 가깝게, 이상은 멀리
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            expectation = quantum_circuit(x[i], self.weights)
            outputs.append(expectation)

        return torch.stack(outputs).float()


# ── 3. 학습 (SVDD 스타일) ──────────────────────────────────────────────────────
def train_vqc(model, train_loader, val_data, epochs=EPOCHS, lr=LR, center=CENTER):
    """
    VQC 학습 (SVDD 스타일 - 정상 데이터만으로 학습)

    학습 전략:
    - 정상 데이터의 출력을 center로 당김
    - Loss = MSE(output, center)
    - 이상 데이터는 학습에 사용 안 함 → 자연스럽게 center에서 멀어짐
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {"train_loss": [], "val_auc": []}

    val_x, val_y = val_data
    center_tensor = torch.tensor(center, dtype=torch.float32, device=DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_batch, _ in train_loader:  # label 사용 안 함
            x_batch = x_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_batch)

            # SVDD Loss: 출력을 center로 당김
            loss = torch.mean((outputs - center_tensor) ** 2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Validation AUC (anomaly score = |output - center|)
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x.to(DEVICE)).cpu().numpy()
            # Anomaly score: center에서 멀수록 이상
            anomaly_scores = np.abs(val_outputs - center)
            val_auc = roc_auc_score(val_y.numpy(), anomaly_scores)
            history["val_auc"].append(val_auc)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss: {avg_loss:.5f}  val_AUC: {val_auc:.4f}")

    return model, history


# ── 4. 평가 (SVDD 스타일) ──────────────────────────────────────────────────────
def evaluate(model, test_x, test_y, center=CENTER, threshold=None):
    """
    모델 평가 (SVDD 스타일)

    Anomaly Score = |output - center|
    - 정상: center 근처 → score 낮음
    - 이상: center에서 멀리 → score 높음
    """
    model.eval()
    with torch.no_grad():
        outputs = model(test_x.to(DEVICE)).cpu().numpy()

    # Anomaly score: center에서의 거리
    anomaly_scores = np.abs(outputs - center)

    # AUC (threshold-free metric)
    auc = roc_auc_score(test_y.numpy(), anomaly_scores)

    # Threshold 자동 결정 (validation 기반 또는 percentile)
    if threshold is None:
        # 정상 데이터의 95 percentile을 threshold로
        normal_scores = anomaly_scores[test_y.numpy() == 0]
        threshold = np.percentile(normal_scores, 95)

    binary_preds = (anomaly_scores > threshold).astype(int)
    acc = accuracy_score(test_y.numpy(), binary_preds)
    f1 = f1_score(test_y.numpy(), binary_preds)
    cm = confusion_matrix(test_y.numpy(), binary_preds)

    print(f"\n[Evaluation Results - SVDD Style]")
    print(f"  Center     : {center}")
    print(f"  Threshold  : {threshold:.4f}")
    print(f"  AUC-ROC    : {auc:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # 분포 통계
    normal_scores = anomaly_scores[test_y.numpy() == 0]
    anomaly_scores_only = anomaly_scores[test_y.numpy() == 1]
    print(f"\n  Score Distribution:")
    print(f"    Normal  - mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
    print(f"    Anomaly - mean: {anomaly_scores_only.mean():.4f}, std: {anomaly_scores_only.std():.4f}")

    return {
        "auc": auc, "accuracy": acc, "f1": f1,
        "anomaly_scores": anomaly_scores,
        "outputs": outputs,
        "threshold": threshold
    }


# ── 5. 시각화 ─────────────────────────────────────────────────────────────────
def visualize_space(model, train_x, test_x, test_y, save_dir, center=CENTER):
    """
    VQC 입력/출력 공간 시각화
    - 좌상: VQC 입력 (8D → t-SNE 2D) - 정상/이상 분포
    - 우상: VQC 출력 (1D) scatter - 정상/이상 분리 정도
    - 좌하: Train vs Test 분포 비교
    - 우하: VQC 출력 vs Anomaly Score
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        train_outputs = model(train_x.to(DEVICE)).cpu().numpy()
        test_outputs = model(test_x.to(DEVICE)).cpu().numpy()

    test_labels = test_y.numpy()
    normal_mask = test_labels == 0
    anomaly_mask = test_labels == 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. VQC 입력 공간 (8D → t-SNE 2D)
    print("[Visualize] Running t-SNE on VQC input space...")
    all_inputs = np.vstack([train_x.numpy(), test_x.numpy()])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_inputs)

    n_train = len(train_x)
    train_emb = embedded[:n_train]
    test_emb = embedded[n_train:]

    axes[0, 0].scatter(train_emb[:, 0], train_emb[:, 1],
                       c='green', alpha=0.3, s=20, label='Train (Normal)')
    axes[0, 0].scatter(test_emb[normal_mask, 0], test_emb[normal_mask, 1],
                       c='blue', alpha=0.6, s=30, label='Test Normal')
    axes[0, 0].scatter(test_emb[anomaly_mask, 0], test_emb[anomaly_mask, 1],
                       c='red', alpha=0.6, s=30, label='Test Anomaly')
    axes[0, 0].set_xlabel("t-SNE 1")
    axes[0, 0].set_ylabel("t-SNE 2")
    axes[0, 0].set_title("VQC Input Space (8D → t-SNE 2D)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. VQC 출력 scatter (1D output vs sample index)
    axes[0, 1].scatter(range(len(test_outputs[normal_mask])),
                       test_outputs[normal_mask],
                       c='blue', alpha=0.6, s=30, label='Normal')
    axes[0, 1].scatter(range(len(test_outputs[normal_mask]), len(test_outputs)),
                       test_outputs[anomaly_mask],
                       c='red', alpha=0.6, s=30, label='Anomaly')
    axes[0, 1].axhline(y=center, color='black', linestyle='--', linewidth=2, label=f'Center ({center})')
    axes[0, 1].set_xlabel("Sample Index")
    axes[0, 1].set_ylabel("VQC Output")
    axes[0, 1].set_title("VQC Output per Sample")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Train vs Test 출력 분포 비교
    axes[1, 0].scatter(range(len(train_outputs)), train_outputs,
                       c='green', alpha=0.5, s=20, label='Train')
    axes[1, 0].scatter(range(len(train_outputs), len(train_outputs) + len(test_outputs[normal_mask])),
                       test_outputs[normal_mask],
                       c='blue', alpha=0.5, s=20, label='Test Normal')
    axes[1, 0].scatter(range(len(train_outputs) + len(test_outputs[normal_mask]),
                             len(train_outputs) + len(test_outputs)),
                       test_outputs[anomaly_mask],
                       c='red', alpha=0.5, s=20, label='Test Anomaly')
    axes[1, 0].axhline(y=center, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("VQC Output")
    axes[1, 0].set_title("Train vs Test Output Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. t-SNE에 VQC 출력을 색상으로 표시
    scatter = axes[1, 1].scatter(test_emb[:, 0], test_emb[:, 1],
                                  c=test_outputs, cmap='coolwarm',
                                  alpha=0.7, s=40, edgecolors='black', linewidths=0.5)
    # 이상 샘플은 X 표시로 강조
    axes[1, 1].scatter(test_emb[anomaly_mask, 0], test_emb[anomaly_mask, 1],
                       facecolors='none', edgecolors='black', s=100, linewidths=2, marker='x')
    plt.colorbar(scatter, ax=axes[1, 1], label='VQC Output')
    axes[1, 1].set_xlabel("t-SNE 1")
    axes[1, 1].set_ylabel("t-SNE 2")
    axes[1, 1].set_title("t-SNE colored by VQC Output (X = Anomaly)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/vqc_space_visualization.png", dpi=150)
    if not args.no_show:
        plt.show()
    print(f"[Saved] {save_dir}/vqc_space_visualization.png")

    # 통계 출력
    print(f"\n[Stats] VQC Output Distribution:")
    print(f"  Train      - mean: {train_outputs.mean():.4f}, std: {train_outputs.std():.4f}")
    print(f"  Test Norm  - mean: {test_outputs[normal_mask].mean():.4f}, std: {test_outputs[normal_mask].std():.4f}")
    print(f"  Test Anom  - mean: {test_outputs[anomaly_mask].mean():.4f}, std: {test_outputs[anomaly_mask].std():.4f}")


def plot_results(history, results, test_y, save_dir, center=CENTER):
    """학습 곡선 및 결과 시각화 (SVDD 스타일)"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Training Loss (SVDD Loss)
    axes[0, 0].plot(history["train_loss"], label="SVDD Loss", color="blue")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss (MSE to Center)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Validation AUC
    axes[0, 1].plot(history["val_auc"], label="Val AUC", color="green")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AUC")
    axes[0, 1].set_title("Validation AUC")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. VQC Output Distribution (raw outputs)
    outputs = results["outputs"]
    normal_outputs = outputs[test_y.numpy() == 0]
    anomaly_outputs = outputs[test_y.numpy() == 1]

    axes[1, 0].hist(normal_outputs, bins=30, alpha=0.7, label="Normal", color="blue", density=True)
    axes[1, 0].hist(anomaly_outputs, bins=30, alpha=0.7, label="Anomaly", color="red", density=True)
    axes[1, 0].axvline(x=center, color="black", linestyle="--", linewidth=2, label=f"Center ({center})")
    axes[1, 0].set_xlabel("VQC Output")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("VQC Output Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Anomaly Score Distribution
    scores = results["anomaly_scores"]
    normal_scores = scores[test_y.numpy() == 0]
    anomaly_scores = scores[test_y.numpy() == 1]
    threshold = results["threshold"]

    axes[1, 1].hist(normal_scores, bins=30, alpha=0.7, label="Normal", color="blue", density=True)
    axes[1, 1].hist(anomaly_scores, bins=30, alpha=0.7, label="Anomaly", color="red", density=True)
    axes[1, 1].axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.3f})")
    axes[1, 1].set_xlabel("Anomaly Score (|output - center|)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Anomaly Score Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/vqc_svdd_results.png", dpi=150)
    if not args.no_show:
        plt.show()
    print(f"[Saved] {save_dir}/vqc_svdd_results.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def load_and_visualize(model_path=None):
    """저장된 모델 로드 후 시각화만 수행"""
    if model_path is None:
        model_path = f"{SAVE_DIR}/vqc_svdd_model.pt"

    print(f"[Load] Loading saved model from {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # 데이터 로딩
    train_vqc_data = np.load(f"{DATA_DIR}/train_vqc.npy")
    test_vqc_data = np.load(f"{DATA_DIR}/test_vqc.npy")
    test_labels = np.load(f"{DATA_DIR}/test_binary_labels.npy")

    train_x = torch.tensor(train_vqc_data, dtype=torch.float32)
    test_x = torch.tensor(test_vqc_data, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.float32)

    # 모델 로드
    model = VQCClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    center = checkpoint.get("center", CENTER)
    print(f"  Center: {center}")
    print(f"  Saved AUC: {checkpoint.get('auc', 'N/A')}")

    # 평가 및 시각화
    results = evaluate(model, test_x, test_y, center=center)
    visualize_space(model, train_x, test_x, test_y, SAVE_DIR, center=center)

    return model, results


if __name__ == "__main__":
    # --visualize 옵션: 저장된 모델로 시각화만
    if args.visualize:
        load_and_visualize()
        exit(0)

    print(f"Device: {DEVICE}")
    print(f"Category: {CATEGORY}")
    print(f"N_QUBITS: {N_QUBITS}, N_LAYERS: {N_LAYERS}")
    print(f"Center: {CENTER} (SVDD target)\n")

    # 1. 데이터 로딩
    print("[Load] Loading preprocessed data...")
    train_vqc_data = np.load(f"{DATA_DIR}/train_vqc.npy")
    test_vqc_data = np.load(f"{DATA_DIR}/test_vqc.npy")
    test_labels = np.load(f"{DATA_DIR}/test_binary_labels.npy")

    print(f"  train_vqc: {train_vqc_data.shape}")
    print(f"  test_vqc:  {test_vqc_data.shape}")
    print(f"  test_labels: {test_labels.shape} (normal: {(test_labels==0).sum()}, anomaly: {(test_labels==1).sum()})")

    # 2. 데이터 준비
    # Train: 정상 데이터만 (SVDD에서는 label 사용 안 함)
    train_x = torch.tensor(train_vqc_data, dtype=torch.float32)
    train_y = torch.zeros(len(train_vqc_data), dtype=torch.float32)  # placeholder

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Test: 정상 + 이상
    test_x = torch.tensor(test_vqc_data, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.float32)

    # 3. 모델 생성
    print(f"\n[Model] VQC with SVDD-style training")
    print(f"  Qubits: {N_QUBITS}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Parameters: {N_LAYERS * N_QUBITS * 2}")

    model = VQCClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS).to(DEVICE)

    # 4. 학습 (SVDD: 정상 데이터를 center로 당김)
    print(f"\n[Train] Training VQC (SVDD style)...")
    model, history = train_vqc(model, train_loader, (test_x, test_y),
                               epochs=EPOCHS, lr=LR, center=CENTER)

    # 5. 평가
    results = evaluate(model, test_x, test_y, center=CENTER)

    # 6. 시각화 및 저장
    plot_results(history, results, test_y, SAVE_DIR, center=CENTER)
    visualize_space(model, train_x, test_x, test_y, SAVE_DIR, center=CENTER)

    # 7. 모델 저장
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "center": CENTER,
        "threshold": results["threshold"],
        "auc": results["auc"]
    }, f"{SAVE_DIR}/vqc_svdd_model.pt")
    print(f"\n[Done] Model saved to {SAVE_DIR}/vqc_svdd_model.pt")
