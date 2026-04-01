"""
dimension.py
MNIST / Fashion-MNIST 전처리 파이프라인
  1. 정상/이상 클래스 분리
  2. Scratch CNN Autoencoder 학습 (정상 데이터만)
  3. Latent vector 추출
  4. 학습 검증 (Loss 곡선 + 복원 이미지)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# ── Config ────────────────────────────────────────────────────────────────────
NORMAL_CLASS = 0
LATENT_DIM   = 64
N_QUBITS     = 8
BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 1e-3
DATA_DIR     = "./data"
SAVE_DIR     = "./preprocessed"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────
def load_anomaly_dataset(normal_class=NORMAL_CLASS, dataset="mnist"):
    transform = transforms.ToTensor()

    DatasetCls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
    train_full = DatasetCls(DATA_DIR, train=True,  download=True, transform=transform)
    test_full  = DatasetCls(DATA_DIR, train=False, download=True, transform=transform)

    normal_idx    = [i for i, (_, y) in enumerate(train_full) if y == normal_class]
    train_dataset = Subset(train_full, normal_idx)

    test_labels = torch.tensor(
        [0 if y == normal_class else 1 for _, y in test_full]
    )

    print(f"[Data] Train (normal=class {normal_class}): {len(train_dataset):,} samples")
    print(f"[Data] Test  (all classes):               {len(test_full):,} samples")
    print(f"[Data]   Normal: {(test_labels==0).sum():,}  Anomaly: {(test_labels==1).sum():,}")

    return train_dataset, test_full, test_labels


# ── 2. 모델 ───────────────────────────────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ── 3. 학습 ───────────────────────────────────────────────────────────────────
def train_autoencoder(model, train_dataset, epochs=EPOCHS, lr=LR):
    loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg_loss = total / len(loader)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch:>2}/{epochs}  loss: {avg_loss:.5f}")

    return model, loss_history


# ── 4. Latent 추출 ────────────────────────────────────────────────────────────
def extract_latents(model, dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, z = model(x.to(DEVICE))
            latents.append(z.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(latents), np.concatenate(labels)


# ── 5. VQC 전처리 (PCA + 정규화) ─────────────────────────────────────────────
def preprocess_for_vqc(train_latents, test_latents, n_qubits=N_QUBITS):
    pca    = PCA(n_components=n_qubits)
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    train_reduced = pca.fit_transform(train_latents)
    test_reduced  = pca.transform(test_latents)

    train_vqc = scaler.fit_transform(train_reduced)
    test_vqc  = scaler.transform(test_reduced)

    print(f"[VQC] PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"[VQC] Output shape → train: {train_vqc.shape}, test: {test_vqc.shape}")

    return train_vqc, test_vqc, pca, scaler


# ── 6. 검증 ───────────────────────────────────────────────────────────────────
def plot_loss_curve(loss_history):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/loss_curve.png")
    plt.show()
    print(f"[Check] Loss curve saved.")


def visualize_reconstruction(model, test_dataset, test_labels, n=10):
    model.eval()

    normal_idx  = (test_labels == 0).nonzero(as_tuple=True)[0][:n]
    anomaly_idx = (test_labels == 1).nonzero(as_tuple=True)[0][:n]

    fig, axes = plt.subplots(4, n, figsize=(n * 1.5, 6))
    titles = ["정상 원본", "정상 복원", "이상 원본", "이상 복원"]

    for row, indices in enumerate([normal_idx, normal_idx, anomaly_idx, anomaly_idx]):
        for col, idx in enumerate(indices):
            x, _ = test_dataset[idx.item()]
            x_in = x.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                recon, _ = model(x_in)
            img = recon.squeeze().cpu() if row % 2 == 1 else x.squeeze()
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(titles[row], fontsize=9)

    plt.suptitle("Reconstruction Visualization", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/reconstruction.png")
    plt.show()
    print(f"[Check] Reconstruction image saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    # 1. 데이터 로딩
    train_dataset, test_dataset, test_binary_labels = load_anomaly_dataset(
        normal_class=NORMAL_CLASS, dataset="mnist"
    )

    # 2. 모델 학습
    print("\n[Train] Autoencoder ...")
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    model, loss_history = train_autoencoder(model, train_dataset)

    # 3. 학습 검증
    print("\n[Check] Validating training ...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_loss_curve(loss_history)
    visualize_reconstruction(model, test_dataset, test_binary_labels)

    # 4. Latent 추출
    print("\n[Extract] Latent vectors ...")
    train_latents, _ = extract_latents(model, train_dataset)
    test_latents,  _ = extract_latents(model, test_dataset)

    # 5. 저장
    np.save(f"{SAVE_DIR}/train_latents.npy",       train_latents)
    np.save(f"{SAVE_DIR}/test_latents.npy",        test_latents)
    np.save(f"{SAVE_DIR}/test_binary_labels.npy",  test_binary_labels.numpy())
    torch.save(model.state_dict(), f"{SAVE_DIR}/autoencoder.pt")

    print(f"\n[Done] Saved to {SAVE_DIR}/")
    print(f"  train_latents.npy      : {train_latents.shape}")
    print(f"  test_latents.npy       : {test_latents.shape}")
    print(f"  test_binary_labels.npy : {test_binary_labels.shape}")