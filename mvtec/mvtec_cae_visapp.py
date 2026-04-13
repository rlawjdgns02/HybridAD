"""
mvtec_cae_visapp.py
VISAPP 2019 (Bergmann et al.) CAE 구조 - Over-generalization 유도용

목적:
  - 이상까지 잘 복원해버리는 AE 구조
  - Reconstruction error는 낮지만, latent space에서 분리 가능한지 검증
  - MVTec 논문에서 "텍스처 클래스에서 이상도 선명하게 복원" 실패 사례로 보고된 구조

원본 구조 (VISAPP 2019 Table 1):
  - 입력: 128×128×1 (grayscale 패치)
  - Latent: 1×1×d (d=100 for texture)
  - 9-layer encoder, symmetric decoder

MVTec 적용 수정:
  - 입력: 256×256×3 (RGB, MVTec 원본 크기에 가깝게)
  - Latent: 1×1×100 (over-generalization 유도)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from PIL import Image
from pathlib import Path
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
parser.add_argument("--no-show", action="store_true", help="Don't display plots")
args, _ = parser.parse_known_args()

if args.no_show:
    matplotlib.use('Agg')

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

CATEGORY     = args.category
LATENT_DIM   = 100        # VISAPP 원본: 텍스처 100, 나노파이버 500
N_QUBITS     = 8
BATCH_SIZE   = 32
EPOCHS       = 200        # VISAPP 원본: 200 epochs
LR           = 2e-4       # VISAPP 원본: 2×10⁻⁴
WEIGHT_DECAY = 1e-5       # VISAPP 원본: 10⁻⁵
IMG_SIZE     = 256        # 256×256 (128의 2배, 구조 확장)
DATA_DIR     = "./data/mvtec_anomaly_detection"
SAVE_DIR     = "./preprocessed/mvtec_visapp"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ── 1. 데이터셋 클래스 ─────────────────────────────────────────────────────────
class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split="train", transform=None):
        self.transform = transform
        self.images = []
        self.labels = []  # 0: normal, 1: anomaly

        category_dir = Path(root_dir) / category / split
        for defect_type in category_dir.iterdir():
            if not defect_type.is_dir():
                continue
            label = 0 if defect_type.name == "good" else 1
            for img_path in defect_type.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(label)

        print(f"[Data] {split}: {len(self.images)} images "
              f"(normal: {self.labels.count(0)}, anomaly: {self.labels.count(1)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── 2. VISAPP 2019 CAE 구조 ───────────────────────────────────────────────────
class VISAPP_CAE(nn.Module):
    """
    VISAPP 2019 (Bergmann et al.) Convolutional Autoencoder

    원본 (128×128 입력):
      Conv1: 64×64×32,  Conv2: 32×32×32,  Conv3: 32×32×32
      Conv4: 16×16×64,  Conv5: 16×16×64,  Conv6: 8×8×128
      Conv7: 8×8×64,    Conv8: 8×8×32,    Conv9: 1×1×d

    256×256 입력으로 확장:
      레이어 하나 추가하여 동일한 1×1×d latent 도달
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # Encoder (256×256×3 → 1×1×latent_dim)
        self.encoder = nn.Sequential(
            # 256 → 128
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 128 → 64
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 64 → 64 (stride 1)
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 64 → 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 32 → 32 (stride 1)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 32 → 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 16 → 16 (stride 1)
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 16 → 16 (stride 1)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 16 → 1 (16×16 kernel로 완전 압축)
            nn.Conv2d(32, latent_dim, kernel_size=16, stride=1, padding=0),
            # 최종: 1×1×latent_dim (linear activation, no ReLU)
        )

        # Decoder (1×1×latent_dim → 256×256×3)
        self.decoder = nn.Sequential(
            # 1 → 16
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=16, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # 16 → 16
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 16 → 16
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 16 → 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 32 → 32
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 32 → 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 64 → 64
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 64 → 128
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 128 → 256
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            # 최종: linear activation (no sigmoid, VISAPP 원본)
        )

        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)           # (B, latent_dim, 1, 1)
        recon = self.decoder(z)       # (B, 3, 256, 256)
        return recon, z

    def get_latent_vector(self, x):
        """VQC 입력용 1D latent vector"""
        z = self.encoder(x)           # (B, latent_dim, 1, 1)
        return z.flatten(start_dim=1) # (B, latent_dim)


# ── 3. 학습 ───────────────────────────────────────────────────────────────────
def train_cae(model, train_dataset, epochs=EPOCHS, lr=LR):
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, persistent_workers=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

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

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss: {avg_loss:.6f}")

    return model, loss_history


# ── 4. Latent 추출 ────────────────────────────────────────────────────────────
def extract_latents(model, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        num_workers=4, persistent_workers=True)
    model.eval()
    latents, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            z = model.get_latent_vector(x.to(DEVICE))
            latents.append(z.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))

    return np.concatenate(latents), np.concatenate(labels)


# ── 5. Reconstruction Error 계산 ──────────────────────────────────────────────
def compute_recon_errors(model, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    errors, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            mse = ((x - recon) ** 2).mean(dim=(1, 2, 3))
            errors.extend(mse.cpu().numpy())
            labels.extend(y if isinstance(y, list) else y.tolist())

    return np.array(errors), np.array(labels)


# ── 6. VQC 전처리 ─────────────────────────────────────────────────────────────
def preprocess_for_vqc(train_latents, test_latents, n_qubits=N_QUBITS):
    pca = PCA(n_components=n_qubits)
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    train_reduced = pca.fit_transform(train_latents)
    test_reduced = pca.transform(test_latents)

    train_vqc = scaler.fit_transform(train_reduced)
    test_vqc = scaler.transform(test_reduced)

    print(f"[VQC] PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"[VQC] Output shape → train: {train_vqc.shape}, test: {test_vqc.shape}")

    return train_vqc, test_vqc, pca, scaler


# ── 7. Latent Space 시각화 ────────────────────────────────────────────────────
def visualize_latent_space(test_latents, test_labels, save_dir):
    """PCA + t-SNE로 latent space 분포 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(test_latents)

    axes[0].scatter(pca_out[test_labels == 0, 0], pca_out[test_labels == 0, 1],
                    s=20, alpha=0.6, label="정상", color="blue")
    axes[0].scatter(pca_out[test_labels == 1, 0], pca_out[test_labels == 1, 1],
                    s=20, alpha=0.6, label="이상", color="red")
    axes[0].set_title(f"PCA: Latent Space ({CATEGORY})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    print("[t-SNE] 계산 중...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_labels)-1))
    tsne_out = tsne.fit_transform(test_latents)

    axes[1].scatter(tsne_out[test_labels == 0, 0], tsne_out[test_labels == 0, 1],
                    s=20, alpha=0.6, label="정상", color="blue")
    axes[1].scatter(tsne_out[test_labels == 1, 0], tsne_out[test_labels == 1, 1],
                    s=20, alpha=0.6, label="이상", color="red")
    axes[1].set_title(f"t-SNE: Latent Space ({CATEGORY})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"VISAPP CAE Latent Space - {CATEGORY} (dim={LATENT_DIM})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_space.png", dpi=150)
    if not args.no_show:
        plt.show()
    plt.close()
    print(f"[Saved] {save_dir}/latent_space.png")


# ── 8. 복원 시각화 ────────────────────────────────────────────────────────────
def visualize_results(model, test_dataset, test_labels, errors, save_dir):
    """복원 이미지 + Reconstruction error 분포 시각화"""

    model.eval()
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))

    # 상단: 정상 원본/복원
    normal_idx = (test_labels == 0).nonzero()[0][:3]
    # 중단: 이상 원본/복원
    anomaly_idx = (test_labels == 1).nonzero()[0][:3]

    def denormalize(x):
        # 단순 [0,1] clamp (정규화 안 했으므로)
        return x.clamp(0, 1)

    for col, idx in enumerate(normal_idx):
        x, _ = test_dataset[idx]
        x_in = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon, _ = model(x_in)

        axes[0, col].imshow(denormalize(x).permute(1,2,0).numpy())
        axes[0, col].set_title(f"Normal #{idx}")
        axes[0, col].axis("off")

        axes[0, col+3].imshow(denormalize(recon.squeeze().cpu()).permute(1,2,0).numpy())
        axes[0, col+3].set_title(f"Recon (err={errors[idx]:.4f})")
        axes[0, col+3].axis("off")

    for col, idx in enumerate(anomaly_idx):
        x, _ = test_dataset[idx]
        x_in = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon, _ = model(x_in)

        axes[1, col].imshow(denormalize(x).permute(1,2,0).numpy())
        axes[1, col].set_title(f"Anomaly #{idx}")
        axes[1, col].axis("off")

        axes[1, col+3].imshow(denormalize(recon.squeeze().cpu()).permute(1,2,0).numpy())
        axes[1, col+3].set_title(f"Recon (err={errors[idx]:.4f})")
        axes[1, col+3].axis("off")

    # 하단: Error 분포
    normal_errors = errors[test_labels == 0]
    anomaly_errors = errors[test_labels == 1]
    auc = roc_auc_score(test_labels, errors)

    for ax in axes[2, :3]:
        ax.axis("off")

    ax_hist = fig.add_subplot(3, 2, 5)
    ax_hist.hist(normal_errors, bins=20, alpha=0.7, label='Normal', color='blue')
    ax_hist.hist(anomaly_errors, bins=20, alpha=0.7, label='Anomaly', color='red')
    ax_hist.set_xlabel('Reconstruction Error')
    ax_hist.set_title(f'Error Distribution (AUC={auc:.4f})')
    ax_hist.legend()

    axes[2, 3].axis("off")
    axes[2, 4].axis("off")
    axes[2, 5].axis("off")

    plt.suptitle(f"VISAPP CAE Results - {CATEGORY}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/visapp_cae_results.png", dpi=150)
    if not args.no_show:
        plt.show()
    plt.close()

    print(f"[Stats] Reconstruction Error AUC: {auc:.4f}")
    print(f"  Normal  - mean: {normal_errors.mean():.6f}, std: {normal_errors.std():.6f}")
    print(f"  Anomaly - mean: {anomaly_errors.mean():.6f}, std: {anomaly_errors.std():.6f}")

    return auc


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Category: {CATEGORY}")
    print(f"Model: VISAPP 2019 CAE (over-generalization)")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"Image size: {IMG_SIZE}×{IMG_SIZE}\n")

    # 저장 디렉토리
    save_dir = f"{SAVE_DIR}/{CATEGORY}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 데이터 로딩
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # VISAPP 원본: 정규화 없음 (pixel값 그대로)
    ])

    train_dataset = MVTecDataset(DATA_DIR, CATEGORY, split="train", transform=transform)
    test_dataset = MVTecDataset(DATA_DIR, CATEGORY, split="test", transform=transform)
    test_binary_labels = torch.tensor([label for _, label in test_dataset])

    # 2. 모델 학습
    print(f"\n[Train] VISAPP CAE ({EPOCHS} epochs)...")
    model = VISAPP_CAE(latent_dim=LATENT_DIM).to(DEVICE)
    print(f"[Model] Latent: 1×1×{LATENT_DIM} = {LATENT_DIM} dim")
    model, loss_history = train_cae(model, train_dataset)

    # 3. Reconstruction error 계산
    print("\n[Eval] Computing reconstruction errors...")
    errors, labels = compute_recon_errors(model, test_dataset)

    # 4. 시각화
    print("\n[Viz] Visualizing results...")
    recon_auc = visualize_results(model, test_dataset, labels, errors, save_dir)

    # 5. Latent 추출
    print("\n[Extract] Latent vectors...")
    train_latents, _ = extract_latents(model, train_dataset)
    test_latents, _ = extract_latents(model, test_dataset)

    # 6. Latent space 시각화
    print("\n[Viz] Latent space distribution...")
    visualize_latent_space(test_latents, labels, save_dir)

    # 7. VQC 전처리
    print("\n[VQC] Preprocessing for quantum circuit...")
    train_vqc, test_vqc, pca, scaler = preprocess_for_vqc(train_latents, test_latents)

    # 7. 저장
    np.save(f"{save_dir}/train_latents.npy", train_latents)
    np.save(f"{save_dir}/test_latents.npy", test_latents)
    np.save(f"{save_dir}/test_binary_labels.npy", labels)
    np.save(f"{save_dir}/train_vqc.npy", train_vqc)
    np.save(f"{save_dir}/test_vqc.npy", test_vqc)
    np.save(f"{save_dir}/recon_errors.npy", errors)
    torch.save(model.state_dict(), f"{save_dir}/visapp_cae.pt")

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"VISAPP CAE Training - {CATEGORY}")
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    print(f"\n[Done] Saved to {save_dir}/")
    print(f"  train_latents.npy : {train_latents.shape}")
    print(f"  test_latents.npy  : {test_latents.shape}")
    print(f"  train_vqc.npy     : {train_vqc.shape}")
    print(f"  test_vqc.npy      : {test_vqc.shape}")
    print(f"  recon_errors.npy  : {errors.shape}")
    print(f"\n[Key Result] Recon Error AUC: {recon_auc:.4f}")
    if recon_auc < 0.65:
        print("  → Over-generalization 성공! Recon error로 탐지 어려움")
        print("  → Latent space에서 VQC 분류 가능성 테스트 필요")
