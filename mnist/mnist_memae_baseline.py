"""
mnist_memae_baseline.py
MemAE 논문의 Baseline AE (Over-generalization 모델)

목적:
  - 이상치까지 너무 잘 복원해버리는 "문제있는" AE
  - Reconstruction error로는 탐지 어려움
  - 하지만 latent space에서 VQC로 분류 가능한지 검증

구조 (MemAE 논문 Section 4.1):
  Encoder: 4 Conv layers (16→32→64→128), kernel 3×3, stride 2
  Decoder: 4 Deconv layers (대칭)
  Activation: BN + LeakyReLU
  Output: Sigmoid

학습 (MemAE 논문 Section 4.2):
  Optimizer: Adam, lr=1e-4
  Loss: MSE
  Batch: 128
  Epochs: 200
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--normal-digit", type=int, default=0, help="Normal digit (0-9)")
parser.add_argument("--n-shot", type=int, default=20, help="Few-shot: number of training samples (0=all)")
parser.add_argument("--no-show", action="store_true", help="Don't display plots")
args, _ = parser.parse_known_args()

if args.no_show:
    matplotlib.use('Agg')

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

NORMAL_DIGIT = args.normal_digit
N_SHOT       = args.n_shot         # 0 = use all, >0 = few-shot
LATENT_DIM   = 128        # 2×2×128 = 512 dim latent
N_QUBITS     = 8
BATCH_SIZE   = min(128, N_SHOT) if N_SHOT > 0 else 128
EPOCHS       = 200        # MemAE 원본
LR           = 1e-4       # MemAE 원본
IMG_SIZE     = 28
SAVE_DIR     = f"./preprocessed/mnist_memae/{NORMAL_DIGIT}_shot{N_SHOT}" if N_SHOT > 0 else f"./preprocessed/mnist_memae/{NORMAL_DIGIT}"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ── 1. MemAE Baseline AE ─────────────────────────────────────────────────────
class MemAE_Baseline(nn.Module):
    """
    MemAE 논문의 Baseline Autoencoder (Over-generalization 모델)

    Encoder: 28×28×1 → 14×14×16 → 7×7×32 → 4×4×64 → 2×2×128
    Decoder: 대칭 구조
    """
    def __init__(self, latent_channels=128):
        super().__init__()

        # Encoder: 4 Conv layers with stride 2
        self.encoder = nn.Sequential(
            # 28×28×1 → 14×14×16
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # 14×14×16 → 7×7×32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 7×7×32 → 4×4×64 (padding=1로 하면 4×4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 4×4×64 → 2×2×128
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: 4 Deconv layers (symmetric)
        self.decoder = nn.Sequential(
            # 2×2×128 → 4×4×64
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 4×4×64 → 7×7×32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 7×7×32 → 14×14×16
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # 14×14×16 → 28×28×1
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output [0, 1]
        )

        self.latent_channels = latent_channels

    def forward(self, x):
        z = self.encoder(x)        # (B, 128, 2, 2)
        recon = self.decoder(z)    # (B, 1, 28, 28)
        return recon, z

    def get_latent_vector(self, x):
        z = self.encoder(x)
        return z.flatten(start_dim=1)  # (B, 128*2*2) = (B, 512)


# ── 2. 데이터 준비 ───────────────────────────────────────────────────────────
def prepare_data(normal_digit=NORMAL_DIGIT, n_shot=N_SHOT):
    """MNIST에서 정상(하나의 숫자) vs 이상(나머지) 분리"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_full = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Train: 정상 데이터만 (normal_digit)
    train_idx = [i for i, (_, label) in enumerate(train_full) if label == normal_digit]

    # Few-shot: 랜덤 샘플링
    if n_shot > 0 and n_shot < len(train_idx):
        np.random.seed(42)  # 재현성
        train_idx = list(np.random.choice(train_idx, n_shot, replace=False))

    train_dataset = Subset(train_full, train_idx)

    # Test: 모든 데이터 (정상 + 이상)
    test_dataset = test_full

    # Binary labels: 0=normal, 1=anomaly
    test_labels = np.array([0 if label == normal_digit else 1 for _, label in test_full])

    print(f"[Data] Normal digit: {normal_digit}")
    print(f"[Data] Few-shot: {n_shot if n_shot > 0 else 'ALL'}")
    print(f"[Data] Train: {len(train_dataset)} (all normal)")
    print(f"[Data] Test: {len(test_dataset)} (normal: {(test_labels==0).sum()}, anomaly: {(test_labels==1).sum()})")

    return train_dataset, test_dataset, test_labels


# ── 3. 학습 ──────────────────────────────────────────────────────────────────
def train_ae(model, train_dataset, epochs=EPOCHS, lr=LR):
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss: {avg_loss:.6f}")

    return model, loss_history


# ── 4. Latent 추출 ───────────────────────────────────────────────────────────
def extract_latents(model, dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    latents, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            z = model.get_latent_vector(x.to(DEVICE))
            latents.append(z.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))

    return np.concatenate(latents), np.concatenate(labels)


# ── 5. Reconstruction Error ──────────────────────────────────────────────────
def compute_recon_errors(model, dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    errors = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            mse = ((x - recon) ** 2).mean(dim=(1, 2, 3))
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


# ── 6. VQC 전처리 ────────────────────────────────────────────────────────────
def preprocess_for_vqc(train_latents, test_latents, n_qubits=N_QUBITS):
    pca = PCA(n_components=n_qubits)
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    train_reduced = pca.fit_transform(train_latents)
    test_reduced = pca.transform(test_latents)

    train_vqc = scaler.fit_transform(train_reduced)
    test_vqc = scaler.transform(test_reduced)

    print(f"[VQC] PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"[VQC] Output: train {train_vqc.shape}, test {test_vqc.shape}")

    return train_vqc, test_vqc, pca, scaler


# ── 7. 시각화 ────────────────────────────────────────────────────────────────
def visualize_all(model, test_dataset, test_labels, errors, test_latents, save_dir):
    """복원 + Error 분포 + Latent space 시각화"""
    model.eval()

    fig = plt.figure(figsize=(16, 12))

    # 1. 복원 이미지 (상단)
    ax1 = fig.add_subplot(3, 4, 1)
    ax2 = fig.add_subplot(3, 4, 2)
    ax3 = fig.add_subplot(3, 4, 3)
    ax4 = fig.add_subplot(3, 4, 4)
    ax5 = fig.add_subplot(3, 4, 5)
    ax6 = fig.add_subplot(3, 4, 6)
    ax7 = fig.add_subplot(3, 4, 7)
    ax8 = fig.add_subplot(3, 4, 8)

    # 정상 샘플 (2개)
    normal_indices = np.where(test_labels == 0)[0][:2]
    # 이상 샘플 (2개)
    anomaly_indices = np.where(test_labels == 1)[0][:2]

    axes_orig = [ax1, ax3, ax5, ax7]
    axes_recon = [ax2, ax4, ax6, ax8]
    indices = list(normal_indices) + list(anomaly_indices)
    titles = ["Normal", "Normal", "Anomaly", "Anomaly"]

    for i, (idx, ax_o, ax_r, title) in enumerate(zip(indices, axes_orig, axes_recon, titles)):
        x, _ = test_dataset[idx]
        x_in = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon, _ = model(x_in)

        ax_o.imshow(x.squeeze().numpy(), cmap='gray')
        ax_o.set_title(f"{title} #{idx}")
        ax_o.axis('off')

        ax_r.imshow(recon.squeeze().cpu().numpy(), cmap='gray')
        ax_r.set_title(f"Recon (err={errors[idx]:.4f})")
        ax_r.axis('off')

    # 2. Error 분포 (좌하단)
    ax_hist = fig.add_subplot(3, 2, 5)
    normal_errors = errors[test_labels == 0]
    anomaly_errors = errors[test_labels == 1]
    auc = roc_auc_score(test_labels, errors)

    ax_hist.hist(normal_errors, bins=50, alpha=0.7, label=f'Normal (n={len(normal_errors)})', color='blue')
    ax_hist.hist(anomaly_errors, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_errors)})', color='red')
    ax_hist.set_xlabel('Reconstruction Error')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title(f'Error Distribution (AUC={auc:.4f})')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # 3. t-SNE Latent space (우하단)
    ax_tsne = fig.add_subplot(3, 2, 6)
    print("[t-SNE] Computing...")

    # 샘플링 (너무 많으면 느림)
    n_samples = min(2000, len(test_latents))
    idx_sample = np.random.choice(len(test_latents), n_samples, replace=False)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_out = tsne.fit_transform(test_latents[idx_sample])
    labels_sample = test_labels[idx_sample]

    ax_tsne.scatter(tsne_out[labels_sample == 0, 0], tsne_out[labels_sample == 0, 1],
                    s=10, alpha=0.5, label="Normal", color="blue")
    ax_tsne.scatter(tsne_out[labels_sample == 1, 0], tsne_out[labels_sample == 1, 1],
                    s=10, alpha=0.5, label="Anomaly", color="red")
    ax_tsne.set_title(f"t-SNE Latent Space (n={n_samples})")
    ax_tsne.legend()
    ax_tsne.grid(True, alpha=0.3)

    shot_str = f"{N_SHOT}-shot" if N_SHOT > 0 else "Full"
    plt.suptitle(f"MemAE Baseline AE - MNIST (Normal={NORMAL_DIGIT}, {shot_str})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/memae_baseline_results.png", dpi=150)
    if not args.no_show:
        plt.show()
    plt.close()

    print(f"\n[Stats] Reconstruction Error:")
    print(f"  Normal  - mean: {normal_errors.mean():.6f}, std: {normal_errors.std():.6f}")
    print(f"  Anomaly - mean: {anomaly_errors.mean():.6f}, std: {anomaly_errors.std():.6f}")
    print(f"  AUC: {auc:.4f}")

    return auc


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Normal digit: {NORMAL_DIGIT}")
    print(f"Few-shot: {N_SHOT if N_SHOT > 0 else 'ALL'}")
    print(f"Model: MemAE Baseline (Over-generalization)")
    print(f"Latent: 2×2×{LATENT_DIM} = {4*LATENT_DIM} dim\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 데이터 준비
    train_dataset, test_dataset, test_labels = prepare_data(NORMAL_DIGIT)

    # 2. 모델 학습
    print(f"\n[Train] MemAE Baseline ({EPOCHS} epochs)...")
    model = MemAE_Baseline(latent_channels=LATENT_DIM).to(DEVICE)
    model, loss_history = train_ae(model, train_dataset)

    # 3. Reconstruction error
    print("\n[Eval] Computing reconstruction errors...")
    errors = compute_recon_errors(model, test_dataset)

    # 4. Latent 추출
    print("\n[Extract] Latent vectors...")
    train_latents, _ = extract_latents(model, train_dataset)
    test_latents, _ = extract_latents(model, test_dataset)

    # 5. 시각화
    print("\n[Viz] Visualizing results...")
    recon_auc = visualize_all(model, test_dataset, test_labels, errors, test_latents, SAVE_DIR)

    # 6. VQC 전처리
    print("\n[VQC] Preprocessing...")
    train_vqc, test_vqc, pca, scaler = preprocess_for_vqc(train_latents, test_latents)

    # 7. 저장
    np.save(f"{SAVE_DIR}/train_latents.npy", train_latents)
    np.save(f"{SAVE_DIR}/test_latents.npy", test_latents)
    np.save(f"{SAVE_DIR}/test_binary_labels.npy", test_labels)
    np.save(f"{SAVE_DIR}/train_vqc.npy", train_vqc)
    np.save(f"{SAVE_DIR}/test_vqc.npy", test_vqc)
    np.save(f"{SAVE_DIR}/recon_errors.npy", errors)
    torch.save(model.state_dict(), f"{SAVE_DIR}/memae_baseline.pt")

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    shot_str = f"{N_SHOT}-shot" if N_SHOT > 0 else "Full"
    plt.title(f"MemAE Baseline Training - MNIST (Normal={NORMAL_DIGIT}, {shot_str})")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/loss_curve.png")
    plt.close()

    print(f"\n[Done] Saved to {SAVE_DIR}/")
    print(f"  train_latents.npy : {train_latents.shape}")
    print(f"  test_latents.npy  : {test_latents.shape}")
    print(f"  train_vqc.npy     : {train_vqc.shape}")
    print(f"  test_vqc.npy      : {test_vqc.shape}")

    print(f"\n[Key Result] Recon Error AUC: {recon_auc:.4f}")
    if recon_auc < 0.7:
        print("  → Over-generalization! Recon error로 탐지 어려움")
        print("  → Latent space에서 VQC 분류 가능성 테스트 필요")
    else:
        print("  → Recon error가 정상/이상을 어느정도 분리함")
