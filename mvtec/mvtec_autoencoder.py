"""
mvtec_autoencoder.py
MVTec AD 전처리 파이프라인
  1. 폴더 구조 기반 라벨링 (train: 정상만, test: 정상+이상)
  2. Pretrained Backbone(ResNet18) 기반 Autoencoder 학습 (정상 데이터만)
  3. Latent vector 추출
  4. 학습 검증 (Loss 곡선 + 복원 이미지)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
CATEGORY     = "bottle"  # 변경 가능: bottle, cable, capsule, etc.
LATENT_DIM   = 256       # bottleneck 채널 수 (latent shape: 7x7xLATENT_DIM)
N_QUBITS     = 8
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-4
IMG_SIZE     = 224  # ResNet 입력 크기
DATA_DIR     = "./data/mvtec_anomaly_detection"
SAVE_DIR     = "./preprocessed/mvtec"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# "mse" | "ssim" | "perceptual" | "ssim+mse"
LOSS_TYPE    = "ssim+mse"


# ── Loss Functions ────────────────────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """구조적 유사도 기반 loss. 엣지/텍스처 보존에 유리."""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._gaussian_window(window_size))

    def _gaussian_window(self, size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)

    def _ssim(self, x, y):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w = self.window.expand(x.size(1), 1, -1, -1)

        mu_x  = nn.functional.conv2d(x, w, padding=self.window_size//2, groups=x.size(1))
        mu_y  = nn.functional.conv2d(y, w, padding=self.window_size//2, groups=x.size(1))
        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_xx = nn.functional.conv2d(x*x, w, padding=self.window_size//2, groups=x.size(1)) - mu_xx
        sig_yy = nn.functional.conv2d(y*y, w, padding=self.window_size//2, groups=x.size(1)) - mu_yy
        sig_xy = nn.functional.conv2d(x*y, w, padding=self.window_size//2, groups=x.size(1)) - mu_xy

        num = (2*mu_xy + C1) * (2*sig_xy + C2)
        den = (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2)
        return (num / den).mean()

    def forward(self, recon, target):
        return 1 - self._ssim(recon, target)


class PerceptualLoss(nn.Module):
    """VGG16 중간 feature 비교. 픽셀이 아닌 '의미적 유사도'로 학습."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # relu2_2(idx 9), relu3_3(idx 16) feature 사용
        self.slice1 = nn.Sequential(*list(vgg.features)[:10]).eval()
        self.slice2 = nn.Sequential(*list(vgg.features)[10:17]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, recon, target):
        f1_r, f1_t = self.slice1(recon), self.slice1(target)
        f2_r, f2_t = self.slice2(f1_r),  self.slice2(f1_t)
        return nn.functional.mse_loss(f1_r, f1_t) + nn.functional.mse_loss(f2_r, f2_t)


def build_loss_fn(loss_type=LOSS_TYPE):
    """LOSS_TYPE에 따라 loss 함수 반환."""
    mse  = nn.MSELoss()
    ssim = SSIMLoss().to(DEVICE)
    perc = PerceptualLoss().to(DEVICE) if "perceptual" in loss_type else None

    if loss_type == "mse":
        return lambda r, t: mse(r, t)
    elif loss_type == "ssim":
        return lambda r, t: ssim(r, t)
    elif loss_type == "ssim+mse":
        return lambda r, t: 0.8 * ssim(r, t) + 0.2 * mse(r, t)
    elif loss_type == "perceptual":
        return lambda r, t: perc(r, t)
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {loss_type}")


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
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def load_mvtec_dataset(category=CATEGORY):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    train_dataset = MVTecDataset(DATA_DIR, category, split="train", transform=transform)
    test_dataset  = MVTecDataset(DATA_DIR, category, split="test",  transform=transform)

    test_labels = torch.tensor([label for _, label in test_dataset])

    return train_dataset, test_dataset, test_labels


# ── 2. 모델 (Pretrained Backbone 기반 Autoencoder) ────────────────────────────
class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_channels=LATENT_DIM):
        super().__init__()

        # Encoder: Pretrained ResNet18 backbone (AvgPool 제외) + Conv bottleneck
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]  # avgpool, fc 제외 → 7x7x512 출력
        )

        # Bottleneck: 채널 수 줄이기 (512 → latent_channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_channels, 1),  # 1x1 conv로 채널 축소
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
        )

        # Decoder: 7x7에서 시작하여 224x224로 업샘플링
        # Upsample+Conv 방식 → ConvTranspose2d의 체커보드 아티팩트 방지
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

        self.decoder = nn.Sequential(
            up_block(latent_channels, 256),  # 14x14
            up_block(256, 128),              # 28x28
            up_block(128, 64),               # 56x56
            up_block(64, 32),                # 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),  # 224x224
        )

    def _freeze_backbone(self, freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        # Encoder
        feat = self.backbone(x)      # 7x7x512
        z = self.bottleneck(feat)    # 7x7xlatent_channels (공간 정보 유지!)

        # Decoder
        recon = self.decoder(z)      # 224x224x3

        return recon, z

    def get_latent_vector(self, x):
        """VQC 입력용 1D latent vector 추출"""
        feat = self.backbone(x)
        z = self.bottleneck(feat)    # 7x7xC
        return z.flatten(start_dim=1)  # (B, 7*7*C)


# ── 3. 학습 ───────────────────────────────────────────────────────────────────
def train_autoencoder(model, train_dataset, epochs=EPOCHS, lr=LR):
    loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    criterion = build_loss_fn(LOSS_TYPE)
    loss_history = []
    print(f"[Loss] Using: {LOSS_TYPE}")

    # 2단계 학습: backbone freeze → unfreeze
    model._freeze_backbone(freeze=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    model.train()
    for epoch in range(1, epochs + 1):
        # epoch 절반 지나면 backbone unfreeze
        if epoch == epochs // 2:
            print("[Model] Unfreezing backbone for fine-tuning...")
            model._freeze_backbone(freeze=False)
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        total = 0.0
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        scheduler.step()
        avg_loss = total / len(loader)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch:>2}/{epochs}  loss: {avg_loss:.5f}  lr: {scheduler.get_last_lr()[0]:.6f}")

    return model, loss_history


# ── 4. Latent 추출 ────────────────────────────────────────────────────────────
def extract_latents(model, dataset):
    """VQC 입력용 1D latent vector 추출 (7x7xC → flatten)"""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    model.eval()
    latents, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            z = model.get_latent_vector(x.to(DEVICE))  # (B, 7*7*C)
            latents.append(z.cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))

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
def plot_loss_curve(loss_history, save_dir):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"MVTec Autoencoder Training Loss ({CATEGORY})")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()
    print(f"[Check] Loss curve saved.")


def visualize_reconstruction(model, test_dataset, test_labels, save_dir, n=6):
    model.eval()

    # ImageNet 역정규화
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def denormalize(x):
        return (x.cpu() * std + mean).clamp(0, 1)

    normal_idx  = (test_labels == 0).nonzero(as_tuple=True)[0][:n]
    anomaly_idx = (test_labels == 1).nonzero(as_tuple=True)[0][:n]

    fig, axes = plt.subplots(4, n, figsize=(n * 2, 8))
    titles = ["Normal Original", "Normal Recon", "Anomaly Original", "Anomaly Recon"]

    for row, indices in enumerate([normal_idx, normal_idx, anomaly_idx, anomaly_idx]):
        for col, idx in enumerate(indices):
            x, _ = test_dataset[idx.item()]
            x_in = x.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                recon, _ = model(x_in)

            if row % 2 == 1:
                img = denormalize(recon.squeeze())
            else:
                img = denormalize(x)

            axes[row, col].imshow(img.permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(titles[row], fontsize=9)

    plt.suptitle(f"Reconstruction Visualization ({CATEGORY})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction.png")
    plt.close()
    print(f"[Check] Reconstruction image saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Category: {CATEGORY}\n")

    # 저장 디렉토리
    save_dir = f"{SAVE_DIR}/{CATEGORY}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 데이터 로딩
    train_dataset, test_dataset, test_binary_labels = load_mvtec_dataset(category=CATEGORY)

    # 2. 모델 학습
    print("\n[Train] ResNet Autoencoder (7x7 spatial latent) ...")
    model = ResNetAutoencoder(latent_channels=LATENT_DIM).to(DEVICE)
    print(f"[Model] Latent shape: 7x7x{LATENT_DIM} = {7*7*LATENT_DIM} dim")
    model, loss_history = train_autoencoder(model, train_dataset)

    # 3. 학습 검증
    print("\n[Check] Validating training ...")
    plot_loss_curve(loss_history, save_dir)
    visualize_reconstruction(model, test_dataset, test_binary_labels, save_dir)

    # 4. Latent 추출
    print("\n[Extract] Latent vectors ...")
    train_latents, _ = extract_latents(model, train_dataset)
    test_latents, _  = extract_latents(model, test_dataset)

    # 5. VQC 전처리
    print("\n[VQC] Preprocessing for quantum circuit ...")
    train_vqc, test_vqc, pca, scaler = preprocess_for_vqc(train_latents, test_latents)

    # 6. 저장
    np.save(f"{save_dir}/train_latents.npy",       train_latents)
    np.save(f"{save_dir}/test_latents.npy",        test_latents)
    np.save(f"{save_dir}/test_binary_labels.npy",  test_binary_labels.numpy())
    np.save(f"{save_dir}/train_vqc.npy",           train_vqc)
    np.save(f"{save_dir}/test_vqc.npy",            test_vqc)
    torch.save(model.state_dict(), f"{save_dir}/autoencoder.pt")

    print(f"\n[Done] Saved to {save_dir}/")
    print(f"  train_latents.npy      : {train_latents.shape}")
    print(f"  test_latents.npy       : {test_latents.shape}")
    print(f"  test_binary_labels.npy : {test_binary_labels.shape}")
    print(f"  train_vqc.npy          : {train_vqc.shape}")
    print(f"  test_vqc.npy           : {test_vqc.shape}")
