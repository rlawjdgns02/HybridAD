"""
analyze_all_categories.py
MVTec 결과 분석 - Over-generalization vs 복원실패 판정

지표:
  1. E_train_good: train 정상 데이터 복원 에러
  2. E_test_good: test 정상 데이터 복원 에러
  3. E_test_anomaly: test 이상 데이터 복원 에러
  4. gap = E_test_good / E_train_good

판정:
  A (Over-gen): E_train_good 낮음, gap ≈ 1, E_test_anomaly도 낮음 → recon으로 탐지 불가
  B (복원실패): E_train_good 높거나 분산 큼 → 모델 자체가 학습 안됨
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/mvtec_anomaly_detection"
RESULT_DIR = "./preprocessed/mvtec"
IMG_SIZE = 224  # ResNet 입력 크기
LATENT_DIM = 16  # ResNetAutoencoder의 latent_channels

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ── 데이터셋 클래스 ────────────────────────────────────────────────────────────
class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split="train", transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        category_dir = Path(root_dir) / category / split
        for defect_type in category_dir.iterdir():
            if not defect_type.is_dir():
                continue
            label = 0 if defect_type.name == "good" else 1
            for img_path in defect_type.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── ResNet Autoencoder 모델 ────────────────────────────────────────────────────
class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_channels=LATENT_DIM):
        super().__init__()

        # Encoder: Pretrained ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]  # avgpool, fc 제외 → 7x7x512 출력
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_channels, 1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
        )

        # Decoder
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

        self.decoder = nn.Sequential(
            up_block(latent_channels, 64),  # 14x14
            up_block(64, 128),              # 28x28
            up_block(128, 64),               # 56x56
            up_block(64, 32),                # 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),  # 224x224
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.bottleneck(feat)
        recon = self.decoder(z)
        return recon, z


# ── SSIM Loss (학습과 동일) ───────────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """구조적 유사도 기반 loss"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._gaussian_window(window_size))

    def _gaussian_window(self, size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)

    def _ssim_per_sample(self, x, y):
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
        ssim_map = num / den
        return 1 - ssim_map.mean(dim=(1, 2, 3))


# ── Reconstruction Error 계산 (0.8*SSIM + 0.2*MSE) ─────────────────────────────
def compute_recon_errors(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    ssim_loss = SSIMLoss().to(DEVICE)
    errors = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            mse = ((x - recon) ** 2).mean(dim=(1, 2, 3))
            ssim = ssim_loss._ssim_per_sample(recon, x)
            combined = 0.8 * ssim + 0.2 * mse
            errors.extend(combined.cpu().numpy())

    return np.array(errors)


# ── 카테고리 분석 ──────────────────────────────────────────────────────────────
def analyze_category(category):
    """단일 카테고리 분석"""
    result_path = Path(RESULT_DIR) / category
    model_path = result_path / "autoencoder.pt"

    if not model_path.exists():
        return None

    # 모델 로드
    model = ResNetAutoencoder(latent_channels=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # 데이터 로드 (ImageNet 정규화 포함)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MVTecDataset(DATA_DIR, category, split="train", transform=transform)
    test_dataset = MVTecDataset(DATA_DIR, category, split="test", transform=transform)

    # Train recon error (전부 normal)
    train_errors = compute_recon_errors(model, train_dataset)

    # Test recon error (저장된 것 사용 or 다시 계산)
    test_errors_path = result_path / "recon_errors.npy"
    test_labels_path = result_path / "test_binary_labels.npy"

    if test_errors_path.exists() and test_labels_path.exists():
        test_errors = np.load(test_errors_path)
        test_labels = np.load(test_labels_path)
    else:
        test_errors = compute_recon_errors(model, test_dataset)
        test_labels = np.array([label for _, label in test_dataset])

    # 분리
    test_good_errors = test_errors[test_labels == 0]
    test_anomaly_errors = test_errors[test_labels == 1]

    # 지표 계산
    E_train_good_mean = train_errors.mean()
    E_train_good_median = np.median(train_errors)
    E_train_good_std = train_errors.std()

    E_test_good_mean = test_good_errors.mean()
    E_test_good_median = np.median(test_good_errors)
    E_test_good_std = test_good_errors.std()

    E_test_anomaly_mean = test_anomaly_errors.mean()
    E_test_anomaly_median = np.median(test_anomaly_errors)
    E_test_anomaly_std = test_anomaly_errors.std()

    # Gap 계산
    gap_mean = E_test_good_mean / E_train_good_mean if E_train_good_mean > 0 else float('inf')
    gap_median = E_test_good_median / E_train_good_median if E_train_good_median > 0 else float('inf')

    # AUC (anomaly detection 성능)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_labels, test_errors)

    # 판정
    # A: train_good 낮고, gap 작고, anomaly도 낮음 → over-gen
    # B: train_good 높거나 std 큼 → 복원 실패

    if E_train_good_mean > 0.01 or E_train_good_std > 0.005:
        verdict = "B (복원실패)"
    elif gap_median > 2.0:
        verdict = "B (일반화실패)"
    elif auc < 0.7:
        verdict = "A (Over-gen)"
    else:
        verdict = "OK (정상작동)"

    return {
        "category": category,
        "n_train": len(train_errors),
        "n_test_good": len(test_good_errors),
        "n_test_anomaly": len(test_anomaly_errors),
        "E_train_good_mean": E_train_good_mean,
        "E_train_good_median": E_train_good_median,
        "E_train_good_std": E_train_good_std,
        "E_test_good_mean": E_test_good_mean,
        "E_test_good_median": E_test_good_median,
        "E_test_good_std": E_test_good_std,
        "E_test_anomaly_mean": E_test_anomaly_mean,
        "E_test_anomaly_median": E_test_anomaly_median,
        "E_test_anomaly_std": E_test_anomaly_std,
        "gap_mean": gap_mean,
        "gap_median": gap_median,
        "AUC": auc,
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Analyzing MVTec results...\n")

    results = []

    for category in ALL_CATEGORIES:
        print(f"[{category}] ", end="", flush=True)
        result = analyze_category(category)
        if result:
            results.append(result)
            print(f"AUC={result['AUC']:.3f}, gap={result['gap_median']:.2f} → {result['verdict']}")
        else:
            print("SKIP (no model)")

    # DataFrame으로 정리
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)

    # 주요 컬럼만 출력
    summary_cols = ["category", "E_train_good_mean", "E_test_good_mean", "E_test_anomaly_mean",
                    "gap_median", "AUC", "verdict"]
    print(df[summary_cols].to_string(index=False))

    # 저장
    df.to_csv(f"{RESULT_DIR}/analysis_summary.csv", index=False)
    print(f"\n[Saved] {RESULT_DIR}/analysis_summary.csv")

    # 통계
    print("\n" + "="*80)
    print("Verdict Distribution")
    print("="*80)
    print(df["verdict"].value_counts().to_string())

    # A vs B 분류
    over_gen = df[df["verdict"].str.contains("Over-gen")]
    recon_fail = df[df["verdict"].str.contains("복원실패|일반화실패")]
    ok = df[df["verdict"].str.contains("OK")]

    print(f"\nA (Over-generalization): {len(over_gen)} categories")
    if len(over_gen) > 0:
        print(f"  → {', '.join(over_gen['category'].tolist())}")

    print(f"B (복원실패/일반화실패): {len(recon_fail)} categories")
    if len(recon_fail) > 0:
        print(f"  → {', '.join(recon_fail['category'].tolist())}")

    print(f"OK (정상작동): {len(ok)} categories")
    if len(ok) > 0:
        print(f"  → {', '.join(ok['category'].tolist())}")
