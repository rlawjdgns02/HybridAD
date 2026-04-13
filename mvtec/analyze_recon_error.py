"""
analyze_recon_error.py
Autoencoder의 Reconstruction Error로 정상/이상 분리 가능성 분석
모든 카테고리 또는 개별 카테고리 분석 가능
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="all", help="MVTec category (or 'all' for all categories)")
parser.add_argument("--no-show", action="store_true", help="Don't display plots (for automation)")
args, _ = parser.parse_known_args()

if args.no_show:
    matplotlib.use('Agg')

# ── Config ────────────────────────────────────────────────────────────────────
LATENT_DIM = 16
IMG_SIZE   = 224
DATA_DIR   = "./data/mvtec_anomaly_detection"
RESULT_DIR = "./preprocessed/mvtec"

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


# ── Dataset ───────────────────────────────────────────────────────────────────
class MVTecDataset:
    def __init__(self, root_dir, category, split="test", transform=None):
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


# ── Model (LATENT_DIM=16 버전, decoder: 64→128→64→32) ─────────────────────────
class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_channels=LATENT_DIM):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_channels, 1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
        )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

        # 학습된 모델과 동일한 구조: 64→128→64→32
        self.decoder = nn.Sequential(
            up_block(latent_channels, 64),   # 14x14
            up_block(64, 128),               # 28x28
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
        """배치 내 각 샘플별 SSIM 계산 (1 - SSIM = loss)"""
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

        # 샘플별 평균 (배치 차원 유지)
        return 1 - ssim_map.mean(dim=(1, 2, 3))  # (B,)


# ── Reconstruction Error 계산 ─────────────────────────────────────────────────
def compute_recon_errors(model, dataset):
    """각 이미지의 reconstruction error (0.8*SSIM + 0.2*MSE, 학습과 동일)"""
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()

    ssim_loss = SSIMLoss().to(DEVICE)
    errors = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            recon, _ = model(x)

            # 학습과 동일한 loss: 0.8 * SSIM + 0.2 * MSE
            mse = ((x - recon) ** 2).mean(dim=(1, 2, 3))  # (B,)
            ssim = ssim_loss._ssim_per_sample(recon, x)   # (B,)
            combined = 0.8 * ssim + 0.2 * mse

            errors.extend(combined.cpu().numpy())
            labels.extend(y if isinstance(y, list) else y.tolist())

    return np.array(errors), np.array(labels)


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_error_distribution(errors, labels, save_dir, category):
    """Reconstruction error 분포 시각화"""
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 히스토그램
    axes[0, 0].hist(normal_errors, bins=30, alpha=0.7, label=f'정상 (n={len(normal_errors)})', color='blue')
    axes[0, 0].hist(anomaly_errors, bins=30, alpha=0.7, label=f'이상 (n={len(anomaly_errors)})', color='red')
    axes[0, 0].set_xlabel('Reconstruction Error (SSIM+MSE)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'[{category}] Reconstruction Error 분포')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot
    axes[0, 1].boxplot([normal_errors, anomaly_errors], labels=['정상', '이상'])
    axes[0, 1].set_ylabel('Reconstruction Error (SSIM+MSE)')
    axes[0, 1].set_title('Reconstruction Error Box Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scatter plot
    axes[1, 0].scatter(range(len(normal_errors)), normal_errors,
                       c='blue', alpha=0.6, s=20, label='정상')
    axes[1, 0].scatter(range(len(normal_errors), len(errors)), anomaly_errors,
                       c='red', alpha=0.6, s=20, label='이상')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Reconstruction Error (SSIM+MSE)')
    axes[1, 0].set_title('샘플별 Reconstruction Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. ROC Curve
    auc = roc_auc_score(labels, errors)
    fpr, tpr, _ = roc_curve(labels, errors)
    axes[1, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve (Reconstruction Error 기반)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/recon_error_analysis.png", dpi=150)
    plt.close()

    return auc, normal_errors, anomaly_errors


def analyze_single_category(category):
    """단일 카테고리 분석"""
    save_dir = f"{RESULT_DIR}/{category}"
    model_path = f"{save_dir}/autoencoder.pt"

    if not Path(model_path).exists():
        print(f"[Skip] {category} - 모델 없음")
        return None

    print(f"\n{'='*50}")
    print(f"[{category}] 분석 시작")
    print(f"{'='*50}")

    # 데이터 로딩
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MVTecDataset(DATA_DIR, category, split="train", transform=transform)
    test_dataset = MVTecDataset(DATA_DIR, category, split="test", transform=transform)

    # 모델 로딩
    model = ResNetAutoencoder(latent_channels=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    # Reconstruction error 계산
    train_errors, train_labels = compute_recon_errors(model, train_dataset)
    test_errors, test_labels = compute_recon_errors(model, test_dataset)

    # 시각화
    auc, normal_errors, anomaly_errors = plot_error_distribution(test_errors, test_labels, save_dir, category)

    # 통계 계산
    test_normal_mask = test_labels == 0
    test_anomaly_mask = test_labels == 1

    E_train = train_errors.mean()
    E_test_good = test_errors[test_normal_mask].mean()
    E_test_anomaly = test_errors[test_anomaly_mask].mean()
    gap = E_test_good / E_train if E_train > 0 else float('inf')

    # 결과 딕셔너리
    result = {
        "category": category,
        "E_train_mean": float(E_train),
        "E_train_std": float(train_errors.std()),
        "E_test_good_mean": float(E_test_good),
        "E_test_good_std": float(test_errors[test_normal_mask].std()),
        "E_test_anomaly_mean": float(E_test_anomaly),
        "E_test_anomaly_std": float(test_errors[test_anomaly_mask].std()),
        "gap": float(gap),
        "AUC": float(auc),
        "n_train": len(train_errors),
        "n_test_normal": int(test_normal_mask.sum()),
        "n_test_anomaly": int(test_anomaly_mask.sum()),
    }

    # Verdict 결정
    if E_train > 0.01 or train_errors.std() > 0.005:
        verdict = "B (복원실패)"
    elif gap > 2.0:
        verdict = "B (일반화실패)"
    elif auc < 0.7:
        verdict = "A (Over-gen)"
    else:
        verdict = "OK (정상작동)"

    result["verdict"] = verdict

    # 출력
    print(f"\n[Stats] {category}")
    print(f"  E_train:       {E_train:.6f} ± {train_errors.std():.6f}")
    print(f"  E_test_good:   {E_test_good:.6f}")
    print(f"  E_test_anomaly:{E_test_anomaly:.6f}")
    print(f"  Gap:           {gap:.2f}")
    print(f"  AUC:           {auc:.4f}")
    print(f"  Verdict:       {verdict}")
    print(f"[Saved] {save_dir}/recon_error_analysis.png")

    return result


def print_summary(results):
    """전체 결과 요약 출력"""
    print("\n" + "="*80)
    print("전체 결과 요약")
    print("="*80)
    print(f"{'Category':<15} {'E_train':<12} {'E_test_good':<12} {'Gap':<8} {'AUC':<8} {'Verdict':<20}")
    print("-"*80)

    for r in results:
        print(f"{r['category']:<15} {r['E_train_mean']:<12.6f} {r['E_test_good_mean']:<12.6f} "
              f"{r['gap']:<8.2f} {r['AUC']:<8.4f} {r['verdict']:<20}")

    print("-"*80)

    # 분류별 집계
    verdicts = {}
    for r in results:
        v = r['verdict'].split()[0]  # "A", "B", "OK"
        if v not in verdicts:
            verdicts[v] = []
        verdicts[v].append(r['category'])

    print("\n[분류별 카테고리]")
    for v, cats in sorted(verdicts.items()):
        print(f"  {v}: {', '.join(cats)}")

    # JSON 저장
    summary_path = f"{RESULT_DIR}/recon_error_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    if args.category.lower() == "all":
        # 모든 카테고리 분석
        print(f"모든 카테고리 분석 시작...")
        results = []

        for category in ALL_CATEGORIES:
            result = analyze_single_category(category)
            if result:
                results.append(result)

        if results:
            print_summary(results)
    else:
        # 단일 카테고리 분석
        result = analyze_single_category(args.category)

        if result:
            print("\n" + "="*50)
            if result['AUC'] > 0.8:
                print(f"[결론] AUC {result['AUC']:.4f} → AE가 정상/이상을 잘 분리함!")
                print("       → VQC 학습 가능성 높음")
            elif result['AUC'] > 0.6:
                print(f"[결론] AUC {result['AUC']:.4f} → 어느 정도 분리되지만 애매함")
                print("       → VQC로 개선 가능성 있음")
            else:
                print(f"[결론] AUC {result['AUC']:.4f} → AE가 정상/이상을 분리 못함")
                print("       → latent space 자체가 문제, VQC도 어려울 수 있음")
            print("="*50)
