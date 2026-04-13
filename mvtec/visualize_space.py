import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
parser.add_argument("--no-show", action="store_true", help="Don't display plots (for automation)")
args, _ = parser.parse_known_args()

if args.no_show:
    matplotlib.use('Agg')  # Non-interactive backend

CATEGORY = args.category
SAVE_DIR  = f"./preprocessed/mvtec/{CATEGORY}"

# 저장된 파일 로딩
test_latents       = np.load(f"{SAVE_DIR}/test_latents.npy")
test_binary_labels = np.load(f"{SAVE_DIR}/test_binary_labels.npy")

print(f"[Load] latents: {test_latents.shape}")
print(f"[Load] normal: {(test_binary_labels==0).sum()}, anomaly: {(test_binary_labels==1).sum()}")

# ── PCA 시각화 ─────────────────────────────────────────────────
pca     = PCA(n_components=2)
pca_out = pca.fit_transform(test_latents)

plt.figure(figsize=(8, 6))
plt.scatter(pca_out[test_binary_labels == 0, 0],
            pca_out[test_binary_labels == 0, 1],
            s=15, alpha=0.6, label="정상", color="blue")
plt.scatter(pca_out[test_binary_labels == 1, 0],
            pca_out[test_binary_labels == 1, 1],
            s=15, alpha=0.6, label="이상", color="red")
plt.title(f"PCA: Latent Space ({CATEGORY})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/pca_latent.png")
if not args.no_show:
    plt.show()
print("[Saved] pca_latent.png")

# ── t-SNE 시각화 ───────────────────────────────────────────────
print("[t-SNE] 계산 중...")
tsne    = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_out = tsne.fit_transform(test_latents)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_out[test_binary_labels == 0, 0],
            tsne_out[test_binary_labels == 0, 1],
            s=15, alpha=0.6, label="정상", color="blue")
plt.scatter(tsne_out[test_binary_labels == 1, 0],
            tsne_out[test_binary_labels == 1, 1],
            s=15, alpha=0.6, label="이상", color="red")
plt.title(f"t-SNE: Latent Space ({CATEGORY})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/tsne_latent.png")
if not args.no_show:
    plt.show()
print("[Saved] tsne_latent.png")
