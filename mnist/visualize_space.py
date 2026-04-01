import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

SAVE_DIR = "./preprocessed"

# 저장된 파일 로딩
test_latents      = np.load(f"{SAVE_DIR}/test_latents.npy")
test_binary_labels = np.load(f"{SAVE_DIR}/test_binary_labels.npy")

# ── t-SNE 시각화 ──────────────────────────────────────────────
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced = tsne.fit_transform(test_latents)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[test_binary_labels == 0, 0],
            reduced[test_binary_labels == 0, 1],
            s=5, alpha=0.5, label="정상 (0)", color="blue")
plt.scatter(reduced[test_binary_labels == 1, 0],
            reduced[test_binary_labels == 1, 1],
            s=5, alpha=0.3, label="이상 (1~9)", color="red")
plt.title("t-SNE: Latent Space Visualization")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/tsne_latent.png")
plt.show()