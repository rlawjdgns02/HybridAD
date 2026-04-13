"""
run_all_visapp.py
VISAPP CAE로 모든 MVTec AD 카테고리에 대해 학습 및 시각화 실행

파이프라인:
  1. VISAPP CAE 학습 (mvtec_cae_visapp.py)
  2. 복원 시각화 + Latent space 시각화 (자동 포함)

사용법:
  python run_all_visapp.py                         # 모든 카테고리 실행
  python run_all_visapp.py --categories carpet grid leather  # 특정 카테고리만
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

# MVTec AD 전체 카테고리 (15개)
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

MVTEC_DIR = Path(__file__).parent
RESULTS_FILE = MVTEC_DIR / "visapp_results_summary.json"


def run_visapp_cae(category):
    """VISAPP CAE 학습 및 시각화 실행"""
    cmd = [
        sys.executable,
        str(MVTEC_DIR / "mvtec_cae_visapp.py"),
        "--category", category,
        "--no-show"
    ]

    print(f"\n{'='*60}")
    print(f"[RUN] mvtec_cae_visapp.py --category {category}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def load_results(category):
    """저장된 결과에서 AUC 로드"""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    result_dir = MVTEC_DIR.parent / f"preprocessed/mvtec_visapp/{category}"
    try:
        errors = np.load(result_dir / "recon_errors.npy")
        labels = np.load(result_dir / "test_binary_labels.npy")
        auc = roc_auc_score(labels, errors)
        return {"recon_auc": float(auc)}
    except Exception:
        return {"recon_auc": None}


def print_summary(all_results):
    """전체 결과 요약 출력"""
    print("\n" + "="*60)
    print(" VISAPP CAE Results Summary")
    print("="*60)

    print(f"\n{'Category':<15} {'Status':^10} {'Recon AUC':^12} {'Over-gen?':^12}")
    print("-"*60)

    for r in all_results:
        cat = r["category"]
        status = "OK" if r["success"] else "FAIL"
        auc = r.get("recon_auc")
        auc_str = f"{auc:.4f}" if auc else "N/A"
        overgen = "Yes" if auc and auc < 0.65 else ("No" if auc else "N/A")

        print(f"{cat:<15} {status:^10} {auc_str:^12} {overgen:^12}")

    print("="*60)

    # Over-generalization 성공 카테고리 (AUC < 0.65)
    overgen_cats = [r["category"] for r in all_results
                    if r.get("recon_auc") and r["recon_auc"] < 0.65]
    if overgen_cats:
        print(f"\n[Over-generalization] AUC < 0.65: {overgen_cats}")
        print("  → 이 카테고리들은 Recon error로 탐지 어려움")
        print("  → Latent space에서 VQC 분류 테스트 필요")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run VISAPP CAE for all MVTec categories")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Specific categories (default: all)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print summary of existing results")
    args = parser.parse_args()

    categories = args.categories if args.categories else ALL_CATEGORIES

    # 유효성 검사
    for cat in categories:
        if cat not in ALL_CATEGORIES:
            print(f"[ERROR] Unknown category: {cat}")
            print(f"Available: {ALL_CATEGORIES}")
            return

    print(f"[Config] Categories: {categories}")
    print(f"[Config] Total: {len(categories)} categories")

    if args.summary_only:
        all_results = []
        for cat in categories:
            result = {"category": cat, "success": True}
            result.update(load_results(cat))
            all_results.append(result)
        print_summary(all_results)
        return

    # 파이프라인 실행
    all_results = []
    for i, category in enumerate(categories, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(categories)}] Category: {category}")
        print(f"{'#'*60}")

        result = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "recon_auc": None
        }

        success = run_visapp_cae(category)
        result["success"] = success

        if success:
            result.update(load_results(category))

        all_results.append(result)

        # 중간 저장
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)

    # 최종 요약
    print_summary(all_results)
    print(f"\n[Saved] Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
