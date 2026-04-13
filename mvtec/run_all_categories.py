"""
run_all_categories.py
모든 MVTec AD 카테고리에 대해 전체 파이프라인 실행

파이프라인:
  1. Autoencoder 학습 (mvtec_autoencoder.py)
#   2. Latent space 시각화 (visualize_space.py)
#   3. Reconstruction error 분석 (analyze_recon_error.py)
#   4. VQC 학습 및 평가 (vqc_classifier.py)

사용법:
  python run_all_categories.py                    # 모든 카테고리 실행
  python run_all_categories.py --categories bottle capsule  # 특정 카테고리만
  python run_all_categories.py --skip-ae          # AE 학습 건너뛰기 (이미 학습된 경우)
  python run_all_categories.py --skip-vqc         # VQC 학습 건너뛰기
"""

import subprocess
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# MVTec AD 전체 카테고리 (15개)
ALL_CATEGORIES = [
    "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

MVTEC_DIR = Path(__file__).parent
RESULTS_FILE = MVTEC_DIR / "all_results_summary.json"


def run_script(script_name, category, extra_args=None, no_show=True):
    """스크립트 실행"""
    cmd = [sys.executable, str(MVTEC_DIR / script_name), "--category", category]
    if no_show:
        cmd.append("--no-show")
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"[RUN] {script_name} --category {category}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_pipeline(category, skip_ae=False, skip_vqc=False):
    """단일 카테고리에 대해 전체 파이프라인 실행"""
    print(f"\n{'#'*60}")
    print(f"# Category: {category}")
    print(f"{'#'*60}")

    results = {
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "ae_trained": False,
        "visualized": False,
        "recon_analyzed": False,
        "vqc_trained": False,
        "error": None
    }

    try:
        # 1. Autoencoder 학습
        if not skip_ae:
            print(f"\n[Step 1/4] Training Autoencoder for {category}...")
            success = run_script("mvtec_autoencoder.py", category)
            results["ae_trained"] = success
            if not success:
                results["error"] = "Autoencoder training failed"
                return results
        else:
            print(f"\n[Step 1/4] Skipping Autoencoder (--skip-ae)")
            results["ae_trained"] = True  # 이미 학습됨

        # # 2. Latent space 시각화
        # print(f"\n[Step 2/4] Visualizing latent space for {category}...")
        # success = run_script("visualize_space.py", category)
        # results["visualized"] = success

        # # 3. Reconstruction error 분석
        # print(f"\n[Step 3/4] Analyzing reconstruction error for {category}...")
        # success = run_script("analyze_recon_error.py", category)
        # results["recon_analyzed"] = success

        # # 4. VQC 학습
        # if not skip_vqc:
        #     print(f"\n[Step 4/4] Training VQC for {category}...")
        #     success = run_script("vqc_classifier.py", category)
        #     results["vqc_trained"] = success
        #     if not success:
        #         results["error"] = "VQC training failed"
        # else:
        #     print(f"\n[Step 4/4] Skipping VQC (--skip-vqc)")
        #     results["vqc_trained"] = True

    except Exception as e:
        results["error"] = str(e)

    return results


def load_vqc_results(category):
    """VQC 결과 파일에서 AUC 등 로드"""
    result_path = MVTEC_DIR.parent / f"results/mvtec/{category}/vqc_svdd_model.pt"
    if result_path.exists():
        import torch
        checkpoint = torch.load(result_path, map_location="cpu", weights_only=False)
        return {
            "auc": checkpoint.get("auc", None),
            "threshold": checkpoint.get("threshold", None)
        }
    return None


def print_summary(all_results):
    """전체 결과 요약 출력"""
    print("\n" + "="*70)
    print(" SUMMARY - All Categories")
    print("="*70)

    print(f"\n{'Category':<15} {'AE':^8} {'Viz':^8} {'Recon':^8} {'VQC':^8} {'AUC':^10}")
    print("-"*70)

    for r in all_results:
        cat = r["category"]
        ae = "OK" if r["ae_trained"] else "FAIL"
        viz = "OK" if r["visualized"] else "FAIL"
        recon = "OK" if r["recon_analyzed"] else "FAIL"
        vqc = "OK" if r["vqc_trained"] else "FAIL"

        # VQC AUC 로드
        vqc_result = load_vqc_results(cat)
        auc = f"{vqc_result['auc']:.4f}" if vqc_result and vqc_result['auc'] else "N/A"

        print(f"{cat:<15} {ae:^8} {viz:^8} {recon:^8} {vqc:^8} {auc:^10}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline for all MVTec AD categories")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Specific categories to run (default: all)")
    parser.add_argument("--skip-ae", action="store_true",
                        help="Skip Autoencoder training (use existing)")
    parser.add_argument("--skip-vqc", action="store_true",
                        help="Skip VQC training")
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
    print(f"[Config] Skip AE: {args.skip_ae}")
    print(f"[Config] Skip VQC: {args.skip_vqc}")

    if args.summary_only:
        # 기존 결과만 요약
        all_results = [{"category": cat, "ae_trained": True, "visualized": True,
                        "recon_analyzed": True, "vqc_trained": True} for cat in categories]
        print_summary(all_results)
        return

    # 파이프라인 실행
    all_results = []
    for category in categories:
        result = run_pipeline(category, skip_ae=args.skip_ae, skip_vqc=args.skip_vqc)
        all_results.append(result)

        # 중간 저장
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)

    # 최종 요약
    print_summary(all_results)

    # 결과 저장
    print(f"\n[Saved] Results summary: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
