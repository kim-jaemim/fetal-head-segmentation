"""
============================================================
FCN(Fully Convolutional Network) 태아 머리 영상 분할 모델 & 실행 파이프라인
============================================================

1. FCN 모델 구성
------------------------------------------------------------
- 모델 구조: FCN (torchvision fcn_resnet50 / fcn_resnet101)
- Backbone : Dilated ResNet50 (기본) / Dilated ResNet101
             (출력 스트라이드 8, 특징맵 1/8 해상도 유지)
- 사전 학습 가중치: ImageNet (weights_backbone만 사용, FCNHead는 처음부터 학습)
- 출력층 구성: num_classes=1 -> Conv2d(512, 1, kernel_size=1), aux_loss=False
- 입력 채널: 3 (흑백 초음파를 3채널로 복제 + ImageNet 정규화)
- 출력 채널: 1 (256x256 로짓 텐서 (N, 1, H, W) -> sigmoid 확률 -> 이진 마스크)

2. 공정 비교 및 실험 설계
------------------------------------------------------------
- U-Net과 동일한 조건에서 비교하기 위해 공통 학습/평가 코드(config.py, data.py,
  dataset.py, augment.py, models.py, train.py, metrics.py)를 그대로 사용한다.
- 데이터 분할: 환자 단위 5-Fold 교차검증 (splits/folds.csv, 총 853장)
- 학습 설정 : BCE + Dice Loss, AdamW(lr=3e-4, weight_decay=1e-4),
             CosineAnnealingLR, 30 epochs, batch_size=16, 256x256
- 비교 실험 :
  1) 증강 효과 비교: aug=True vs aug=False (-noaug)
  2) 백본 깊이 비교: fcn_resnet50 vs fcn_resnet101
  3) 후처리 효과 비교: 타원 피팅 전(-seg) vs 타원 피팅 후(공식 점수)

3. 백본 및 설정별 5-Fold 평균 실험 결과 (Colab 실행 후 기입)
------------------------------------------------------------
Model / Variant         Params       IoU      Dice     Time(ms)   HC_MAE(mm)
----------------------------------------------------------------------------
fcn_resnet50            32,946,753   TBD      TBD      TBD        TBD
fcn_resnet50-seg        32,946,753   TBD      TBD      TBD        TBD
fcn_resnet50-noaug      32,946,753   TBD      TBD      TBD        TBD
fcn_resnet101           51,938,881   TBD      TBD      TBD        TBD
----------------------------------------------------------------------------

4. 실행 예시 (레포 루트 또는 Colab)
------------------------------------------------------------
- 사전 점검만 실행          : python src/fcn.py --check
- FCN-ResNet50 5-Fold     : python src/fcn.py --backbone resnet50
- 전체 비교 실험 일괄 실행  : python src/fcn.py --all
- 결과 요약 표 / 시각화     : python src/fcn.py --summarize --plot_dir outputs/fcn
============================================================
"""
import argparse
import glob
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models import segmentation as seg

# 레포 루트에서 `from src import fcn`으로 부를 때도 형제 모듈을 찾을 수 있도록 경로 보정
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
import models
import train
from data import load_fold
from metrics import count_params

BACKBONES = {
    "resnet50": (seg.fcn_resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (seg.fcn_resnet101, ResNet101_Weights.IMAGENET1K_V1),
}


class _FCNOutputWrapper(nn.Module):
    """torchvision segmentation 모델의 dict({'out': tensor}) 출력을 텐서로 통일한다."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m(x)["out"]


def build_fcn(backbone="resnet50", pretrained=True):
    """태아 머리 이진 분할용 FCN 모델(출력 shape: (N, 1, H, W) 로짓)을 생성한다.

    Args:
        backbone (str): 'resnet50' 또는 'resnet101'
        pretrained (bool): 백본에 ImageNet 사전학습 가중치 적용 여부

    Returns:
        nn.Module: (N, 1, H, W) 로짓을 반환하는 FCN 모델
    """
    if backbone not in BACKBONES:
        raise ValueError(f"지원하지 않는 백본 '{backbone}'. 사용 가능: {tuple(BACKBONES)}")

    fn, weights = BACKBONES[backbone]
    net = fn(
        weights=None,
        weights_backbone=weights if pretrained else None,
        num_classes=1,
        aux_loss=False,
    )
    return _FCNOutputWrapper(net)


def make_predict_fn(model, device="cuda", size=256, thr=0.5, ellipse=True):
    """공통 전처리(dataset.preprocess_image)와 후처리(타원 피팅)를 묶은 predict_fn을 반환한다."""
    return models.make_predict_fn(model, device=device, size=size, thr=thr, ellipse=ellipse)


def configure_drive_root(drive_root=None):
    """체크포인트/결과 저장 루트를 설정한다.

    상단 공통 설정이 U-Net 전용 폴더로 잡혀 있고 별도 지정이 없으면 FCN 전용 폴더로 분리한다.
    """
    if drive_root is None:
        if "U-Net" not in getattr(config, "DRIVE_ROOT", ""):
            return config.DRIVE_ROOT
        drive_root = "/content/drive/MyDrive/FCN_실험결과"

    os.environ["CV_DRIVE_ROOT"] = drive_root
    config.DRIVE_ROOT = drive_root
    config.CKPT_DIR = os.path.join(drive_root, "checkpoints")
    config.RESULT_DIR = os.path.join(drive_root, "results")
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    return drive_root


def _resolve_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def check_setup(backbones=("resnet50", "resnet101"), pretrained=True, device=None,
                data_dir=None, split_dir=None):
    """데이터 경로, FCN 입출력 shape, 파라미터 수, predict_fn 동작을 사전 점검한다."""
    device = _resolve_device(device)
    data_dir = data_dir or config.DATA_DIR
    split_dir = split_dir or config.SPLIT_DIR

    print(f"Device    : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"DATA_DIR  : {data_dir} -> exists: {os.path.isdir(data_dir)}")
    print(f"HC_CSV    : {config.HC_CSV} -> exists: {os.path.exists(config.HC_CSV)}")
    print(f"CKPT_DIR  : {config.CKPT_DIR}")
    print(f"RESULT_DIR: {config.RESULT_DIR}")

    (_, _, _), (_, va_i, va_g) = load_fold(0, data_dir, split_dir)
    info = {}
    for backbone in backbones:
        m = build_fcn(backbone, pretrained=pretrained).to(device).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 256, 256, device=device))
        if out.shape != (2, 1, 256, 256):
            raise AssertionError(f"fcn_{backbone} 출력 shape 오류: {out.shape}")
        params = count_params(m)
        info[f"fcn_{backbone}"] = params
        print(f"fcn_{backbone:10s} | params: {params:,} | out: {tuple(out.shape)}")

        pred0 = make_predict_fn(m, device=device, size=256, ellipse=True)(va_i[0])
        if pred0.shape != va_g[0].shape or not set(pd.unique(pred0.ravel())).issubset({0, 1}):
            raise AssertionError(f"fcn_{backbone} predict_fn 출력 검증 실패: {pred0.shape}")
        del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return info


def run(backbone="resnet50", folds=None, **kwargs):
    """공통 학습 파이프라인(train.py)을 통해 단일 FCN 백본의 폴드 학습·검증을 실행한다."""
    if backbone not in BACKBONES:
        raise ValueError(f"지원하지 않는 백본 '{backbone}'. 사용 가능: {tuple(BACKBONES)}")
    return train.run(f"fcn_{backbone}", folds=folds, **kwargs)


def run_experiments(folds=None, include_noaug=True, include_resnet101=True,
                    epochs=30, batch_size=16, lr=3e-4, **kwargs):
    """노트북의 FCN 비교 실험(기본 ResNet50, 증강 OFF, ResNet101)을 순서대로 실행한다."""
    results = {
        "fcn_resnet50": run("resnet50", folds=folds, epochs=epochs, batch_size=batch_size,
                            lr=lr, aug=True, **kwargs),
    }
    if include_noaug:
        results["fcn_resnet50-noaug"] = run(
            "resnet50", folds=folds, epochs=epochs, batch_size=batch_size,
            lr=lr, aug=False, **kwargs,
        )
    if include_resnet101:
        results["fcn_resnet101"] = run(
            "resnet101", folds=folds, epochs=epochs, batch_size=batch_size,
            lr=lr, aug=True, **kwargs,
        )
    return results


def summarize(result_dir=None):
    """results/logs/fcn_*_summary.csv를 모아 모델별 폴드 평균·표준편차 표를 반환한다."""
    result_dir = result_dir or config.RESULT_DIR
    log_dir = os.path.join(result_dir, "logs")
    files = sorted(glob.glob(os.path.join(log_dir, "fcn_*_summary.csv")))
    if not files:
        print("요약 CSV가 아직 없습니다:", log_dir)
        return None

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    cols = [c for c in ("IoU", "Dice", "seg_IoU", "seg_Dice", "ms_per_image", "HC_MAE_mm", "params", "train_sec") if c in df]
    summary = df.groupby("method")[cols].agg(["mean", "std"]).round(4)
    print("\n[FCN 폴드 요약]")
    print(summary.to_string())
    return summary


def _plot_history(log_dir, out_path):
    import matplotlib.pyplot as plt

    hist_files = sorted(glob.glob(os.path.join(log_dir, "fcn_resnet50__fold*_history.csv")))
    if not hist_files:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for hp in hist_files:
        tag = os.path.basename(hp).split("__")[1].replace("_history.csv", "")
        hdf = pd.read_csv(hp)
        axes[0].plot(hdf["epoch"], hdf["train_loss"], label=tag)
        axes[1].plot(hdf["epoch"], hdf["val_dice_256"], label=tag)
    axes[0].set_title("FCN-ResNet50 Train Loss")
    axes[1].set_title("FCN-ResNet50 Val Dice 256")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _plot_cases(fold, result_dir, data_dir, split_dir, device, out_path):
    import matplotlib.pyplot as plt

    ckpt_file = config.ckpt_path("fcn_resnet50", fold)
    val_csv = os.path.join(result_dir, f"fcn_resnet50__val__fold{fold}.csv")
    if not (os.path.exists(ckpt_file) and os.path.exists(val_csv)):
        return None

    (_, _, _), (_, va_i, va_g) = load_fold(fold, data_dir, split_dir)
    per_df = pd.read_csv(val_csv)

    model = build_fcn("resnet50", pretrained=False).to(device)
    ckpt_obj = torch.load(ckpt_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_obj["state_dict"])

    pred_seg = make_predict_fn(model, device=device, size=256, ellipse=False)
    pred_ell = make_predict_fn(model, device=device, size=256, ellipse=True)

    sorted_idx = per_df["iou"].argsort().values
    k = min(3, len(sorted_idx))
    worst = list(sorted_idx[:k])
    best = list(sorted_idx[-k:][::-1])
    picks = [(i, f"Worst {n + 1}") for n, i in enumerate(worst)] + [
        (i, f"Best {n + 1}") for n, i in enumerate(best)
    ]

    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.2 * len(picks)), squeeze=False)
    for row, (idx, tag) in enumerate(picks):
        img, gt = va_i[idx], va_g[idx]
        p_seg, p_ell = pred_seg(img), pred_ell(img)
        r = per_df.iloc[idx]

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"{tag}: {r['image']}")

        axes[row, 1].imshow(img, cmap="gray")
        if gt.max() > 0:
            axes[row, 1].contour(gt, colors="lime", linewidths=1.2)
        if p_seg.max() > 0:
            axes[row, 1].contour(p_seg, colors="cyan", linewidths=1.2)
        axes[row, 1].set_title("GT (green) vs Raw FCN (cyan)")

        axes[row, 2].imshow(img, cmap="gray")
        if gt.max() > 0:
            axes[row, 2].contour(gt, colors="lime", linewidths=1.2)
        if p_ell.max() > 0:
            axes[row, 2].contour(p_ell, colors="red", linewidths=1.2)
        hc_txt = f" | HC err {r['hc_err']:.2f}mm" if "hc_err" in r and pd.notna(r["hc_err"]) else ""
        empty_txt = " [EMPTY]" if p_ell.max() == 0 else ""
        axes[row, 2].set_title(f"Ellipse FCN{empty_txt} | IoU {r['iou']:.3f}{hc_txt}")

        for c in range(3):
            axes[row, c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_analysis_plots(out_dir, fold=0, result_dir=None, data_dir=None, split_dir=None, device=None):
    """학습 곡선과 Fold 성공(Best 3)/실패(Worst 3) 정성 비교 이미지를 저장한다."""
    os.makedirs(out_dir, exist_ok=True)
    result_dir = result_dir or config.RESULT_DIR
    data_dir = data_dir or config.DATA_DIR
    split_dir = split_dir or config.SPLIT_DIR
    device = _resolve_device(device)

    saved = []
    hist_png = _plot_history(os.path.join(result_dir, "logs"), os.path.join(out_dir, "fcn_resnet50_curves.png"))
    if hist_png:
        saved.append(hist_png)
    case_png = _plot_cases(fold, result_dir, data_dir, split_dir, device,
                           os.path.join(out_dir, f"fcn_resnet50_fold{fold}_cases.png"))
    if case_png:
        saved.append(case_png)
    return saved


def main():
    ap = argparse.ArgumentParser(description="FCN 태아 머리 분할 실행 파이프라인 (공통 train.py 연동)")
    ap.add_argument("--backbone", default="resnet50", choices=tuple(BACKBONES))
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="예: 0 1 2 (기본: 전체 폴드)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--no_aug", action="store_true", help="증강 없이 학습 (증강 효과 비교용)")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--drive_root", default=None, help="체크포인트/결과 저장 루트 덮어쓰기")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--result_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prepare_data", action="store_true", help="Drive zip 압축 해제 및 HC csv 준비")
    ap.add_argument("--check", action="store_true", help="데이터/모델/predict_fn 사전 점검만 실행")
    ap.add_argument("--all", action="store_true", help="ResNet50(증강 ON/OFF) + ResNet101 일괄 실행")
    ap.add_argument("--summarize", action="store_true", help="폴드 요약 표 출력")
    ap.add_argument("--plot_dir", default=None, help="학습 곡선 및 성공/실패 시각화 이미지 저장 폴더")
    a = ap.parse_args()

    configure_drive_root(a.drive_root)
    if a.prepare_data:
        config.prepare_data()

    if a.check:
        check_setup(pretrained=not a.no_pretrained, device=a.device,
                    data_dir=a.data_dir, split_dir=a.split_dir)
        return

    common_kwargs = dict(
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        size=a.size,
        seed=a.seed,
        num_workers=a.num_workers,
        pretrained=not a.no_pretrained,
        overwrite=a.overwrite,
        data_dir=a.data_dir,
        split_dir=a.split_dir,
        result_dir=a.result_dir,
        device=a.device,
    )

    if a.all:
        run_experiments(folds=a.folds, **common_kwargs)
    elif not (a.summarize or a.plot_dir):
        run(a.backbone, folds=a.folds, aug=not a.no_aug, **common_kwargs)

    if a.summarize or a.all:
        summarize(result_dir=a.result_dir)
    if a.plot_dir:
        fold_for_plot = a.folds[0] if a.folds else 0
        saved = save_analysis_plots(a.plot_dir, fold=fold_for_plot, result_dir=a.result_dir,
                                    data_dir=a.data_dir, split_dir=a.split_dir, device=a.device)
        print("저장된 시각화:", saved)


if __name__ == "__main__":
    main()
