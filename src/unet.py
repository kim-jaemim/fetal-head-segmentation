
"""
============================================================
U-Net(ResNet34) 태아 머리 영상 분할 모델 & 실행 파이프라인
============================================================

1. U-Net 모델 구성
------------------------------------------------------------
- 모델 구조: U-Net
- Encoder : ResNet34
- Decoder : U-Net Decoder
- Skip Connection : Encoder의 특징을 Decoder로 전달
- 사전 학습 가중치: ImageNet
- 입력 채널: 3 (흑백 초음파 영상을 3채널로 변환)
- 출력 채널: 1 (태아 머리 영역의 이진 분할 로짓)
- 입력 크기: 256 x 256
- 출력 크기: (N, 1, 256, 256)

2. 공정 비교 및 실험 설계
------------------------------------------------------------
- 공통 학습/평가 코드 사용
  (config.py, data.py, dataset.py, augment.py,
   models.py, train.py, metrics.py)
- 데이터 분할: 환자 단위 5-Fold 교차검증
- 학습 설정: BCE + Dice Loss, AdamW,
             CosineAnnealingLR, 30 epochs,
             batch_size=16, learning rate=3e-4
- 최종 모델 이름: unet_resnet34

3. 최종 모델 5-Fold 평균 실험 결과
------------------------------------------------------------
- IoU       : 0.9537
- Dice      : 0.9745
- HC_MAE    : 1.9765 mm
- Time      : 12.3083 ms/image
- Parameters: 24,436,369

4. 실행 예시
------------------------------------------------------------
- 모델 사전 점검:
  python src/unet.py --check

- Fold 0 학습:
  python src/unet.py --folds 0

- 전체 5-Fold 학습:
  python src/unet.py --folds 0 1 2 3 4

- 결과 요약:
  python src/unet.py --summarize

- 결과 요약 및 시각화:
  python src/unet.py --summarize --plot_dir outputs/unet
============================================================
"""

import argparse
import glob
import os
import sys

import pandas as pd
import torch

# 형제 모듈 경로 설정
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
import models
import train

from data import load_fold
from metrics import count_params


MODEL_NAME = "unet_resnet34"


# ============================================================
# 1. U-Net 모델 생성
# ============================================================

def build_unet(pretrained=True):
    """
    models.py에 등록된 U-Net ResNet34 모델을 생성한다.

    Returns:
        nn.Module: (N, 1, H, W) 로짓을 반환하는 U-Net 모델
    """

    if MODEL_NAME not in models.REGISTRY:
        raise ValueError(
            f"{MODEL_NAME} 모델이 models.py에 등록되어 있지 않습니다."
        )

    return models.build_model(
        MODEL_NAME,
        pretrained=pretrained
    )


# ============================================================
# 2. 예측 함수
# ============================================================

def make_predict_fn(model, device="cuda", size=256,
                    thr=0.5, ellipse=True):
    """
    공통 전처리 및 후처리를 적용한 예측 함수를 반환한다.
    """

    return models.make_predict_fn(
        model,
        device=device,
        size=size,
        thr=thr,
        ellipse=ellipse
    )


# ============================================================
# 3. Google Drive 저장 경로 설정
# ============================================================

def configure_drive_root(drive_root=None):
    """
    U-Net 체크포인트와 실험 결과 저장 경로를 설정한다.
    """

    if drive_root is None:
        drive_root = "/content/drive/MyDrive/U-Net_실험결과"

    os.environ["CV_DRIVE_ROOT"] = drive_root

    config.DRIVE_ROOT = drive_root
    config.CKPT_DIR = os.path.join(
        drive_root,
        "checkpoints"
    )
    config.RESULT_DIR = os.path.join(
        drive_root,
        "results"
    )

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    return drive_root


# ============================================================
# 4. 실행 장치 설정
# ============================================================

def _resolve_device(device=None):
    return device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# 5. 모델 사전 점검
# ============================================================

def check_setup(pretrained=True, device=None,
                data_dir=None, split_dir=None):
    """
    데이터 경로, 모델 입출력 크기, 파라미터 수,
    predict_fn 동작을 사전 점검한다.
    """

    device = _resolve_device(device)

    data_dir = data_dir or config.DATA_DIR
    split_dir = split_dir or config.SPLIT_DIR

    print(
        f"Device     : {device}"
        + (
            f" ({torch.cuda.get_device_name(0)})"
            if device == "cuda" else ""
        )
    )

    print(
        f"DATA_DIR   : {data_dir} "
        f"-> exists: {os.path.isdir(data_dir)}"
    )

    print(f"CKPT_DIR   : {config.CKPT_DIR}")
    print(f"RESULT_DIR : {config.RESULT_DIR}")

    # Fold 0 검증 데이터 확인
    (_, _, _), (_, val_images, val_masks) = load_fold(
        0,
        data_dir,
        split_dir
    )

    model = build_unet(
        pretrained=pretrained
    ).to(device)

    model.eval()

    # 모델 출력 크기 확인
    with torch.no_grad():
        output = model(
            torch.randn(
                2, 3, 256, 256,
                device=device
            )
        )

    if output.shape != (2, 1, 256, 256):
        raise AssertionError(
            f"U-Net 출력 shape 오류: {output.shape}"
        )

    params = count_params(model)

    print(f"Model      : {MODEL_NAME}")
    print(f"Parameters : {params:,}")
    print(f"Output     : {tuple(output.shape)}")

    # 실제 검증 영상으로 예측 함수 확인
    predict = make_predict_fn(
        model,
        device=device,
        size=256,
        ellipse=True
    )

    pred = predict(val_images[0])

    if pred.shape != val_masks[0].shape:
        raise AssertionError(
            f"예측 마스크 크기 오류: {pred.shape}"
        )

    if not set(pd.unique(pred.ravel())).issubset({0, 1}):
        raise AssertionError(
            "예측 마스크가 이진값이 아닙니다."
        )

    print(f"Validation images : {len(val_images)}")
    print(f"Prediction shape  : {pred.shape}")
    print("U-Net 사전 점검 완료")

    del model, predict

    if device == "cuda":
        torch.cuda.empty_cache()

    return params


# ============================================================
# 6. 학습 실행
# ============================================================

def run(folds=None, **kwargs):
    """
    공통 train.py를 이용하여 U-Net ResNet34를 학습한다.
    """

    return train.run(
        model_name=MODEL_NAME,
        folds=folds,
        **kwargs
    )


# ============================================================
# 7. 5-Fold 결과 요약
# ============================================================

def summarize(result_dir=None):
    """
    5개 Fold의 검증 CSV를 불러와 평균 및 표준편차를 계산한다.
    """

    result_dir = result_dir or config.RESULT_DIR

    files = sorted(
        glob.glob(
            os.path.join(
                result_dir,
                f"{MODEL_NAME}__val__fold*.csv"
            )
        )
    )

    if len(files) != 5:
        print(
            f"Fold 결과 파일을 확인하세요. "
            f"현재 파일 개수: {len(files)}"
        )
        return None

    fold_means = []

    for file in files:
        df = pd.read_csv(file)

        fold_means.append({
            "IoU": df["iou"].mean(),
            "Dice": df["dice"].mean(),
            "Time(ms)": df["ms"].mean(),
            "HC_MAE(mm)": df["hc_err"].abs().mean()
        })

    fold_df = pd.DataFrame(fold_means)

    summary = pd.DataFrame({
        "평가 지표": [
            "IoU",
            "Dice",
            "Time(ms)",
            "HC_MAE(mm)"
        ],
        "평균": [
            fold_df["IoU"].mean(),
            fold_df["Dice"].mean(),
            fold_df["Time(ms)"].mean(),
            fold_df["HC_MAE(mm)"].mean()
        ],
        "표준편차": [
            fold_df["IoU"].std(),
            fold_df["Dice"].std(),
            fold_df["Time(ms)"].std(),
            fold_df["HC_MAE(mm)"].std()
        ]
    })

    print("\n[U-Net ResNet34 5-Fold 결과]")
    print(summary.round(4).to_string(index=False))

    return summary


# ============================================================
# 8. 학습 곡선 시각화
# ============================================================

def _plot_history(log_dir, out_path):
    """
    Fold별 학습 Loss 및 검증 Dice를 시각화한다.
    """

    import matplotlib.pyplot as plt

    history_files = sorted(
        glob.glob(
            os.path.join(
                log_dir,
                f"{MODEL_NAME}__fold*_history.csv"
            )
        )
    )

    if not history_files:
        print("학습 History CSV가 없습니다:", log_dir)
        return None

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4)
    )

    for file in history_files:
        name = os.path.basename(file)
        tag = name.split("__")[1].replace(
            "_history.csv", ""
        )

        df = pd.read_csv(file)

        axes[0].plot(
            df["epoch"],
            df["train_loss"],
            label=tag
        )

        axes[1].plot(
            df["epoch"],
            df["val_dice_256"],
            label=tag
        )

    axes[0].set_title("U-Net Train Loss")
    axes[1].set_title("U-Net Validation Dice")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return out_path


# ============================================================
# 9. 성공 및 실패 사례 시각화
# ============================================================

def _plot_cases(fold, result_dir, data_dir,
                split_dir, device, out_path):
    """
    해당 Fold의 IoU를 기준으로 성공 및 실패 사례를 선정하고
    원본, 정답, U-Net 예측 결과를 시각화한다.
    """

    import matplotlib.pyplot as plt

    ckpt_file = os.path.join(
        config.CKPT_DIR,
        MODEL_NAME,
        f"fold{fold}.pth"
    )

    val_csv = os.path.join(
        result_dir,
        f"{MODEL_NAME}__val__fold{fold}.csv"
    )

    if not (
        os.path.exists(ckpt_file)
        and os.path.exists(val_csv)
    ):
        print(f"Fold {fold}: 체크포인트 또는 CSV가 없습니다.")
        return None

    (_, _, _), (_, val_images, val_masks) = load_fold(
        fold,
        data_dir,
        split_dir
    )

    df = pd.read_csv(val_csv).reset_index(drop=True)

    # CSV 이미지 이름과 검증 영상 순서 확인
    if len(df) != len(val_images):
        raise ValueError(
            f"Fold {fold}: CSV와 검증 이미지 개수가 다릅니다."
        )

    image_to_idx = {
        name: idx
        for idx, name in enumerate(df["image"])
    }

    model = build_unet(
        pretrained=False
    ).to(device)

    ckpt = torch.load(
        ckpt_file,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 타원 보정 전후 예측 함수
    predict_seg = make_predict_fn(
        model,
        device=device,
        size=256,
        ellipse=False
    )

    predict_ellipse = make_predict_fn(
        model,
        device=device,
        size=256,
        ellipse=True
    )

    # IoU 기준 성공 및 실패 사례 선정
    sorted_idx = df["iou"].argsort().values

    k = min(3, len(sorted_idx))

    worst = list(sorted_idx[:k])
    best = list(sorted_idx[-k:][::-1])

    picks = (
        [(i, f"Worst {n + 1}") for n, i in enumerate(worst)]
        + [(i, f"Best {n + 1}") for n, i in enumerate(best)]
    )

    fig, axes = plt.subplots(
        len(picks),
        3,
        figsize=(12, 3.5 * len(picks)),
        squeeze=False
    )

    for row, (csv_idx, tag) in enumerate(picks):

        image_name = df.iloc[csv_idx]["image"]
        image_idx = image_to_idx[image_name]

        image = val_images[image_idx]
        gt = val_masks[image_idx]

        pred_seg = predict_seg(image)
        pred_ellipse = predict_ellipse(image)

        result = df.iloc[csv_idx]

        # 원본 영상
        axes[row, 0].imshow(
            image,
            cmap="gray"
        )

        axes[row, 0].set_title(
            f"{tag}: {image_name}"
        )

        # 정답 마스크와 보정 전 예측
        axes[row, 1].imshow(
            image,
            cmap="gray"
        )

        if gt.max() > 0:
            axes[row, 1].contour(
                gt,
                colors="lime",
                linewidths=1.2
            )

        if pred_seg.max() > 0:
            axes[row, 1].contour(
                pred_seg,
                colors="cyan",
                linewidths=1.2
            )

        axes[row, 1].set_title(
            "GT (green) vs Raw U-Net (cyan)"
        )

        # 정답 마스크와 타원 보정 후 예측
        axes[row, 2].imshow(
            image,
            cmap="gray"
        )

        if gt.max() > 0:
            axes[row, 2].contour(
                gt,
                colors="lime",
                linewidths=1.2
            )

        if pred_ellipse.max() > 0:
            axes[row, 2].contour(
                pred_ellipse,
                colors="red",
                linewidths=1.2
            )

        hc_text = (
            f" | HC err {result['hc_err']:.2f}mm"
            if "hc_err" in result and pd.notna(result["hc_err"])
            else ""
        )

        axes[row, 2].set_title(
            f"Ellipse U-Net | IoU {result['iou']:.3f}{hc_text}"
        )

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    del model, predict_seg, predict_ellipse

    if device == "cuda":
        torch.cuda.empty_cache()

    return out_path


# ============================================================
# 10. 분석 시각화 저장
# ============================================================

def save_analysis_plots(
    out_dir,
    fold=0,
    result_dir=None,
    data_dir=None,
    split_dir=None,
    device=None
):
    """
    학습 곡선과 성공/실패 사례 시각화 이미지를 저장한다.
    """

    os.makedirs(out_dir, exist_ok=True)

    result_dir = result_dir or config.RESULT_DIR
    data_dir = data_dir or config.DATA_DIR
    split_dir = split_dir or config.SPLIT_DIR

    device = _resolve_device(device)

    saved = []

    log_dir = os.path.join(
        result_dir,
        "logs"
    )

    history_png = _plot_history(
        log_dir,
        os.path.join(
            out_dir,
            "unet_resnet34_curves.png"
        )
    )

    if history_png:
        saved.append(history_png)

    case_png = _plot_cases(
        fold,
        result_dir,
        data_dir,
        split_dir,
        device,
        os.path.join(
            out_dir,
            f"unet_resnet34_fold{fold}_cases.png"
        )
    )

    if case_png:
        saved.append(case_png)

    return saved


# ============================================================
# 11. 실행 파이프라인
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="U-Net ResNet34 태아 머리 영상 분할 파이프라인"
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="학습할 Fold 번호 (예: 0 또는 0 1 2 3 4)"
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=256)

    parser.add_argument(
        "--no_aug",
        action="store_true",
        help="데이터 증강 없이 학습"
    )

    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--drive_root",
        default=None,
        help="체크포인트 및 결과 저장 경로"
    )

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split_dir", default=None)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--check",
        action="store_true",
        help="데이터 및 모델 사전 점검"
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
        help="5-Fold 결과 요약"
    )

    parser.add_argument(
        "--plot_dir",
        default=None,
        help="학습 곡선 및 성공/실패 사례 저장 폴더"
    )

    args = parser.parse_args()

    # 저장 경로 설정
    configure_drive_root(args.drive_root)

    # 모델 사전 점검
    if args.check:
        check_setup(
            pretrained=not args.no_pretrained,
            device=args.device,
            data_dir=args.data_dir,
            split_dir=args.split_dir
        )
        return

    # 학습 설정
    common_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        size=args.size,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        overwrite=args.overwrite,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        result_dir=args.result_dir,
        device=args.device
    )

    # 학습 실행
    if not (args.summarize or args.plot_dir):
        print("===== U-Net ResNet34 학습 시작 =====")
        print(f"Model : {MODEL_NAME}")
        print(f"Folds : {args.folds or '전체 Fold'}")

        run(
            folds=args.folds,
            aug=not args.no_aug,
            **common_kwargs
        )

    # 결과 요약
    if args.summarize:
        summarize(
            result_dir=args.result_dir
        )

    # 시각화
    if args.plot_dir:
        fold_for_plot = (
            args.folds[0]
            if args.folds
            else 0
        )

        saved = save_analysis_plots(
            args.plot_dir,
            fold=fold_for_plot,
            result_dir=args.result_dir,
            data_dir=args.data_dir,
            split_dir=args.split_dir,
            device=args.device
        )

        print("저장된 시각화:", saved)


if __name__ == "__main__":
    main()
