
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
from pathlib import Path

import torch
import pandas as pd

import models
import train
import config


MODEL_NAME = "unet_resnet34"


# 1. U-Net 모델 생성
def build_unet(pretrained=True):
    return models.build_model(
        MODEL_NAME,
        pretrained=pretrained
    )


# 2. 모델 실행 점검
def check_setup():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_unet().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        output = model(x)

    print("모델 점검 완료")
    print("사용 장치:", device)
    print("입력 크기:", x.shape)
    print("출력 크기:", output.shape)
    print("파라미터 수:", sum(p.numel() for p in model.parameters()))


# 3. U-Net 학습
def run(folds, **kwargs):
    train.run(
        model_name=MODEL_NAME,
        folds=folds,
        **kwargs
    )


# 4. 5-Fold 결과 요약
def summarize():
    result_dir = Path(config.RESULT_DIR)
    files = sorted(result_dir.glob(f"{MODEL_NAME}__val__fold*.csv"))

    if not files:
        print("결과 CSV 파일을 찾을 수 없습니다.")
        print("결과 경로:", result_dir)
        return

    results = []

    for file in files:
        df = pd.read_csv(file)

        metric_columns = {
            "IoU": ["IoU", "iou"],
            "Dice": ["Dice", "dice"],
            "HC_MAE": ["HC_err_mm", "HC_MAE", "hc_err_mm"],
        }

        row = {"Fold": file.stem.split("fold")[-1]}

        for metric, candidates in metric_columns.items():
            column = next(
                (name for name in candidates if name in df.columns),
                None
            )

            if column is not None:
                row[metric] = df[column].mean()

        results.append(row)

    summary = pd.DataFrame(results)

    print("\nFold별 결과")
    print(summary.to_string(index=False))

    print("\n전체 평균")
    print(summary.select_dtypes(include="number").mean())

    print("\n표준편차")
    print(summary.select_dtypes(include="number").std())


# 5. 실행 설정
def main():
    parser = argparse.ArgumentParser(
        description="U-Net ResNet34 학습 및 평가"
    )

    parser.add_argument(
        "--folds", nargs="+", type=int,
        default=[0, 1, 2, 3, 4]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summarize", action="store_true")

    args = parser.parse_args()

    if args.check:
        check_setup()

    elif args.summarize:
        summarize()

    else:
        run(
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            size=args.size,
            seed=args.seed,
            num_workers=args.num_workers,
            aug=not args.no_aug,
            pretrained=not args.no_pretrained,
            overwrite=args.overwrite
        )


if __name__ == "__main__":
    main()
