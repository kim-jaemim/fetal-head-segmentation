
"""
============================================================
Attention U-Net 태아 머리 영상 분할 모델 & 실행 파이프라인
============================================================

1. Attention U-Net 모델 구성
------------------------------------------------------------
- 모델 구조: Attention U-Net
- Encoder : ResNet34 (ImageNet 사전 학습 가중치 사용)
- Decoder : Attention Gate + Convolution Block
- Skip Connection : Encoder의 특징을 Attention Gate로 전달
- 입력 채널 : 3 (흑백 초음파 영상을 3채널로 변환)
- 출력 채널 : 1 (태아 머리 영역의 이진 분할 로짓)
- 입력 크기 : 256 x 256
- 출력 크기 : (N, 1, 256, 256)

2. 공정 비교 및 실험 설계
------------------------------------------------------------
- 기존 U-Net과 동일한 공통 학습/평가 코드 사용
  (config.py, data.py, dataset.py, augment.py,
   models.py, train.py, metrics.py)
- 데이터 분할 : 환자 단위 5-Fold 교차검증
- 학습 설정 : BCE + Dice Loss, AdamW,
             CosineAnnealingLR, 30 epochs,
             batch_size=16, learning rate=3e-4
- 기본 실험 : Attention U-Net ResNet34
- 모델 이름 : attention_unet_resnet34
- 기존 U-Net 모델 및 결과는 수정하지 않음

3. Attention U-Net 실험 결과 (Fold 0)
------------------------------------------------------------
- IoU       : 0.9580
- Dice      : 0.9783
- HC_MAE    : 1.7935 mm
- Parameters: 24,563,693
- Validation images: 170

※ 위 수치는 Fold 0 실험 결과이며,
   전체 5-Fold 평균 결과가 아님.

4. 실행 예시
------------------------------------------------------------
- 모델 사전 점검:
  python src/unet.py --check

- Fold 0 학습:
  python src/unet.py --folds 0

- 전체 5-Fold 학습:
  python src/unet.py --folds 0 1 2 3 4
============================================================
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
import models
import train

from data import load_fold
from metrics import count_params


# ============================================================
# 1. Attention Gate
# ============================================================

class AttentionGate(nn.Module):
    """
    Encoder 특징과 Decoder 특징을 이용하여
    중요한 영역에 가중치를 부여하는 모듈
    """

    def __init__(self, skip_channels, gate_channels, inter_channels):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1,
                size=x1.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        attention = self.relu(g1 + x1)
        attention = self.psi(attention)

        return x * attention


# ============================================================
# 2. Decoder Block
# ============================================================

class DecoderBlock(nn.Module):
    """
    Attention Gate를 적용한 Encoder 특징과
    Decoder 특징을 결합하는 모듈
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.attention = AttentionGate(
            skip_channels,
            in_channels,
            max(out_channels // 2, 1)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        skip = self.attention(skip, x)
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


# ============================================================
# 3. Attention U-Net 모델
# ============================================================

class AttentionUNet(nn.Module):
    """
    ResNet34 Encoder와 Attention Decoder를 결합한
    태아 머리 영상 분할 모델
    """

    def __init__(self, pretrained=True, classes=1):
        super().__init__()

        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet" if pretrained else None
        )

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        self.segmentation_head = nn.Conv2d(
            64,
            classes,
            kernel_size=1
        )

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.encoder(x)

        x = features[-1]

        x = self.decoder4(x, features[-2])
        x = self.decoder3(x, features[-3])
        x = self.decoder2(x, features[-4])
        x = self.decoder1(x, features[-5])

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return self.segmentation_head(x)


# ============================================================
# 4. 공통 모델 Registry 등록
# ============================================================

MODEL_NAME = "attention_unet_resnet34"


def register_model():
    """
    공통 models.py의 모델 목록에 Attention U-Net 등록
    """

    models.REGISTRY[MODEL_NAME] = (
        lambda pretrained=True: AttentionUNet(
            pretrained=pretrained
        )
    )


register_model()


# ============================================================
# 5. 모델 사전 점검
# ============================================================

def check_setup(pretrained=True, device=None):
    """
    Attention U-Net의 입출력 크기와 파라미터 수를 확인한다.
    """

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device     : {device}")
    print(f"DATA_DIR   : {config.DATA_DIR}")
    print(f"RESULT_DIR : {config.RESULT_DIR}")
    print(f"CKPT_DIR   : {config.CKPT_DIR}")

    model = AttentionUNet(pretrained=pretrained).to(device)
    model.eval()

    with torch.no_grad():
        output = model(
            torch.randn(2, 3, 256, 256, device=device)
        )

    if output.shape != (2, 1, 256, 256):
        raise AssertionError(
            f"출력 크기 오류: {output.shape}"
        )

    params = count_params(model)

    print(f"Model      : {MODEL_NAME}")
    print(f"Parameters : {params:,}")
    print(f"Output     : {tuple(output.shape)}")

    del model

    return params


# ============================================================
# 6. 학습 실행
# ============================================================

def run(folds=None, **kwargs):
    """
    공통 train.py를 이용하여 Attention U-Net을 학습한다.
    """

    register_model()

    return train.run(
        model_name=MODEL_NAME,
        folds=folds,
        **kwargs
    )


# ============================================================
# 7. 실행 파이프라인
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Attention U-Net 태아 머리 영상 분할"
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0],
        help="학습할 Fold 번호 (예: 0 또는 0 1 2 3 4)"
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--no_pretrained",
        action="store_true"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="모델 사전 점검만 실행"
    )

    parser.add_argument(
        "--device",
        default=None
    )

    args = parser.parse_args()

    if args.check:
        check_setup(
            pretrained=not args.no_pretrained,
            device=args.device
        )
        return

    print("===== Attention U-Net 학습 시작 =====")
    print(f"Model : {MODEL_NAME}")
    print(f"Folds : {args.folds}")

    result = run(
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        size=args.size,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        overwrite=args.overwrite,
        device=args.device
    )

    print("\n===== Attention U-Net 학습 결과 =====")
    print(result)


if __name__ == "__main__":
    main()
