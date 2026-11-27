
"""
U-Net 모델 정의
==================================================
1. 모델 설명
   - U-Net 구조 사용
   - Encoder: ResNet34
   - ImageNet 사전 학습 가중치 사용

2. 백본 선정 이유
   - 5-Fold Cross Validation 실험 결과를 비교
   - ResNet34는 IoU 0.9537, Dice 0.9745로
     네 가지 백본 중 가장 높은 Dice를 기록
   - ResNet50과 비슷한 분할 정확도를 보이면서
     처리 시간은 더 짧음
   - 정확도와 처리 속도의 균형을 고려하여 선정

3. 백본별 5-Fold 평균 실험 결과
   ------------------------------------------------
   Backbone          IoU      Dice     Time(ms)   HC_MAE(mm)
   ------------------------------------------------
   ResNet18          0.9532   0.9742    9.67       2.00
   ResNet34          0.9537   0.9745   12.31       1.98
   ResNet50          0.9537   0.9740   15.58       1.94
   EfficientNet-B0   0.9529   0.9741   16.66       2.12
   ------------------------------------------------

4. 평가 지표
   - IoU: 예측 마스크와 정답 마스크의 겹치는 정도
   - Dice: 예측 마스크와 정답 마스크의 유사도
   - Time: 이미지 한 장의 평균 처리 시간
   - HC_MAE: 머리둘레 평균 절대 오차(mm)

5. 참고
   - 위 실험 결과는 타원 보정 후 평가 결과
   - 학습 및 검증 설정은 실험 당시의 설정을 따름
"""

import segmentation_models_pytorch as smp


def build_unet(pretrained=True):
    """
    ResNet34 Encoder 기반 U-Net 모델 생성

    Args:
        pretrained (bool):
            ImageNet 사전 학습 가중치 사용 여부

    Returns:
        U-Net 모델
    """

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None
    )

    return model
