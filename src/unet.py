# U-Net 기반 태아 머리 영상 분할 모델

import segmentation_models_pytorch as smp


def build_unet(pretrained=True):
    """
    ResNet34를 Encoder로 사용하는 U-Net 모델 생성

    Args:
        pretrained (bool):
            True이면 ImageNet 사전 학습 가중치 사용

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
