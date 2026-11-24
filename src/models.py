"""백본을 이름으로 골라 쓰는 모델 목록과, 모든 딥러닝 모델이 공유하는 predict_fn 생성기.

새 백본을 붙이려면 REGISTRY에 한 줄만 추가하면 된다. 모델 출력은 항상 (N, 1, H, W) 로짓이다.

설치 (Colab): pip install segmentation-models-pytorch
"""


def _unet(encoder):
    def build(pretrained=True):
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=encoder,
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=3, classes=1)
    return build


def _fcn(depth):
    def build(pretrained=True):
        import torch.nn as nn
        from torchvision.models import ResNet50_Weights, ResNet101_Weights
        from torchvision.models import segmentation as seg

        if depth == 50:
            fn, w = seg.fcn_resnet50, ResNet50_Weights.IMAGENET1K_V1
        else:
            fn, w = seg.fcn_resnet101, ResNet101_Weights.IMAGENET1K_V1
        net = fn(weights=None, weights_backbone=w if pretrained else None,
                 num_classes=1, aux_loss=False)

        class Wrap(nn.Module):  # torchvision 모델은 dict를 돌려주므로 텐서로 통일
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)["out"]

        return Wrap(net)
    return build


REGISTRY = {
    "unet_resnet18": _unet("resnet18"),
    "unet_resnet34": _unet("resnet34"),
    "unet_resnet50": _unet("resnet50"),
    "unet_efficientnet-b0": _unet("efficientnet-b0"),
    "fcn_resnet50": _fcn(50),
    "fcn_resnet101": _fcn(101),
}


def list_models():
    return sorted(REGISTRY)


def build_model(name, pretrained=True):
    """이름으로 모델을 만든다. 사전학습 가중치는 허용되지만, 학습 없이 그대로 쓰는 것은 불가."""
    if name not in REGISTRY:
        raise KeyError(f"알 수 없는 모델 '{name}'. 사용 가능: {list_models()}")
    return REGISTRY[name](pretrained)


def make_predict_fn(model, device="cuda", size=256, thr=0.5, ellipse=True):
    """모델 -> predict_fn (원본 540x800 흑백 이미지 -> 같은 크기 0/1 마스크).

    U-Net, FCN 모두 이 함수를 써서 전처리(dataset.preprocess_image)와 후처리(복원, 이진화, 타원 피팅)를 똑같이 맞춘다.
    ellipse=False로 만들면 타원 피팅 전 분할 마스크의 점수를 따로 잴 수 있다 (추가 분석용).
    """
    import cv2
    import numpy as np
    import torch

    from dataset import preprocess_image  # 학습(SegDataset)과 똑같은 전처리
    from metrics import fit_ellipse_mask

    model = model.to(device).eval()

    @torch.no_grad()
    def predict(img):
        h, w = img.shape
        t = torch.from_numpy(preprocess_image(img, size)).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()  # .cpu()가 GPU 작업을 마칠 때까지 기다려서 시간이 정확히 재진다
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        seg = (prob > thr).astype(np.uint8)
        return fit_ellipse_mask(seg) if ellipse else seg

    return predict
