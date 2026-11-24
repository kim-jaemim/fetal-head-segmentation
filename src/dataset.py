"""학습용 데이터셋과 공통 전처리.

학습(SegDataset)과 추론(models.make_predict_fn)이 같은 전처리 함수를 쓰므로
"학습할 때와 평가할 때 입력이 다르게 만들어지는" 실수를 막는다.

- 증강(augment)은 원본 해상도(540x800)에서 먼저 적용하고, 그 다음 256x256으로 줄인다.
- 검증용 데이터셋은 augment=None으로 만든다.
"""
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet 평균/표준편차 (사전학습 인코더 기준)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img, size=256):
    """(H, W) uint8 흑백 -> (3, size, size) float32. 흑백을 3채널로 복제하고 정규화한다."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (np.stack([x, x, x], axis=-1) - MEAN) / STD
    return x.transpose(2, 0, 1).astype(np.float32)


def preprocess_mask(mask, size=256):
    """(H, W) 0/1 -> (1, size, size) float32. 마스크는 값이 섞이지 않게 최근접 보간으로 줄인다."""
    m = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return m[None].astype(np.float32)


class SegDataset:
    """PyTorch DataLoader에 바로 넣을 수 있는 분할 데이터셋 (__len__, __getitem__ 구현).

    images, masks: data.load_fold / load_split이 돌려준 리스트
    augment      : augment.Augmenter 또는 None (검증·테스트는 None)
    """

    def __init__(self, images, masks, size=256, augment=None):
        assert len(images) == len(masks)
        self.images, self.masks = images, masks
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.images)

    def get_arrays(self, i):
        img, m = self.images[i], self.masks[i]
        if self.augment is not None:
            img, m = self.augment(img, m)
        return preprocess_image(img, self.size), preprocess_mask(m, self.size)

    def __getitem__(self, i):
        import torch
        x, y = self.get_arrays(i)
        return torch.from_numpy(x), torch.from_numpy(y)
