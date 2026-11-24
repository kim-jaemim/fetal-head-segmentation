"""학습용 데이터 증강 (U-Net, FCN 공통). 학습 폴드에만 적용하고, 검증·테스트에는 쓰지 않는다.

증강 4가지 (순서대로 적용):
  1. 강체 회전 + 평행 이동   : 이미지와 마스크에 같은 변환 (크기는 변하지 않음)
  2. 좌우 반전              : 이미지와 마스크에 같은 변환
  3. 밝기 / 대비 / 감마       : 이미지에만 적용 (마스크는 그대로)
  4. 스페클 노이즈 + 블러     : 이미지에만 적용 (마스크는 그대로)

매 호출마다 새로 랜덤하게 변환하므로(에폭마다 다른 모양) 증강본을 파일로 저장할 필요가 없다.
입력/출력: img (H, W) uint8 흑백, mask (H, W) 0/1 uint8 -> 같은 형태로 반환.
"""
import os

import cv2
import numpy as np


class Augmenter:
    def __init__(self,
                 rot=15, shift=0.08, min_keep=0.9,               # 1. 회전 최대 각도(도), 이동 최대 비율, 머리 영역 최소 보존 비율
                 p_flip=0.5,                                      # 2. 좌우 반전 확률
                 p_gamma=0.5, gamma=(0.7, 1.5),                   # 3. 감마
                 p_contrast=0.5, contrast=(0.8, 1.2),             #    대비
                 p_brightness=0.5, brightness=(-0.1, 0.1),        #    밝기 (0~1 스케일 기준 가감)
                 p_speckle=0.5, speckle=(0.02, 0.08),             # 4. 스페클 노이즈 세기(곱셈 노이즈의 표준편차)
                 p_blur=0.3, blur_sigma=(0.3, 1.2),               #    가우시안 블러 시그마
                 seed=None):
        self.rot, self.shift, self.min_keep = rot, shift, min_keep
        self.p_flip = p_flip
        self.p_gamma, self.gamma = p_gamma, gamma
        self.p_contrast, self.contrast = p_contrast, contrast
        self.p_brightness, self.brightness = p_brightness, brightness
        self.p_speckle, self.speckle = p_speckle, speckle
        self.p_blur, self.blur_sigma = p_blur, blur_sigma
        self.seed = seed
        self._rng_obj, self._pid = None, None

    def _rng(self):
        # DataLoader의 여러 worker(프로세스)가 똑같은 난수를 쓰지 않도록 프로세스마다 따로 만든다
        pid = os.getpid()
        if self._rng_obj is None or self._pid != pid:
            self._rng_obj = np.random.default_rng(None if self.seed is None else [self.seed, pid])
            self._pid = pid
        return self._rng_obj

    # ------------------------------------------------------------ 1. 강체 회전 + 이동
    def _rigid(self, img, mask, rng):
        h, w = img.shape
        area0 = mask.sum()
        for _ in range(5):  # 머리가 화면 밖으로 너무 잘리면 다시 뽑는다
            angle = rng.uniform(-self.rot, self.rot)
            tx = rng.uniform(-self.shift, self.shift) * w
            ty = rng.uniform(-self.shift, self.shift) * h
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)  # 크기 1.0 = 강체(크기 불변)
            M[:, 2] += (tx, ty)
            m2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if area0 == 0 or m2.sum() >= self.min_keep * area0:
                i2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                return i2, m2
        return img, mask  # 5번 다 실패하면 기하 변환 없이 반환

    # ------------------------------------------------------------ 3. 밝기 / 대비 / 감마
    def _photometric(self, x, rng):
        if rng.random() < self.p_gamma:
            x = np.power(x, rng.uniform(*self.gamma))
        if rng.random() < self.p_contrast:
            x = (x - 0.5) * rng.uniform(*self.contrast) + 0.5
        if rng.random() < self.p_brightness:
            x = x + rng.uniform(*self.brightness)
        return np.clip(x, 0.0, 1.0)

    # ------------------------------------------------------------ 4. 스페클 + 블러
    def _noise_blur(self, x, rng):
        if rng.random() < self.p_blur:  # 약한 가우시안 블러 (장비 화질 차이)
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(*self.blur_sigma))
        if rng.random() < self.p_speckle:  # 스페클은 곱셈 노이즈: 밝은 곳일수록 흔들림이 크다
            x = x * (1.0 + rng.uniform(*self.speckle) * rng.standard_normal(x.shape).astype(np.float32))
        return np.clip(x, 0.0, 1.0)

    def __call__(self, img, mask):
        rng = self._rng()
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask.astype(np.uint8))

        img, mask = self._rigid(img, mask, rng)                      # 1
        if rng.random() < self.p_flip:                               # 2
            img, mask = img[:, ::-1], mask[:, ::-1]

        x = img.astype(np.float32) / 255.0
        x = self._photometric(x, rng)                                # 3
        x = self._noise_blur(x, rng)                                 # 4
        out = np.rint(x * 255.0).astype(np.uint8)
        return np.ascontiguousarray(out), np.ascontiguousarray(mask)


def preview(img, mask, aug=None, n=5, seed=0, save_path=None):
    """원본 1장과 증강 n장을 나란히 그려서(정답 윤곽 = 초록) 마스크가 이미지와 맞는지 눈으로 확인한다."""
    import matplotlib.pyplot as plt

    aug = aug or Augmenter(seed=seed)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))
    for k, ax in enumerate(axes):
        i, m = (img, mask) if k == 0 else aug(img, mask)
        ax.imshow(i, cmap="gray", vmin=0, vmax=255)
        ax.contour(m, colors="lime", linewidths=1)
        ax.set_title("original" if k == 0 else f"aug {k}")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    return fig
