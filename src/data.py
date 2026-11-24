"""데이터 로딩 공통 코드 (HC18).

팀원 전원이 같은 정답 마스크, 같은 분할을 쓰도록 여기서만 불러온다.
"""
import os

import cv2
import numpy as np
import pandas as pd

IMG_SHAPE = (540, 800)  # (높이, 너비). HC18 원본 크기


def fill_mask(mask):
    """정답 마스크(타원 테두리) -> 속이 찬 0/1 마스크.

    HC18의 annotation은 타원의 '테두리'만 그려져 있어서 그대로는 분할 정답으로 쓸 수 없다.
    테두리의 바깥 윤곽을 찾아 안쪽을 채운다.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def list_pairs(data_dir):
    """data_dir 안의 (이미지, 정답 마스크) 짝을 표로 만든다.

    파일명 예: 001_HC.png (이미지), 001_HC_Annotation.png (마스크), 001_2HC.png (같은 환자의 두 번째 촬영)
    patient 열 = 파일명 앞 3자리(환자 번호). 같은 환자의 사진은 여러 장일 수 있다.
    """
    files = sorted(os.listdir(data_dir))
    file_set = set(files)
    imgs = [f for f in files if f.endswith(".png") and not f.endswith("_Annotation.png")]
    pairs = [(f, f.replace(".png", "_Annotation.png")) for f in imgs]
    missing = [m for _, m in pairs if m not in file_set]
    if missing:
        raise FileNotFoundError(f"정답 마스크가 없는 이미지 {len(missing)}장: {missing[:3]} ...")
    df = pd.DataFrame(pairs, columns=["image", "mask"])
    df["patient"] = df["image"].str.split("_").str[0]
    return df


def load_split(name, data_dir, split_dir, allow_test=False):
    """분할 csv를 읽어 이미지와 (채워진) 정답 마스크를 불러온다.

    name: 'train' | 'val' | 'test'
    반환: (표, 이미지 리스트, 정답 마스크 리스트)  -- 이미지는 (540, 800) uint8 흑백

    test는 최종 평가 때 팀장만 한 번 쓰는 데이터라서 allow_test=True를 명시해야 불러와진다.
    """
    if name not in ("train", "val", "test"):
        raise ValueError("name은 'train', 'val', 'test' 중 하나여야 합니다.")
    if name == "test" and not allow_test:
        raise PermissionError(
            "test 세트는 최종 평가 때 팀장만 한 번 사용합니다. "
            "파라미터 조정과 모델 선택은 val 세트로 하세요."
        )
    d = pd.read_csv(os.path.join(split_dir, f"{name}.csv"))
    return _read_rows(d, data_dir)


def _read_rows(d, data_dir):
    """표의 각 행(image, mask)을 읽어 (표, 이미지 리스트, 채워진 정답 마스크 리스트)로 반환."""
    d = d.reset_index(drop=True)
    imgs, gts = [], []
    for _, r in d.iterrows():
        img = cv2.imread(os.path.join(data_dir, r["image"]), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(os.path.join(data_dir, r["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            raise FileNotFoundError(f"읽을 수 없는 파일: {r['image']} / {r['mask']}")
        imgs.append(img)
        gts.append(fill_mask(m))
    return d, imgs, gts


def load_fold(fold, data_dir, split_dir):
    """K-폴드 교차검증용: fold번 폴드를 검증용으로, 나머지 폴드를 학습용으로 불러온다.

    splits/folds.csv(개발용 데이터 + fold 열)를 사용한다. test 세트는 포함되지 않는다.
    반환: (train_표, train_이미지, train_정답), (val_표, val_이미지, val_정답)
    """
    f = pd.read_csv(os.path.join(split_dir, "folds.csv"))
    k = int(f["fold"].max()) + 1
    if not 0 <= fold < k:
        raise ValueError(f"fold는 0 ~ {k - 1} 사이여야 합니다.")
    tr = _read_rows(f[f["fold"] != fold], data_dir)
    va = _read_rows(f[f["fold"] == fold], data_dir)
    return tr, va


def n_folds(split_dir):
    """folds.csv에 들어 있는 폴드 수 K."""
    f = pd.read_csv(os.path.join(split_dir, "folds.csv"))
    return int(f["fold"].max()) + 1
