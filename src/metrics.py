"""평가 공통 코드: IoU, Dice, 처리 시간, 파라미터 수, 타원 피팅, 머리 둘레(HC) 오차.

세 방법(고전, U-Net, FCN)이 모두 이 코드로 같은 방식으로 채점된다.
각자 지표 코드를 따로 만들지 말고 여기 함수를 쓴다.
"""
import time

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 분할 점수
def iou_dice(pred, gt, eps=1e-7):
    """같은 크기의 0/1 마스크 두 개 -> (IoU, Dice). 이미지 한 장 기준."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    return iou, dice


# ---------------------------------------------------------------- 타원 피팅 (공통)
def fit_ellipse_mask(mask):
    """분할 마스크(0/1) -> 가장 큰 덩어리에 타원을 맞춰 속이 찬 타원 마스크(0/1)로 반환.

    U-Net, FCN은 분할 후 이 함수를 마지막에 호출한다 (분할 후 타원 피팅).
    마스크가 비었거나 점이 5개 미만이면 빈 마스크를 반환한다.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) >= 5]  # fitEllipse는 점이 5개 이상 필요
    out = np.zeros_like(mask)
    if not contours:
        return out
    c = max(contours, key=cv2.contourArea)  # 가장 큰 덩어리를 머리로 간주
    cv2.ellipse(out, cv2.fitEllipse(c), 1, thickness=-1)
    return out


# ---------------------------------------------------------------- 머리 둘레 (HC)
def hc_from_mask(mask, pixel_size_mm):
    """0/1 마스크 -> 타원 피팅 -> 둘레(mm). 측정할 수 없으면 None."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) >= 5]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.fitEllipse(c)  # w, h = 타원의 전체 축 길이(픽셀)
    a, b = w / 2, h / 2  # 반지름
    perimeter_px = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))  # Ramanujan 근사
    return perimeter_px * pixel_size_mm


def load_hc_table(csv_path):
    """training_set_pixel_size_and_HC.csv -> 파일명으로 찾는 표 (열: pixel_mm, hc_mm)."""
    t = pd.read_csv(csv_path)
    t = t.rename(columns=dict(zip(t.columns[:3], ["filename", "pixel_mm", "hc_mm"])))
    return t.set_index("filename")


# ---------------------------------------------------------------- 전체 평가
def evaluate(predict_fn, images, gts, split_df=None, hc_table=None,
             warmup=3, return_per_image=False):
    """predict_fn으로 모든 이미지를 채점한다.

    predict_fn : 원본 크기(540x800) 흑백 이미지 1장 -> 같은 크기의 0/1 마스크
                 (GPU를 쓰면 함수 안에서 결과를 CPU numpy로 바꿔 돌려줄 것: 시간이 정확히 재진다)
    images, gts: 같은 순서의 리스트 (data.load_split의 결과)
    split_df   : load_split이 돌려준 표 (이미지 이름을 per-image 결과에 붙일 때, HC 계산 시 필요)
    hc_table   : load_hc_table 결과. 주면 HC 오차(mm)도 계산한다.
    warmup     : 시간 측정 전에 미리 돌려보는 장수 (GPU 첫 호출이 느린 영향 제거)
    반환       : {'IoU', 'Dice', 'ms_per_image', ('HC_MAE_mm', 'HC_fail')}
                 return_per_image=True면 (요약, 이미지별 표)로 반환 -> 실패 사례 찾기에 쓴다.
    """
    if hc_table is not None and split_df is None:
        raise ValueError("hc_table을 쓰려면 split_df도 함께 넘겨야 합니다.")

    for img in images[:warmup]:
        predict_fn(img)

    rows = []
    for k, (img, gt) in enumerate(zip(images, gts)):
        t0 = time.perf_counter()
        pred = predict_fn(img)
        ms = (time.perf_counter() - t0) * 1000
        pred = (np.asarray(pred) > 0).astype(np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f"예측 크기 {pred.shape} != 정답 크기 {gt.shape}. "
                             "predict_fn은 원본 크기 마스크를 돌려줘야 합니다.")
        iou, dice = iou_dice(pred, gt)
        row = {"iou": iou, "dice": dice, "ms": ms}
        if split_df is not None:
            name = split_df.iloc[k]["image"]
            row["image"] = name
            if hc_table is not None:
                gt_hc = hc_table.loc[name, "hc_mm"]
                hc = hc_from_mask(pred, hc_table.loc[name, "pixel_mm"])
                row["hc_gt"] = gt_hc
                row["hc_pred"] = np.nan if hc is None else hc
                row["hc_err"] = np.nan if hc is None else abs(hc - gt_hc)
        rows.append(row)

    per = pd.DataFrame(rows)
    summary = {"IoU": float(per["iou"].mean()),
               "Dice": float(per["dice"].mean()),
               "ms_per_image": float(per["ms"].mean())}
    if hc_table is not None:
        summary["HC_MAE_mm"] = float(per["hc_err"].mean())  # 측정 실패는 제외하고 평균
        summary["HC_fail"] = int(per["hc_err"].isna().sum())
    return (summary, per) if return_per_image else summary


def count_params(model):
    """딥러닝 모델의 파라미터 수 (PyTorch)."""
    return sum(p.numel() for p in model.parameters())
