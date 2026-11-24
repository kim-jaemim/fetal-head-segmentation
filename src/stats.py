"""정답(GT)과 비교한 결과를 세 방법 공통으로 통계 처리한다 (공동 평가).

흐름:
  1. 각자 metrics.evaluate(..., return_per_image=True)로 이미지별 점수를 얻고
     save_per_image로 results/ 폴더(Drive)에 저장한다.
  2. 팀장이 load_results로 모든 방법의 결과를 모아 아래 함수들로 한 번에 비교한다.

저장 파일 이름 규칙: {method}__{split}__fold{fold}.csv
  method : 'classical', 'unet_resnet34', 'fcn_resnet50' 등
  split  : 'val' (폴드의 검증용, 교차검증 결과) 또는 'test' (최종 평가)
  fold   : 0 ~ K-1 (그 폴드로 학습한 모델), 학습이 없는 고전 기법은 폴드별 검증 결과를 같은 규칙으로 저장
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats as sps


# ---------------------------------------------------------------- 저장 / 모으기
def save_per_image(per, method, split, fold, out_dir):
    """이미지별 점수 표(evaluate의 return_per_image 결과)를 규칙에 맞춰 저장한다."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{method}__{split}__fold{fold}.csv")
    per.to_csv(path, index=False)
    return path


def load_results(out_dir, split):
    """out_dir에서 split('val' 또는 'test')의 모든 결과를 모아 한 표로 만든다. (method, fold 열 추가)"""
    frames = []
    for path in sorted(glob.glob(os.path.join(out_dir, f"*__{split}__fold*.csv"))):
        method, _, fold = os.path.basename(path)[:-4].split("__")
        d = pd.read_csv(path)
        d["method"] = method
        d["fold"] = fold.replace("fold", "")
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"{out_dir}에 {split} 결과가 없습니다.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 기술 통계
def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """평균의 95% 신뢰구간 (부트스트랩: 이미지를 복원추출해서 평균을 반복 계산)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def describe(df, metrics=("iou", "dice", "hc_err", "ms")):
    """방법별 평균, 표준편차, 중앙값, 95% 신뢰구간. 폴드 모델이 여러 개면 이미지별로 먼저 평균낸다."""
    rows = []
    per_img = df.groupby(["method", "image"], as_index=False)[[m for m in metrics if m in df]].mean()
    for method, g in per_img.groupby("method"):
        for m in metrics:
            if m not in g or g[m].isna().all():
                continue
            lo, hi = bootstrap_ci(g[m])
            rows.append({"method": method, "metric": m, "n": int(g[m].notna().sum()),
                         "mean": g[m].mean(), "std": g[m].std(), "median": g[m].median(),
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def fold_table(df, metric="iou"):
    """폴드별 평균 점수 표 + 폴드 간 평균과 표준편차 (모델이 데이터 분할에 얼마나 민감한지)."""
    t = df.groupby(["fold", "method"])[metric].mean().unstack("method")
    t.loc["mean"] = t.mean()
    t.loc["std"] = t.iloc[:-1].std()
    return t


# ---------------------------------------------------------------- 방법 간 비교
def pivot_metric(df, metric="iou"):
    """이미지 x 방법 표. 같은 이미지에 대한 점수를 방법끼리 나란히 놓는다(짝지은 비교용)."""
    return df.groupby(["image", "method"])[metric].mean().unstack("method")


def paired_compare(pivot, a, b):
    """같은 이미지에서 방법 a와 b를 비교: 평균 차이, 신뢰구간, Wilcoxon 부호순위 검정, a가 이긴 비율."""
    d = pivot[[a, b]].dropna()
    diff = d[a] - d[b]
    lo, hi = bootstrap_ci(diff)
    try:
        p = float(sps.wilcoxon(d[a], d[b]).pvalue)
    except ValueError:  # 모든 차이가 0인 경우
        p = 1.0
    return {"a": a, "b": b, "n": len(d), "mean_diff(a-b)": float(diff.mean()),
            "ci95_low": lo, "ci95_high": hi, "p_wilcoxon": p,
            "a_win_rate": float((diff > 0).mean())}


def compare_all(pivot):
    """모든 방법 쌍을 비교하고 다중비교 보정(Bonferroni)한 p값을 함께 낸다."""
    cols = list(pivot.columns)
    rows = [paired_compare(pivot, cols[i], cols[j])
            for i in range(len(cols)) for j in range(i + 1, len(cols))]
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p_wilcoxon"] * len(out)).clip(upper=1.0)
    return out


# ---------------------------------------------------------------- 머리 둘레(HC) 일치도
def bland_altman(per):
    """예측 HC와 정답 HC의 일치도: 편향(bias), 95% 일치한계, MAE, 상관계수. per는 한 방법의 결과."""
    d = per.dropna(subset=["hc_pred", "hc_gt"])
    diff = d["hc_pred"] - d["hc_gt"]
    bias, sd = float(diff.mean()), float(diff.std())
    r = float(np.corrcoef(d["hc_pred"], d["hc_gt"])[0, 1]) if len(d) > 2 else float("nan")
    return {"n": len(d), "bias_mm": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "mae_mm": float(diff.abs().mean()), "pearson_r": r}


def plot_bland_altman(ax, per, title=""):
    """Bland-Altman 그림: 가로 = 예측과 정답의 평균, 세로 = 차이(예측 - 정답)."""
    d = per.dropna(subset=["hc_pred", "hc_gt"])
    mean = (d["hc_pred"] + d["hc_gt"]) / 2
    diff = d["hc_pred"] - d["hc_gt"]
    s = bland_altman(per)
    ax.scatter(mean, diff, s=8, alpha=0.5)
    ax.axhline(s["bias_mm"], color="k")
    ax.axhline(s["loa_low"], color="r", ls="--")
    ax.axhline(s["loa_high"], color="r", ls="--")
    ax.set_xlabel("mean HC (mm)")
    ax.set_ylabel("pred - GT (mm)")
    ax.set_title(title)


def plot_box(pivot, ax, metric="IoU"):
    """방법별 이미지 점수 분포 (상자 그림). 평균만으로는 안 보이는 실패 사례의 꼬리를 보여준다."""
    pivot.plot(kind="box", ax=ax)
    ax.set_ylabel(metric)
