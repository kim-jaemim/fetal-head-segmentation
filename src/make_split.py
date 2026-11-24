"""환자 단위 학습/검증/테스트 분할을 만들어 splits/*.csv로 저장한다.

한 번만 실행하는 스크립트다. 이미 csv가 있으면 덮어쓰지 않는다(--force로만 덮어씀).
분할이 바뀌면 팀원들의 결과를 비교할 수 없으므로 팀원은 이 스크립트를 다시 돌리지 않는다.

사용 예:
    python src/make_split.py --data_dir /content/data/training_set/training_set --split_dir splits
"""
import argparse
import os

import numpy as np

from data import list_pairs


def make_split(df, seed=42, ratios=(0.70, 0.15)):
    """환자 번호를 섞어서 학습/검증/테스트로 나눈다 (한 환자의 사진은 한 세트에만 들어감)."""
    rng = np.random.default_rng(seed)
    patients = df["patient"].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    split_of = {}
    for i, p in enumerate(patients):
        if i < n_train:
            split_of[p] = "train"
        elif i < n_train + n_val:
            split_of[p] = "val"
        else:
            split_of[p] = "test"

    out = df.copy()
    out["split"] = out["patient"].map(split_of)
    # 같은 환자가 두 세트에 섞이지 않았는지 검증
    assert out.groupby("patient")["split"].nunique().max() == 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="이미 있는 csv를 덮어쓴다 (보통 쓰지 않음)")
    args = ap.parse_args()

    os.makedirs(args.split_dir, exist_ok=True)
    existing = [s for s in ("train", "val", "test")
                if os.path.exists(os.path.join(args.split_dir, f"{s}.csv"))]
    if existing and not args.force:
        print(f"이미 분할 파일이 있습니다 {existing}. 다시 만들지 않습니다. (덮어쓰려면 --force)")
        return

    df = make_split(list_pairs(args.data_dir), seed=args.seed)
    print("이미지 수:\n", df["split"].value_counts().to_string())
    print("환자 수:\n", df.groupby("split")["patient"].nunique().to_string())

    for s in ("train", "val", "test"):
        df[df["split"] == s][["image", "mask", "patient"]].to_csv(
            os.path.join(args.split_dir, f"{s}.csv"), index=False)
    print("저장 완료:", args.split_dir)


if __name__ == "__main__":
    main()
