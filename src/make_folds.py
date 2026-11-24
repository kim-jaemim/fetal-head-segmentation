"""개발용 데이터(train + val)를 환자 단위 K-폴드로 나눠 splits/folds.csv로 저장한다.

- test.csv는 그대로 둔다. 모든 방법이 같은 테스트 세트로 최종 비교되어야 하기 때문이다.
- 한 환자의 사진은 한 폴드에만 들어간다 (환자 단위 분할, 데이터 누수 방지).
- 환자 번호를 시드로 섞은 뒤 K개로 나눠서, 폴드마다 환자 수가 거의 같다.
- 이미 folds.csv가 있으면 덮어쓰지 않는다 (--force로만 덮어씀). 한 번 만들고 팀 전원이 그대로 쓴다.

사용 예:
    python src/make_folds.py --split_dir splits --k 5
"""
import argparse
import os

import numpy as np
import pandas as pd


def make_folds(dev, k=5, seed=42):
    """dev: image, mask, patient 열이 있는 표 -> fold 열(0 ~ k-1)을 추가해서 반환."""
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(dev["patient"].unique()))  # 정렬 후 섞어서 입력 순서와 무관하게 재현
    rng.shuffle(patients)
    fold_of = {}
    for fold, group in enumerate(np.array_split(patients, k)):
        for p in group:
            fold_of[p] = fold

    out = dev.copy()
    out["fold"] = out["patient"].map(fold_of)
    # 같은 환자가 두 폴드에 섞이지 않았는지 검증
    assert out.groupby("patient")["fold"].nunique().max() == 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = os.path.join(args.split_dir, "folds.csv")
    if os.path.exists(out_path) and not args.force:
        print("이미 folds.csv가 있습니다. 다시 만들지 않습니다. (덮어쓰려면 --force)")
        return

    train = pd.read_csv(os.path.join(args.split_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.split_dir, "val.csv"))
    test = pd.read_csv(os.path.join(args.split_dir, "test.csv"))
    dev = pd.concat([train, val], ignore_index=True)

    # test 환자가 개발용 데이터에 섞여 있지 않은지 확인
    assert not set(dev["patient"]) & set(test["patient"]), "test 환자가 개발용 데이터에 섞여 있습니다."

    folds = make_folds(dev, k=args.k, seed=args.seed)
    print(f"개발용 {len(folds)}장 / {folds['patient'].nunique()}명 -> {args.k}폴드")
    print(folds.groupby("fold").agg(images=("image", "size"), patients=("patient", "nunique")).to_string())
    folds.to_csv(out_path, index=False)
    print("저장 완료:", out_path)


if __name__ == "__main__":
    main()
