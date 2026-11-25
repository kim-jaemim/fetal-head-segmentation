"""U-Net / FCN 폴드별 학습 스크립트 (모든 딥러닝 모델 공통).

U-Net과 FCN이 같은 학습 설정(손실, 옵티마이저, 에폭, 증강)을 쓰도록 여기서만 학습한다.
증강(augment.Augmenter)은 학습 폴드에만 기본으로 적용되고, 검증은 원본으로 한다.

한 폴드에서 하는 일:
  1. 학습 폴드로 학습 -> 폴드 검증 Dice(256x256)가 가장 높은 에폭의 체크포인트를 Drive에 저장
  2. 그 체크포인트로 원본 크기 검증 이미지를 평가 (IoU, Dice, 시간, HC 오차, 파라미터 수)
  3. 이미지별 결과를 results/ 에 저장 -> stats.py로 방법 간 비교
     - {method}__val__fold{k}.csv       : 최종 출력(타원 마스크) 기준 = 공식 점수
     - {method}-seg__val__fold{k}.csv   : 타원 피팅 전 분할 마스크 기준 = 추가 분석용

test 세트는 이 스크립트에서 절대 불러오지 않는다 (최종 평가는 확정 후 팀장이 따로 한 번만).

노트북에서:
    import train
    train.run('unet_resnet34', folds=[0], epochs=30)        # fold 0만 먼저 시험
    train.run('unet_resnet34')                              # 전체 폴드
    train.run('unet_resnet34', aug=False)                   # 증강 없이 (증강 효과 비교용, method 이름에 -noaug 붙음)

터미널(!python)에서:
    python src/train.py --model fcn_resnet50 --folds 0 1 2 3 4
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import models
from augment import Augmenter
from data import load_fold, n_folds
from dataset import SegDataset
from metrics import count_params, evaluate, load_hc_table
from stats import save_per_image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits, target, eps=1.0):
    """부드러운 Dice 손실 (예측 확률과 정답이 많이 겹칠수록 작아짐)."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


@torch.no_grad()
def val_dice(model, loader, device):
    """폴드 검증 Dice (256x256, 임계값 0.5). 에폭마다 최고 모델을 고르는 기준."""
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        scores.append(((2 * inter + 1e-7) / (den + 1e-7)).cpu())
    return torch.cat(scores).mean().item()


def method_name(model_name, aug):
    return model_name if aug else f"{model_name}-noaug"


def train_fold(model_name, fold, epochs=30, batch_size=16, lr=3e-4, weight_decay=1e-4,
               size=256, aug=True, seed=42, num_workers=2, pretrained=True,
               overwrite=False, data_dir=None, split_dir=None, result_dir=None, device=None):
    data_dir = data_dir or config.DATA_DIR
    split_dir = split_dir or config.SPLIT_DIR
    result_dir = result_dir or config.RESULT_DIR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    method = method_name(model_name, aug)

    res_path = os.path.join(result_dir, f"{method}__val__fold{fold}.csv")
    if os.path.exists(res_path) and not overwrite:
        print(f"[{method} fold{fold}] 결과가 이미 있어 건너뜁니다 (다시 하려면 overwrite=True)")
        return None

    set_seed(seed + fold)
    (tr_d, tr_i, tr_g), (va_d, va_i, va_g) = load_fold(fold, data_dir, split_dir)
    print(f"[{method} fold{fold}] 학습 {len(tr_i)}장 / 검증 {len(va_i)}장 / device={device} / 증강={'적용' if aug else '없음'}")

    train_ds = SegDataset(tr_i, tr_g, size=size, augment=Augmenter(seed=seed) if aug else None)
    val_ds = SegDataset(va_i, va_g, size=size, augment=None)  # 검증은 항상 증강 없음
    gen = torch.Generator()
    gen.manual_seed(seed + fold)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin, generator=gen,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    model = models.build_model(model_name, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # 오래된 torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt = config.ckpt_path(method, fold)
    best, best_epoch, history = -1.0, 0, []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        vd = val_dice(model, val_loader, device)
        if vd > best:
            best, best_epoch = vd, epoch
            torch.save({"state_dict": model.state_dict(), "model_name": model_name, "epoch": epoch,
                        "val_dice_256": vd, "size": size}, ckpt)
        history.append({"epoch": epoch, "train_loss": total / n, "val_dice_256": vd,
                        "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})
        print(f"  epoch {epoch:3d}/{epochs}  loss {total / n:.4f}  val_dice(256) {vd:.4f}"
              f"{'  * best' if epoch == best_epoch else ''}  ({time.time() - t0:.0f}s)")
    train_sec = time.time() - t_start

    # ---- 최고 에폭 체크포인트로 원본 크기 평가
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    hc = None
    if os.path.exists(config.HC_CSV):
        hc = load_hc_table(config.HC_CSV)
        if set(va_d["image"]) - set(hc.index):
            print("  (HC csv의 파일명이 검증 이미지와 달라 HC 오차는 건너뜁니다)")
            hc = None

    pred_final = models.make_predict_fn(model, device=device, size=size, ellipse=True)
    s_final, per_final = evaluate(pred_final, va_i, va_g, split_df=va_d, hc_table=hc, return_per_image=True)
    pred_seg = models.make_predict_fn(model, device=device, size=size, ellipse=False)
    s_seg, per_seg = evaluate(pred_seg, va_i, va_g, split_df=va_d, hc_table=hc, return_per_image=True)

    os.makedirs(result_dir, exist_ok=True)
    save_per_image(per_final, method, "val", fold, result_dir)
    save_per_image(per_seg, f"{method}-seg", "val", fold, result_dir)

    summary = {"method": method, "fold": fold, "best_epoch": best_epoch, "val_dice_256": best,
               **s_final, "seg_IoU": s_seg["IoU"], "seg_Dice": s_seg["Dice"],
               "params": count_params(model), "train_sec": round(train_sec)}
    log_dir = os.path.join(result_dir, "logs")  # stats.load_results가 읽지 않는 하위 폴더
    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(log_dir, f"{method}__fold{fold}_history.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(log_dir, f"{method}__fold{fold}_summary.csv"), index=False)
    print(f"[{method} fold{fold}] 최종(타원) IoU {s_final['IoU']:.4f} / Dice {s_final['Dice']:.4f}"
          f" | 분할만 IoU {s_seg['IoU']:.4f} | 시간 {s_final['ms_per_image']:.1f}ms/장 | 파라미터 {summary['params']:,}")
    return summary


def run(model_name, folds=None, **kwargs):
    """여러 폴드를 차례로 학습하고 요약 표를 돌려준다. folds=None이면 전체 폴드."""
    split_dir = kwargs.get("split_dir") or config.SPLIT_DIR
    folds = list(range(n_folds(split_dir))) if folds is None else list(folds)
    rows = []
    for k in folds:
        s = train_fold(model_name, k, **kwargs)
        if s is not None:
            rows.append(s)
    df = pd.DataFrame(rows)
    if len(df) > 1:
        print(f"\n폴드 평균 IoU {df['IoU'].mean():.4f} ± {df['IoU'].std():.4f}, "
              f"Dice {df['Dice'].mean():.4f} ± {df['Dice'].std():.4f}")
    return df


def main():
    ap = argparse.ArgumentParser(description="폴드별 학습 (U-Net / FCN 공통)")
    ap.add_argument("--model", required=True, choices=models.list_models())
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="예: 0 1 2 (기본: 전체 폴드)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--no_aug", action="store_true", help="증강 없이 학습 (증강 효과 비교용)")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run(a.model, folds=a.folds, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, size=a.size, aug=not a.no_aug, seed=a.seed,
        num_workers=a.num_workers, pretrained=not a.no_pretrained, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
