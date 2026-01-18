import os
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PVSegmentationDataset
from model import UNet

from datetime import datetime


@dataclass
class Config:
    train_images = "data_strat/train/images"
    train_masks  = "data_strat/train/masks"
    val_images   = "data_strat/val/images"
    val_masks    = "data_strat/val/masks"

    batch_size = 1      # variable sizes without padding => must be 1
    num_workers = 0     # Windows safe
    epochs = 30
    patience = 5

    lrs = (1e-3, 3e-4, 1e-4)
    out_dir = "outputs"
    seed = 0

    # UNet has 4 pooling stages => need H,W multiples of 16 to get exact output size
    pad_multiple = 16


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pad_to_multiple(x, multiple=16, value=0.0):
    """
    Pad x on bottom/right so that H and W are multiples of `multiple`.
    Returns (x_pad, orig_h, orig_w).
    x: [B,C,H,W]
    """
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), value=value)  # (L,R,T,B)
    return x_pad, h, w


def crop_to_hw(x, h, w):
    """Crop x to original height/width (top-left). x: [B,C,H,W]"""
    return x[:, :, :h, :w]


def soft_dice_loss(logits, targets, smooth=1.0, eps=1e-8):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * inter + smooth) / (den + smooth + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_iou_from_logits(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))

    dice = (2.0 * inter + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    iou  = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def pos_ratio(tensor01):
    return tensor01.mean().item()


def run_one_lr(cfg: Config, lr: float, device):
    print(f"\n=== TRAINING lr={lr} on device={device} ===")

    train_ds = PVSegmentationDataset(cfg.train_images, cfg.train_masks)
    val_ds   = PVSegmentationDataset(cfg.val_images, cfg.val_masks)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, f"unet_best_lr{lr}.pth")

    best_val = -1.0
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        tr_loss = 0.0
        tr_dice = 0.0
        tr_iou  = 0.0
        tr_gt_pos = 0.0
        tr_pr_pos = 0.0

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            if Y.ndim == 3:
                Y = Y.unsqueeze(1)

            # Pad X and Y to multiples of 16, forward, then crop logits back
            X_pad, h0, w0 = pad_to_multiple(X, multiple=cfg.pad_multiple, value=0.0)
            Y_pad, _, _   = pad_to_multiple(Y, multiple=cfg.pad_multiple, value=0.0)

            logits_pad = model(X_pad)

            logits = crop_to_hw(logits_pad, h0, w0)
            Yc = crop_to_hw(Y_pad, h0, w0)

            loss = bce(logits, Yc) + soft_dice_loss(logits, Yc)

            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_loss += loss.item()

            dice, iou = dice_iou_from_logits(logits.detach(), Yc, thr=0.5)
            tr_dice += dice
            tr_iou  += iou

            preds01 = (torch.sigmoid(logits.detach()) >= 0.5).float()
            tr_gt_pos += pos_ratio((Yc >= 0.5).float())
            tr_pr_pos += pos_ratio(preds01)

        ntr = len(train_loader)
        tr_loss /= ntr
        tr_dice /= ntr
        tr_iou  /= ntr
        tr_gt_pos /= ntr
        tr_pr_pos /= ntr

        # -------------------- VAL --------------------
        model.eval()
        va_loss = 0.0
        va_dice = 0.0
        va_iou  = 0.0
        va_gt_pos = 0.0
        va_pr_pos = 0.0

        with torch.no_grad():
            for X, Y in val_loader:
                X = X.to(device)
                Y = Y.to(device)

                if Y.ndim == 3:
                    Y = Y.unsqueeze(1)

                X_pad, h0, w0 = pad_to_multiple(X, multiple=cfg.pad_multiple, value=0.0)
                Y_pad, _, _   = pad_to_multiple(Y, multiple=cfg.pad_multiple, value=0.0)

                logits_pad = model(X_pad)

                logits = crop_to_hw(logits_pad, h0, w0)
                Yc = crop_to_hw(Y_pad, h0, w0)

                loss = bce(logits, Yc) + soft_dice_loss(logits, Yc)
                va_loss += loss.item()

                dice, iou = dice_iou_from_logits(logits, Yc, thr=0.5)
                va_dice += dice
                va_iou  += iou

                preds01 = (torch.sigmoid(logits) >= 0.5).float()
                va_gt_pos += pos_ratio((Yc >= 0.5).float())
                va_pr_pos += pos_ratio(preds01)

        nva = len(val_loader)
        va_loss /= nva
        va_dice /= nva
        va_iou  /= nva
        va_gt_pos /= nva
        va_pr_pos /= nva

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} dice {tr_dice:.4f} iou {tr_iou:.4f} "
            f"(gt_pos {tr_gt_pos:.4f} pred_pos {tr_pr_pos:.4f}) || "
            f"val loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f} "
            f"(gt_pos {va_gt_pos:.4f} pred_pos {va_pr_pos:.4f})"
        )

        # Early stopping on val dice
        if va_dice > best_val + 1e-4:
            best_val = va_dice
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ saved best: {best_path} (val dice={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  🛑 early stopping (no val improvement for {cfg.patience} epochs)")
                break

    return best_val, best_path


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a unique output folder per run to avoid overwriting checkpoints
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("Run outputs dir:", cfg.out_dir)

    results = []
    for lr in cfg.lrs:
        best_val, best_path = run_one_lr(cfg, lr, device)
        results.append((lr, best_val, best_path))

    print("\n=== SUMMARY (val dice) ===")
    results.sort(key=lambda x: x[1], reverse=True)
    for lr, valdice, path in results:
        print(f"lr={lr:g} | best val dice={valdice:.4f} | ckpt={path}")


if __name__ == "__main__":
    main()
