import os
import csv
import argparse
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PVSegmentationDataset
from model import UNet


# ----------------------------
# Utils: safe output dir
# ----------------------------
def make_unique_dir(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    out_dir = base_dir
    k = 1
    while os.path.exists(out_dir) and os.listdir(out_dir):
        out_dir = f"{base_dir}_{k}"
        k += 1
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ----------------------------
# Pad/crop (same logic as train)
# ----------------------------
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
    return x[:, :, :h, :w]


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def dice_iou_from_logits(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))

    dice = (2.0 * inter + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    iou = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def pos_ratio(tensor01):
    return tensor01.mean().item()


# ----------------------------
# Visualization helpers
# ----------------------------
def to_uint8_img(x_chw):
    # x_chw: torch [3,H,W] in [0,1]
    x = (x_chw.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return x


def to_uint8_mask(y_1hw):
    # y_1hw: torch [1,H,W] in {0,1}
    y = (y_1hw.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    return y


def save_viz(out_path, img_rgb_uint8, gt_mask_uint8, pr_mask_uint8):
    """
    Saves a 2x2 grid:
    [image | overlay]
    [gt    | pred   ]
    """
    H, W, _ = img_rgb_uint8.shape

    gt_rgb = np.stack([gt_mask_uint8]*3, axis=-1)
    pr_rgb = np.stack([pr_mask_uint8]*3, axis=-1)

    # overlay: tint predicted pixels in red
    overlay = img_rgb_uint8.copy().astype(np.float32)

    m2 = (pr_mask_uint8 > 0)  # shape (H,W) boolean
    overlay[m2, 0] = overlay[m2, 0] * 0.65 + 255 * 0.35  # R
    overlay[m2, 1] = overlay[m2, 1] * 0.65              # G
    overlay[m2, 2] = overlay[m2, 2] * 0.65              # B

    overlay = overlay.clip(0, 255).astype(np.uint8)

    top = np.concatenate([img_rgb_uint8, overlay], axis=1)
    bot = np.concatenate([gt_rgb, pr_rgb], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    Image.fromarray(grid).save(out_path)


# ----------------------------
# Main test
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_images", default="data_strat/test/images")
    ap.add_argument("--test_masks", default="data_strat/test/masks")
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--pad_multiple", type=int, default=16)
    ap.add_argument("--num_viz", type=int, default=12)
    ap.add_argument("--out_base", default="outputs")
    args = ap.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Safe output dir (never overwrite)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = make_unique_dir(os.path.join(args.out_base, f"test_{stamp}"))
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    print("Test outputs dir:", out_dir)
    print("Checkpoint:", args.ckpt)
    print("Threshold:", args.thr)

    # Data
    ds = PVSegmentationDataset(args.test_images, args.test_masks)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Accumulators
    dices = []
    ious = []
    gt_pos_list = []
    pr_pos_list = []

    per_image_rows = []
    viz_count = 0

    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            X = X.to(device)  # [1,3,H,W]
            Y = Y.to(device)  # [1,1,H,W]

            if Y.ndim == 3:
                Y = Y.unsqueeze(1)

            # pad -> forward -> crop back (same as train)
            X_pad, h0, w0 = pad_to_multiple(X, multiple=args.pad_multiple, value=0.0)
            logits_pad = model(X_pad)
            logits = crop_to_hw(logits_pad, h0, w0)

            # metrics
            dice, iou = dice_iou_from_logits(logits, Y, thr=args.thr)
            dices.append(dice)
            ious.append(iou)

            probs = torch.sigmoid(logits)
            preds01 = (probs >= args.thr).float()

            gt_pos = pos_ratio((Y >= 0.5).float())
            pr_pos = pos_ratio(preds01)

            gt_pos_list.append(gt_pos)
            pr_pos_list.append(pr_pos)

            # store per-image
            per_image_rows.append([i, dice, iou, gt_pos, pr_pos])

            # viz
            if viz_count < args.num_viz:
                img_u8 = to_uint8_img(X[0])
                gt_u8 = to_uint8_mask(Y[0])
                pr_u8 = to_uint8_mask(preds01[0])
                save_viz(os.path.join(viz_dir, f"sample_{i:04d}.png"), img_u8, gt_u8, pr_u8)
                viz_count += 1

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(ds)}...")

    # Aggregate
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_gt_pos = float(np.mean(gt_pos_list)) if gt_pos_list else 0.0
    mean_pr_pos = float(np.mean(pr_pos_list)) if pr_pos_list else 0.0

    print("\n=== TEST RESULTS ===")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")
    print(f"GT pos ratio  : {mean_gt_pos:.4f}")
    print(f"Pred pos ratio: {mean_pr_pos:.4f}")

    # Write summary
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Threshold: {args.thr}\n")
        f.write(f"Pad multiple: {args.pad_multiple}\n")
        f.write(f"Num test images: {len(ds)}\n")
        f.write(f"Mean Dice: {mean_dice:.6f}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")
        f.write(f"GT pos ratio: {mean_gt_pos:.6f}\n")
        f.write(f"Pred pos ratio: {mean_pr_pos:.6f}\n")

    # Write per-image CSV
    csv_path = os.path.join(out_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "dice", "iou", "gt_pos_ratio", "pred_pos_ratio"])
        w.writerows(per_image_rows)

    print("\nSaved:")
    print(" -", os.path.join(out_dir, "summary.txt"))
    print(" -", csv_path)
    print(" -", viz_dir)


if __name__ == "__main__":
    main()
