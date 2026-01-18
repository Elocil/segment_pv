import os
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PVSegmentationDataset
from model import UNet


# --------- same padding logic as train.py ----------
def pad_to_multiple(x, multiple=16, value=0.0):
    # x: [B,C,H,W]
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), value=value)  # (L,R,T,B)
    return x_pad, h, w

def crop_to_hw(x, h, w):
    return x[:, :, :h, :w]


@torch.no_grad()
def dice_iou_from_logits(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))

    dice = (2.0 * inter + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    iou  = (inter + eps) / (union + eps)

    return dice, iou  # tensors [B]


@torch.no_grad()
def prf_from_logits(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))

    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    f1   = (2 * prec * rec + eps) / (prec + rec + eps)

    return prec, rec, f1  # tensors [B]


def eval_checkpoint(ckpt_path, loader, device, thr=0.5, pad_multiple=16):
    model = UNet(in_channels=3, out_channels=1).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    bce = nn.BCEWithLogitsLoss(reduction="mean")

    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum  = 0.0
    prec_sum = 0.0
    rec_sum  = 0.0
    f1_sum   = 0.0
    gt_pos_sum = 0.0
    pr_pos_sum = 0.0
    n = 0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        if Y.ndim == 3:
            Y = Y.unsqueeze(1)

        # pad -> forward -> crop back (same as training)
        X_pad, h0, w0 = pad_to_multiple(X, multiple=pad_multiple, value=0.0)
        Y_pad, _, _   = pad_to_multiple(Y, multiple=pad_multiple, value=0.0)

        logits_pad = model(X_pad)
        logits = crop_to_hw(logits_pad, h0, w0)
        Yc     = crop_to_hw(Y_pad, h0, w0)

        loss = bce(logits, Yc)
        loss_sum += loss.item()

        dice_b, iou_b = dice_iou_from_logits(logits, Yc, thr=thr)
        prec_b, rec_b, f1_b = prf_from_logits(logits, Yc, thr=thr)

        dice_sum += dice_b.mean().item()
        iou_sum  += iou_b.mean().item()
        prec_sum += prec_b.mean().item()
        rec_sum  += rec_b.mean().item()
        f1_sum   += f1_b.mean().item()

        pr01 = (torch.sigmoid(logits) >= thr).float()
        gt_pos_sum += (Yc >= 0.5).float().mean().item()
        pr_pos_sum += pr01.mean().item()

        n += 1

    # averages across batches
    return {
        "loss_bce": loss_sum / max(n, 1),
        "dice":     dice_sum / max(n, 1),
        "iou":      iou_sum / max(n, 1),
        "precision":prec_sum / max(n, 1),
        "recall":   rec_sum / max(n, 1),
        "f1":       f1_sum / max(n, 1),
        "gt_pos":   gt_pos_sum / max(n, 1),
        "pred_pos": pr_pos_sum / max(n, 1),
        "batches":  n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_images", default="data_strat/test/images")
    ap.add_argument("--test_masks",  default="data_strat/test/masks")
    ap.add_argument("--ckpt_lr1e3", required=True)
    ap.add_argument("--ckpt_lr3e4", required=True)
    ap.add_argument("--ckpt_lr1e4", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--pad_multiple", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Threshold:", args.thr)

    # dataset / loader
    ds = PVSegmentationDataset(args.test_images, args.test_masks)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # output dir + csv
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"test_metrics_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics_test_compare.csv")

    ckpts = [
        ("lr=1e-3", args.ckpt_lr1e3),
        ("lr=3e-4", args.ckpt_lr3e4),
        ("lr=1e-4", args.ckpt_lr1e4),
    ]

    rows = []
    for label, ckpt in ckpts:
        print("\nEvaluating:", label)
        print("  ckpt:", ckpt)
        m = eval_checkpoint(ckpt, loader, device, thr=args.thr, pad_multiple=args.pad_multiple)
        row = {"label": label, "ckpt": ckpt, **m}
        rows.append(row)

        print(
            f"  dice={m['dice']:.4f}  iou={m['iou']:.4f}  "
            f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  f1={m['f1']:.4f}  "
            f"(gt_pos={m['gt_pos']:.4f} pred_pos={m['pred_pos']:.4f})"
        )

    # write csv
    fieldnames = ["label", "ckpt", "loss_bce", "dice", "iou", "precision", "recall", "f1", "gt_pos", "pred_pos", "batches"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print("\nDone.")
    print("Wrote:", csv_path)
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
