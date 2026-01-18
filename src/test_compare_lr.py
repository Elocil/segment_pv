import os
import argparse
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PVSegmentationDataset
from model import UNet


def make_unique_dir(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    out_dir = base_dir
    k = 1
    while os.path.exists(out_dir) and os.listdir(out_dir):
        out_dir = f"{base_dir}_{k}"
        k += 1
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def pad_to_multiple(x, multiple=16, value=0.0):
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), value=value)
    return x_pad, h, w


def crop_to_hw(x, h, w):
    return x[:, :, :h, :w]


def to_uint8_img(x_chw):
    x = (x_chw.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return x


def to_uint8_mask01(mask_1hw):
    y = (mask_1hw.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    return y


def mask_to_rgb(mask_u8):
    return np.stack([mask_u8]*3, axis=-1)


def overlay_red(img_u8, pred_mask_u8, alpha=0.35):
    overlay = img_u8.astype(np.float32).copy()
    m = (pred_mask_u8 > 0)  # (H,W)

    # tint in red
    overlay[m, 0] = overlay[m, 0] * (1 - alpha) + 255 * alpha
    overlay[m, 1] = overlay[m, 1] * (1 - alpha)
    overlay[m, 2] = overlay[m, 2] * (1 - alpha)

    return overlay.clip(0, 255).astype(np.uint8)


@torch.no_grad()
def predict_mask(model, X, thr=0.5, pad_multiple=16):
    # X: [1,3,H,W]
    X_pad, h0, w0 = pad_to_multiple(X, multiple=pad_multiple, value=0.0)
    logits_pad = model(X_pad)
    logits = crop_to_hw(logits_pad, h0, w0)
    probs = torch.sigmoid(logits)
    pred01 = (probs >= thr).float()  # [1,1,H,W]
    return pred01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_images", default="data_strat/test/images")
    ap.add_argument("--test_masks", default="data_strat/test/masks")

    ap.add_argument("--ckpt_lr1e3", required=True)
    ap.add_argument("--ckpt_lr3e4", required=True)
    ap.add_argument("--ckpt_lr1e4", required=True)

    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--pad_multiple", type=int, default=16)
    ap.add_argument("--num_viz", type=int, default=20)
    ap.add_argument("--out_base", default="outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = make_unique_dir(os.path.join(args.out_base, f"test_compare_{stamp}"))
    print("Compare outputs dir:", out_dir)

    ds = PVSegmentationDataset(args.test_images, args.test_masks)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # build 3 models (same architecture) and load weights
    def load_model(ckpt_path):
        m = UNet(in_channels=3, out_channels=1).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        return m

    m_1e3 = load_model(args.ckpt_lr1e3)
    m_3e4 = load_model(args.ckpt_lr3e4)
    m_1e4 = load_model(args.ckpt_lr1e4)

    viz_count = 0

    for i, (X, Y) in enumerate(loader):
        if viz_count >= args.num_viz:
            break

        X = X.to(device)
        Y = Y.to(device)
        if Y.ndim == 3:
            Y = Y.unsqueeze(1)

        # predictions
        p_1e3 = predict_mask(m_1e3, X, thr=args.thr, pad_multiple=args.pad_multiple)
        p_3e4 = predict_mask(m_3e4, X, thr=args.thr, pad_multiple=args.pad_multiple)
        p_1e4 = predict_mask(m_1e4, X, thr=args.thr, pad_multiple=args.pad_multiple)

        # convert to uint8
        img_u8 = to_uint8_img(X[0])
        gt_u8 = to_uint8_mask01(Y[0])
        p1_u8 = to_uint8_mask01(p_1e3[0])
        p2_u8 = to_uint8_mask01(p_3e4[0])
        p3_u8 = to_uint8_mask01(p_1e4[0])

        # build 5 columns
        col1 = img_u8
        col2 = mask_to_rgb(gt_u8)

        # show overlays (more readable than raw masks)
        col3 = overlay_red(img_u8, p1_u8)
        col4 = overlay_red(img_u8, p2_u8)
        col5 = overlay_red(img_u8, p3_u8)

        grid = np.concatenate([col1, col2, col3, col4, col5], axis=1)
        Image.fromarray(grid).save(os.path.join(out_dir, f"compare_{i:04d}.png"))

        viz_count += 1

    # write a small note
    with open(os.path.join(out_dir, "readme.txt"), "w", encoding="utf-8") as f:
        f.write("Columns: [RGB | GT mask | pred overlay LR=1e-3 | pred overlay LR=3e-4 | pred overlay LR=1e-4]\n")
        f.write(f"thr={args.thr}, pad_multiple={args.pad_multiple}\n")
        f.write(f"ckpt lr1e-3: {args.ckpt_lr1e3}\n")
        f.write(f"ckpt lr3e-4: {args.ckpt_lr3e4}\n")
        f.write(f"ckpt lr1e-4: {args.ckpt_lr1e4}\n")

    print("Done. Wrote", viz_count, "comparisons.")
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
