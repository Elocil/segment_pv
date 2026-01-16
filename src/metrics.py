import torch

@torch.no_grad()
def masked_dice_iou_from_logits(logits, targets, valid, thr=0.5, eps=1e-8):
    # logits/targets/valid: [B,1,H,W]
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    preds = preds * valid
    targets = targets * valid

    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))

    dice = (2.0 * inter + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    iou  = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()

@torch.no_grad()
def masked_pos_ratio(tensor01, valid, eps=1e-8):
    # tensor01: [B,1,H,W] values in {0,1}
    return ((tensor01 * valid).sum() / (valid.sum() + eps)).item()
