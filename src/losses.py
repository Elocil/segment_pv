import torch
import torch.nn.functional as F

def masked_bce_with_logits(logits, targets, valid, eps=1e-8):
    # logits/targets/valid: [B,1,H,W]
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = bce * valid
    return bce.sum() / (valid.sum() + eps)

def masked_soft_dice_loss(logits, targets, valid, smooth=1.0, eps=1e-8):
    probs = torch.sigmoid(logits)

    probs = probs * valid
    targets = targets * valid

    inter = (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2.0 * inter + smooth) / (den + smooth + eps)
    return 1.0 - dice.mean()

def total_masked_loss(logits, targets, valid, alpha=0.5):
    # alpha=0.5 => 50% BCE, 50% Dice
    return (1.0 - alpha) * masked_bce_with_logits(logits, targets, valid) + alpha * masked_soft_dice_loss(logits, targets, valid)
