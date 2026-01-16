import torch
import torch.nn.functional as F

def pad_collate(batch):
    xs, ys = zip(*batch)

    max_h = max(x.shape[1] for x in xs)
    max_w = max(x.shape[2] for x in xs)

    X, Y, V = [], [], []

    for x, y in zip(xs, ys):
        _, h, w = x.shape
        pad_h = max_h - h
        pad_w = max_w - w

        # pad order: (left, right, top, bottom)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        y_pad = F.pad(y, (0, pad_w, 0, pad_h), value=0.0)

        # valid mask: 1 on real pixels, 0 on padded pixels
        v = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        v[:, :h, :w] = 1.0

        X.append(x_pad)
        Y.append(y_pad)
        V.append(v)

    return torch.stack(X, 0), torch.stack(Y, 0), torch.stack(V, 0)
