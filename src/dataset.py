from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class PVSegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        image_suffix: str = ".jpg",
        mask_suffix: str = "_mask.png",
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        self.image_paths = sorted(self.images_dir.glob(f"*{self.image_suffix}"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        if self.masks_dir:
            missing = []
            for p in self.image_paths:
                m = self.masks_dir / f"{p.stem}{self.mask_suffix}"
                if not m.exists():
                    missing.append(m.name)
            if missing:
                raise FileNotFoundError(f"Missing masks (examples): {missing[:10]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # H,W,3
        x = torch.from_numpy(img_np).permute(2, 0, 1)     # 3,H,W

        if not self.masks_dir:
            return x, img_path.name

        mask_path = self.masks_dir / f"{img_path.stem}{self.mask_suffix}"
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask, dtype=np.uint8)
        y = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)  # 1,H,W

        return x, y
