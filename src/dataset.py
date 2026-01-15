from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PVSegmentationDataset(Dataset):
    """
    Segmentation dataset:
      image: xxxx.jpg
      mask : xxxx_mask.png (binaire)
    """
    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        image_suffix: str = ".jpg",
        mask_suffix: str = "_mask.png",
        resize: Optional[Tuple[int, int]] = None,  # (W, H)
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else None
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.resize = resize

        self.image_paths = sorted(self.images_dir.glob(f"*{self.image_suffix}"))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"Aucune image trouvée dans {self.images_dir} avec suffix {self.image_suffix}")

        if self.masks_dir is not None:
            missing = []
            for img_path in self.image_paths:
                mask_path = self.masks_dir / f"{img_path.stem}{self.mask_suffix}"
                if not mask_path.exists():
                    missing.append(mask_path.name)
            if missing:
                raise FileNotFoundError(f"{len(missing)} masques manquants. Exemples: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if self.resize is not None:
            img = img.resize(self.resize, resample=Image.BILINEAR)
        return img

    def _load_mask(self, img_stem: str) -> Image.Image:
        assert self.masks_dir is not None
        mask_path = self.masks_dir / f"{img_stem}{self.mask_suffix}"
        mask = Image.open(mask_path).convert("L")
        if self.resize is not None:
            mask = mask.resize(self.resize, resample=Image.NEAREST)
        return mask

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)

        img_np = np.array(img, dtype=np.float32) / 255.0  # HWC
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # CHW

        if self.masks_dir is None:
            return img_t, img_path.name

        mask = self._load_mask(img_path.stem)
        mask_np = np.array(mask, dtype=np.uint8)

        # force binaire 0/1
        mask_bin = (mask_np > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0)  # 1xHxW

        return img_t, mask_t
