from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_SUFFIX = ".jpg"
MASK_SUFFIX = "_mask.png"

def sample_overlay(images_dir, masks_dir, n=8, seed=0):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    imgs = sorted(images_dir.glob(f"*{IMG_SUFFIX}"))
    assert len(imgs) > 0, f"Aucune image dans {images_dir}"

    random.seed(seed)
    idxs = random.sample(range(len(imgs)), k=min(n, len(imgs)))

    print(f"\n=== {images_dir} ===")
    print(f"Nb images: {len(imgs)}")

    for k, idx in enumerate(idxs, 1):
        img_path = imgs[idx]
        mask_path = masks_dir / f"{img_path.stem}{MASK_SUFFIX}"
        if not mask_path.exists():
            raise FileNotFoundError(f"Masque manquant: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_np = np.array(img)
        mask_np = np.array(mask)

        # force binaire pour inspection
        mask_bin = (mask_np > 0).astype(np.uint8)

        print(f"[{k}] {img_path.name} | img={img_np.shape} mask={mask_np.shape} "
              f"| mask unique raw={np.unique(mask_np)[:10]} | bin unique={np.unique(mask_bin)}")

        plt.figure()
        plt.imshow(img_np)
        plt.imshow(mask_bin, alpha=0.35)
        plt.title(f"{img_path.name}")
        plt.axis("off")
        plt.show()

def main():
    sample_overlay("data_strat/train/images", "data_strat/train/masks", n=8, seed=1)
    sample_overlay("data_strat/val/images", "data_strat/val/masks", n=8, seed=2)

if __name__ == "__main__":
    main()
