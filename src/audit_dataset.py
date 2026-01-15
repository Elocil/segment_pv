from pathlib import Path
from PIL import Image
import numpy as np

IMG_SUFFIX = ".jpg"
MASK_SUFFIX = "_mask.png"

def audit_split(split_name: str, images_dir: str, masks_dir: str, max_print=10):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    imgs = sorted(images_dir.glob(f"*{IMG_SUFFIX}"))
    if len(imgs) == 0:
        raise FileNotFoundError(f"Aucune image dans {images_dir}")

    missing_masks = 0
    size_mismatch = 0
    non_binary = 0
    empty_masks = 0
    bad_images = 0

    examples = {"missing": [], "mismatch": [], "nonbinary": [], "empty": [], "badimg": []}

    for img_path in imgs:
        mask_path = masks_dir / f"{img_path.stem}{MASK_SUFFIX}"

        if not mask_path.exists():
            missing_masks += 1
            if len(examples["missing"]) < max_print:
                examples["missing"].append(img_path.name)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception:
            bad_images += 1
            if len(examples["badimg"]) < max_print:
                examples["badimg"].append(img_path.name)
            continue

        if img.size != mask.size:  # PIL size = (W,H)
            size_mismatch += 1
            if len(examples["mismatch"]) < max_print:
                examples["mismatch"].append((img_path.name, img.size, mask.size))

        mask_np = np.array(mask)
        uniq = np.unique(mask_np)

        # binaire attendu : {0,255} ou {0,1} ou {0}
        is_binary = set(uniq.tolist()).issubset({0, 1, 255})
        if not is_binary:
            non_binary += 1
            if len(examples["nonbinary"]) < max_print:
                examples["nonbinary"].append((img_path.name, uniq[:10].tolist()))

        # masque vide : aucun pixel >0
        if (mask_np > 0).sum() == 0:
            empty_masks += 1
            if len(examples["empty"]) < max_print:
                examples["empty"].append(img_path.name)

    total = len(imgs)
    print(f"\n=== AUDIT {split_name.upper()} ===")
    print(f"Total images: {total}")
    print(f"Missing masks: {missing_masks}")
    print(f"Bad/corrupt files: {bad_images}")
    print(f"Size mismatches: {size_mismatch}")
    print(f"Non-binary masks: {non_binary}")
    print(f"Empty masks (no PV pixels): {empty_masks}")

    if examples["missing"]:
        print("Examples missing masks:", examples["missing"])
    if examples["badimg"]:
        print("Examples bad images:", examples["badimg"])
    if examples["mismatch"]:
        print("Examples size mismatch:", examples["mismatch"][:3])
    if examples["nonbinary"]:
        print("Examples non-binary:", examples["nonbinary"][:3])
    if examples["empty"]:
        print("Examples empty masks:", examples["empty"][:5])

def main():
    audit_split("train", "data_strat/train/images", "data_strat/train/masks")
    audit_split("val",   "data_strat/val/images",   "data_strat/val/masks")
    audit_split("test",  "data_strat/test/images",  "data_strat/test/masks")

if __name__ == "__main__":
    main()
