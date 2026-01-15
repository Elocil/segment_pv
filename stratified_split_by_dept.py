from pathlib import Path
import random
import shutil
from collections import defaultdict

# =========================
# CONFIG
# =========================
# Dossiers source (tes données actuelles)
SRC_SETS = [
    ("train", Path("data/train/images"), Path("data/train/masks")),
    ("val",   Path("data/val/images"),   Path("data/val/masks")),
    # Si tu as déjà un test et que tu veux aussi l'inclure dans le pool source, décommente :
    # ("test",  Path("data/test/images"),  Path("data/test/masks")),
]

# Dossier destination (nouvelle organisation)
DST_ROOT = Path("data_strat")

IMAGE_SUFFIX = ".jpg"
MASK_SUFFIX = "_mask.png"

# Ratios (train/val/test) - doit sommer à 1.0
R_TRAIN = 0.70
R_VAL   = 0.15
R_TEST  = 0.15

SEED = 42
DRY_RUN = False  # mets False quand tu es sûre

# =========================
# HELPERS
# =========================
def dept_from_stem(stem: str) -> str:
    """
    Département = 3 premiers caractères du nom de fichier (sans extension).
    Ex: '022291' -> '022'
    """
    if len(stem) < 3:
        raise ValueError(f"Nom de fichier trop court pour extraire le département: {stem}")
    return stem[:3]

def ensure_dirs(root: Path):
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

def load_pairs():
    """
    Charge toutes les paires (image, masque) depuis les sets source.
    Retourne une liste de tuples (img_path, mask_path).
    """
    pairs = []
    for set_name, img_dir, mask_dir in SRC_SETS:
        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Chemins source manquants pour {set_name}: {img_dir} / {mask_dir}")

        imgs = sorted(img_dir.glob(f"*{IMAGE_SUFFIX}"))
        if len(imgs) == 0:
            print(f"⚠️ Aucune image trouvée dans {img_dir}")
            continue

        for img_path in imgs:
            mask_path = mask_dir / f"{img_path.stem}{MASK_SUFFIX}"
            if not mask_path.exists():
                raise FileNotFoundError(f"Masque manquant: {mask_path} (pour image {img_path.name})")
            pairs.append((img_path, mask_path))

    if len(pairs) == 0:
        raise RuntimeError("Aucune paire image/masque trouvée dans les dossiers source.")
    return pairs

def stratified_split(pairs):
    """
    Split stratifié par département.
    Retourne dict: {"train":[...], "val":[...], "test":[...]}
    """
    groups = defaultdict(list)
    for img_path, mask_path in pairs:
        d = dept_from_stem(img_path.stem)
        groups[d].append((img_path, mask_path))

    random.seed(SEED)

    splits = {"train": [], "val": [], "test": []}
    summary = {}

    for dept, dept_pairs in groups.items():
        random.shuffle(dept_pairs)
        n = len(dept_pairs)

        n_train = int(round(n * R_TRAIN))
        n_val   = int(round(n * R_VAL))
        # pour éviter les erreurs d'arrondi, le reste va au test
        n_test  = n - n_train - n_val

        # garde-fous minimaux (évite un dept qui disparaît d'un split)
        # Si un département a très peu d'images, ces garde-fous peuvent répartir 1-1-... de manière raisonnable.
        if n >= 3:
            n_train = max(1, n_train)
            n_val   = max(1, n_val)
            n_test  = max(1, n_test)
            # re-normalise si on a dépassé n
            while (n_train + n_val + n_test) > n:
                # retire du train d'abord (le plus grand)
                if n_train >= max(n_val, n_test) and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break

        train_part = dept_pairs[:n_train]
        val_part   = dept_pairs[n_train:n_train + n_val]
        test_part  = dept_pairs[n_train + n_val:]

        splits["train"].extend(train_part)
        splits["val"].extend(val_part)
        splits["test"].extend(test_part)

        summary[dept] = {"total": n, "train": len(train_part), "val": len(val_part), "test": len(test_part)}

    return splits, summary

def copy_pairs(split_name, pairs_list):
    dst_img_dir = DST_ROOT / split_name / "images"
    dst_msk_dir = DST_ROOT / split_name / "masks"

    for img_path, mask_path in pairs_list:
        dst_img = dst_img_dir / img_path.name
        dst_msk = dst_msk_dir / mask_path.name

        if DRY_RUN:
            continue
        shutil.copy2(img_path, dst_img)
        shutil.copy2(mask_path, dst_msk)

def main():
    print("=== Chargement des paires (image, masque) depuis data/train + data/val ===")
    pairs = load_pairs()
    print(f"Total paires trouvées : {len(pairs)}")

    ensure_dirs(DST_ROOT)

    splits, summary = stratified_split(pairs)

    print("\n=== Résumé par département ===")
    depts_sorted = sorted(summary.keys())
    for d in depts_sorted:
        s = summary[d]
        print(f"Dept {d}: total={s['total']}, train={s['train']}, val={s['val']}, test={s['test']}")

    print("\n=== Totaux ===")
    print("TRAIN:", len(splits["train"]))
    print("VAL  :", len(splits["val"]))
    print("TEST :", len(splits["test"]))
    print("DRY_RUN =", DRY_RUN)

    # Copie des fichiers
    print("\n=== Copie des fichiers vers data_strat/... ===")
    copy_pairs("train", splits["train"])
    copy_pairs("val",   splits["val"])
    copy_pairs("test",  splits["test"])

    if DRY_RUN:
        print("\n✅ DRY RUN terminé : rien n'a été copié. Passe DRY_RUN=False pour exécuter réellement.")
    else:
        print("\n✅ Split terminé : fichiers copiés dans data_strat/.")

if __name__ == "__main__":
    main()
