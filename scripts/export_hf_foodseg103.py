from pathlib import Path
import json
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "EduardoPacheco/FoodSeg103"

OUT = Path("data/FoodSeg103")
IMG_ROOT = OUT / "Images" / "img_dir"
MASK_ROOT = OUT / "Images" / "ann_dir"
IMAGESETS = OUT / "ImageSets"

def get_field(example, candidates):
    for name in candidates:
        if name in example:
            return name
    raise KeyError(f"Could not find any of these fields in example: {candidates}. Found: {list(example.keys())}")

def save_mask(mask_obj, out_path):
    arr = np.array(mask_obj)

    # FoodSeg103 masks should be single-channel class-id masks.
    if arr.ndim == 3:
        raise RuntimeError(
            f"Mask is multi-channel with shape {arr.shape}. "
            "Expected a single-channel class-id mask."
        )

    max_id = int(arr.max()) if arr.size else 0
    if max_id > 255:
        raise RuntimeError(f"Mask has class id >255 ({max_id}); cannot save as uint8 PNG safely.")

    Image.fromarray(arr.astype(np.uint8), mode="L").save(out_path)

def main():
    print(f"Loading {REPO_ID} from Hugging Face...")
    ds = load_dataset(REPO_ID)

    print(ds)
    print("Available splits:", list(ds.keys()))

    split_map = {
        "train": "train",
        "validation": "test",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    IMAGESETS.mkdir(parents=True, exist_ok=True)

    for hf_split, out_split in split_map.items():
        if hf_split not in ds:
            raise RuntimeError(f"Missing expected split: {hf_split}. Available: {list(ds.keys())}")

        split_ds = ds[hf_split]
        img_dir = IMG_ROOT / out_split
        mask_dir = MASK_ROOT / out_split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting HF split '{hf_split}' -> project split '{out_split}'")
        print(f"Samples: {len(split_ds)}")

        first = split_ds[0]
        image_key = get_field(first, ["image", "img"])
        mask_key = get_field(first, ["label", "mask", "segmentation", "annotation"])

        stems = []

        for i, ex in enumerate(split_ds):
            stem = f"{out_split}_{i:06d}"
            stems.append(stem)

            image = ex[image_key].convert("RGB")
            mask = ex[mask_key]

            image.save(img_dir / f"{stem}.jpg", quality=95)
            save_mask(mask, mask_dir / f"{stem}.png")

            if (i + 1) % 500 == 0:
                print(f"  exported {i + 1}/{len(split_ds)}")

        with open(IMAGESETS / f"{out_split}.txt", "w") as f:
            for stem in stems:
                f.write(stem + "\n")

        print(f"Finished {out_split}: {len(stems)} samples")

    # Download and convert id2label.json -> category_id.txt
    try:
        id2label_file = hf_hub_download(
            repo_id=REPO_ID,
            filename="id2label.json",
            repo_type="dataset",
        )
        with open(id2label_file, "r") as f:
            raw = json.load(f)

        id2label = {int(k): v for k, v in raw.items()}

        with open(OUT / "category_id.txt", "w") as f:
            for k in sorted(id2label):
                f.write(f"{k}\t{id2label[k]}\n")

        print(f"\nWrote label map: {OUT / 'category_id.txt'}")

    except Exception as e:
        print("\nWARNING: Could not download id2label.json.")
        print("Reason:", repr(e))
        print("Creating fallback category_id.txt from observed mask IDs.")

        observed = set()
        for split in ["train", "test"]:
            for mask_path in (MASK_ROOT / split).glob("*.png"):
                arr = np.array(Image.open(mask_path))
                observed.update(int(v) for v in np.unique(arr))

        with open(OUT / "category_id.txt", "w") as f:
            for k in sorted(observed):
                f.write(f"{k}\tclass_{k}\n")

    print("\nDone.")
    print("Created dataset at:", OUT.resolve())
    print("\nExpected layout:")
    print("data/FoodSeg103/")
    print("  Images/img_dir/train/*.jpg")
    print("  Images/img_dir/test/*.jpg")
    print("  Images/ann_dir/train/*.png")
    print("  Images/ann_dir/test/*.png")
    print("  ImageSets/train.txt")
    print("  ImageSets/test.txt")
    print("  category_id.txt")

if __name__ == "__main__":
    main()
