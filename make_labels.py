import json
import cv2
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()

DATASET_DIR = Path(args.dataset)

for split in ["train", "val"]:
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    for json_path in images_dir.glob("*.json"):
        with open(json_path) as f:
            data = json.load(f)

        h, w = data["imageHeight"], data["imageWidth"]
        txt_path = labels_dir / json_path.with_suffix(".txt").name

        with open(txt_path, "w") as out:
            for shape in data["shapes"]:
                pts = np.array(shape["points"], dtype=np.float32)

                if len(pts) < 3:
                    print(f"Skipping shape with {len(pts)} points in {json_path.name}")
                    continue

                rect = cv2.minAreaRect(pts)
                corners = cv2.boxPoints(rect)

                corners[:, 0] /= w
                corners[:, 1] /= h

                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
                out.write(f"0 {coords}\n")

        print(f"Converted {json_path.name} -> {txt_path}")

print("Done")