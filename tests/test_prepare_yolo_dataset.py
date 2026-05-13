from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw
import yaml


def test_prepare_yolo_dataset_script_creates_labels_and_yaml(tmp_path: Path) -> None:
    image_dir = tmp_path / "raw" / "images"
    mask_dir = tmp_path / "raw" / "masks"
    out_dir = tmp_path / "processed"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for idx in range(3):
        Image.new("RGB", (100, 80), color=(20, 30, 40)).save(image_dir / f"case_{idx}.jpg")
        mask = Image.new("L", (100, 80), color=0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([10, 20, 40, 50], fill=255)
        mask.save(mask_dir / f"case_{idx}.png")

    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_yolo_dataset.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--name",
            "synthetic",
            "--images",
            str(image_dir),
            "--masks",
            str(mask_dir),
            "--out",
            str(out_dir),
            "--splits",
            "train=0.67,val=0.33",
            "--min-area",
            "1",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    dataset_root = out_dir / "synthetic"
    labels = sorted((dataset_root / "labels").rglob("*.txt"))
    assert len(labels) == 3
    assert all(label.read_text(encoding="utf-8").startswith("0 ") for label in labels)

    config = yaml.safe_load((dataset_root / "synthetic.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "polyp"}
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"

