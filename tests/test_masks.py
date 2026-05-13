from pathlib import Path

import numpy as np
from PIL import Image

from polyp_edge.masks import connected_component_boxes, pair_images_and_masks, whole_mask_box


def test_whole_mask_box_normalizes_coordinates() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:50, 40:100] = True

    boxes = whole_mask_box(mask, min_area=1)

    assert len(boxes) == 1
    box = boxes[0]
    assert box.class_id == 0
    assert box.x_center == 0.35
    assert box.y_center == 0.35
    assert box.width == 0.30
    assert box.height == 0.30


def test_connected_component_boxes_returns_multiple_regions() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:4, 1:5] = True
    mask[10:15, 12:18] = True

    boxes = connected_component_boxes(mask, min_area=1)

    assert len(boxes) == 2
    assert sorted(box.area_px for box in boxes) == [12, 30]


def test_pair_images_and_masks_by_stem(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    Image.new("RGB", (4, 4)).save(image_dir / "a.jpg")
    Image.new("RGB", (4, 4)).save(image_dir / "b.jpg")
    Image.new("L", (4, 4)).save(mask_dir / "a.png")
    Image.new("L", (4, 4)).save(mask_dir / "c.png")

    pairs = pair_images_and_masks(image_dir, mask_dir)

    assert pairs == [(image_dir / "a.jpg", mask_dir / "a.png")]

