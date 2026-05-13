from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    area_px: int

    def to_yolo_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_binary_mask(mask_path: Path, threshold: int = 0) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    arr = np.asarray(mask)
    return arr > threshold


def whole_mask_box(mask: np.ndarray, class_id: int = 0, min_area: int = 10) -> list[Box]:
    ys, xs = np.nonzero(mask)
    if len(xs) < min_area:
        return []
    return [_box_from_pixels(xs, ys, mask.shape, class_id)]


def connected_component_boxes(
    mask: np.ndarray,
    class_id: int = 0,
    min_area: int = 10,
) -> list[Box]:
    """Return one box per connected component using a dependency-free flood fill.

    Public polyp datasets usually contain one polyp per image, so whole-mask boxes are
    preferred. This component mode is included for robustness when multiple regions exist.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[Box] = []

    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        xs: list[int] = []
        ys: list[int] = []

        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(xs) >= min_area:
            boxes.append(_box_from_pixels(np.asarray(xs), np.asarray(ys), mask.shape, class_id))

    return boxes


def _box_from_pixels(xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int], class_id: int) -> Box:
    height, width = shape
    x_min = float(xs.min())
    x_max = float(xs.max() + 1)
    y_min = float(ys.min())
    y_max = float(ys.max() + 1)

    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    x_center = (x_min + box_w / 2.0) / width
    y_center = (y_min + box_h / 2.0) / height

    return Box(
        class_id=class_id,
        x_center=_clip01(x_center),
        y_center=_clip01(y_center),
        width=_clip01(box_w / width),
        height=_clip01(box_h / height),
        area_px=int(len(xs)),
    )


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def pair_images_and_masks(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = {p.stem: p for p in iter_images(image_dir)}
    masks = {p.stem: p for p in iter_images(mask_dir)}
    pairs = [(images[stem], masks[stem]) for stem in sorted(images.keys() & masks.keys())]
    return pairs
