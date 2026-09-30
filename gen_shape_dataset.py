"""
Generate a synthetic dataset of binary shape-mask crops (Circle / Square /
Rectangle) for training the small CNN that augments classify_object's
shape classification in live_demo.

CRITICAL design point: at inference time (mycv.shape_cnn.predict_shape /
live_demo's _run_shape_cnn), the CNN always receives a crop that has
already been tightened to the OBJECT'S OWN bounding box -- by definition
that means the shape touches all four edges of the crop before it gets
resized to the 24x24 input. An earlier version of this generator instead
drew shapes with background margin inside a fixed-size canvas (size
varying, so most training examples had lots of empty border) -- a
completely different input distribution from what the model sees in
practice, which made it confidently WRONG on real bbox-cropped input.

This version fixes that: every shape is drawn on a generous canvas,
degraded (see `_augment` below) and cropped to its own tight bounding box
AT NATIVE RESOLUTION, and THEN resized to the 24x24 input size using the
exact same resize helper (`mycv.shape_cnn._resize_mask_to_input`) used at
inference -- guaranteeing the training and inference distributions match,
including how much a given degradation gets smoothed away by the resize.

Uses mycv.rotate_image for rotated squares/rectangles, so the CNN sees
non-axis-aligned shapes too -- something the aspect-ratio heuristic
(axis-aligned-bbox-based) cannot represent at all.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from mycv.geometry import rotate_image
from mycv.shape_cnn import _resize_mask_to_input, INPUT_SIZE
from tools.degrade import DEGRADATIONS

CLASS_NAMES = ["Circle", "Square", "Rectangle"]

_BASE = 100  # generous canvas so rotation/size variation never clips


def _add_boundary_noise(mask01: np.ndarray, rng, flip_prob=0.05):
    """Flip a small fraction of boundary-adjacent pixels to mimic real
    segmentation noise (jagged edges from imperfect colour thresholding).
    Applied at NATIVE resolution (before crop+resize) -- see
    `_augment` and the module docstring for why degradation order matters."""
    binary = mask01 > 0
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    is_boundary = binary & (
        (~padded[1:-1, 2:]) | (~padded[1:-1, :-2]) |
        (~padded[2:, 1:-1]) | (~padded[:-2, 1:-1])
    )
    is_boundary |= (~binary) & (
        (padded[1:-1, 2:]) | (padded[1:-1, :-2]) |
        (padded[2:, 1:-1]) | (padded[:-2, 1:-1])
    )
    flips = is_boundary & (rng.random(binary.shape) < flip_prob)
    out = binary.copy()
    out[flips] = ~out[flips]
    return out.astype(np.float64)


# Degradation types applied at NATIVE resolution, BEFORE crop+resize --
# this matters. An earlier version applied degradations AFTER resizing to
# the final 24x24 input, at which point e.g. a nominal "3% salt-and-pepper"
# flip rate has no subsequent smoothing to dilute it (17 flipped pixels
# out of 576, each one fully visible to the model). Real segmentation
# noise happens at the camera's native resolution and then gets partially
# averaged away by the bilinear-resize-then-threshold step down to 24x24
# -- degrading post-resize was a harsher, less representative corruption
# than what the model actually sees in practice, and (confirmed by
# benchmarking) produced a systematic train/eval mismatch specifically
# for salt-and-pepper noise. `tools/benchmark_shape_cnn.py` evaluates
# with degradations applied at native resolution for the same reason --
# this generator now matches that order exactly.
_AUGMENT_POOL = ["clean", "clean", "boundary_jitter", "boundary_jitter"] + list(DEGRADATIONS.keys())


def _augment(mask01, rng):
    choice = rng.choice(_AUGMENT_POOL)
    if choice == "clean":
        return mask01
    if choice == "boundary_jitter":
        return _add_boundary_noise(mask01, rng, flip_prob=0.05)
    return DEGRADATIONS[choice](mask01, rng)


def _crop_to_bbox(binary_mask: np.ndarray) -> np.ndarray:
    """Crop to the tight foreground bounding box, at native resolution
    (no resize) -- used so degradations can be applied at native
    resolution, matching `tools/benchmark_shape_cnn.py`'s evaluation
    order (and reality: real segmentation noise happens at the camera's
    native resolution, THEN gets partially smoothed away by the resize
    to the CNN's 24x24 input -- degrading AFTER resizing, as an earlier
    version of this generator did, applies the same nominal noise
    fraction with no such smoothing, making it a much harsher and less
    representative corruption than what the model will actually see)."""
    ys, xs = np.where(binary_mask > 0)
    if ys.size == 0:
        return binary_mask
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return binary_mask[y1:y2, x1:x2]


def _crop_to_bbox_and_resize(binary_mask: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """
    Crop to the tight bounding box of the foreground, then resize to
    `size` x `size` using the SAME resize helper `shape_cnn` uses at
    inference time -- this is what makes training match real usage.
    """
    cropped = _crop_to_bbox(binary_mask)
    return _resize_mask_to_input(cropped, size=size)


def _draw_circle(rng):
    r = rng.uniform(10.0, 35.0)
    cx = _BASE / 2 + rng.uniform(-3, 3)
    cy = _BASE / 2 + rng.uniform(-3, 3)
    yy, xx = np.mgrid[0:_BASE, 0:_BASE]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.float64)


def _draw_rect(rng, force_square):
    if force_square:
        w = h = rng.uniform(20.0, 50.0)
    else:
        w = rng.uniform(18.0, 55.0)
        ratio = rng.choice([rng.uniform(1.4, 2.2), rng.uniform(1 / 2.2, 1 / 1.4)])
        h = np.clip(w / ratio, 10.0, 60.0)

    canvas = np.zeros((_BASE, _BASE), dtype=np.float64)
    cy = cx = _BASE / 2
    y0, y1 = int(cy - h / 2), int(cy + h / 2)
    x0, x1 = int(cx - w / 2), int(cx + w / 2)
    canvas[y0:y1, x0:x1] = 255.0

    # A square has 90-degree rotational symmetry, so [0, 90) covers every
    # DISTINCT appearance -- but a CNN has no built-in reflection
    # invariance, so a range like [0, 45) (an earlier version of this
    # generator) leaves the model having never seen the mirror-image half
    # of that range. Confirmed by direct testing: a square rotated 60
    # degrees is the exact pixel-for-pixel mirror of one rotated 30
    # degrees (fliplr matches exactly), yet a model trained only on
    # [0, 45) confidently mis-called the 60-75 degree range "Rectangle".
    # [0, 90) has no such gap. A non-square rectangle only has 180-degree
    # symmetry and no such shortcut, so it gets the fuller [-90, 90) range.
    angle = rng.uniform(0, 90) if force_square else rng.uniform(-90, 90)
    rotated = rotate_image(canvas, angle)
    return (rotated >= 127).astype(np.float64)


def generate_shape_dataset(n_per_class=1200, size=INPUT_SIZE, seed=0, noise=True):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * 3, 1, size, size), dtype=np.float64)
    y = np.zeros(n_per_class * 3, dtype=np.int64)

    idx = 0
    for _ in range(n_per_class):
        raw = _draw_circle(rng)
        if noise:
            raw = _augment(raw, rng)          # degrade at NATIVE resolution
        m = _resize_mask_to_input(_crop_to_bbox(raw), size)   # then crop+resize
        X[idx, 0] = m
        y[idx] = 0
        idx += 1

    for _ in range(n_per_class):
        raw = _draw_rect(rng, force_square=True)
        if raw.sum() < 4:
            raw = _draw_rect(rng, force_square=True)
        if noise:
            raw = _augment(raw, rng)
        m = _resize_mask_to_input(_crop_to_bbox(raw), size)
        X[idx, 0] = m
        y[idx] = 1
        idx += 1

    for _ in range(n_per_class):
        raw = _draw_rect(rng, force_square=False)
        if raw.sum() < 4:
            raw = _draw_rect(rng, force_square=False)
        if noise:
            raw = _augment(raw, rng)
        m = _resize_mask_to_input(_crop_to_bbox(raw), size)
        X[idx, 0] = m
        y[idx] = 2
        idx += 1

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = generate_shape_dataset(n_per_class=5, size=24, seed=1)
    for i in range(len(X)):
        print(CLASS_NAMES[y[i]])
        for row in X[i, 0]:
            print("".join("#" if v > 0 else "." for v in row))
        print()
