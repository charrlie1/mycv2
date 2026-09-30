"""
tools/degrade.py
================
Segmentation-realistic degradation functions for synthetic shape masks:
erosion/dilation, boundary jitter, holes, small stray blobs, salt-and-pepper
noise, blur-then-rethreshold, and low-resolution quantization.

These simulate the actual failure modes of a real colour/motion
segmentation pipeline (mycv.tracking.color_mask_adaptive -> morphology
cleanup -> connected components), not just "add random noise" -- so a
model benchmarked against them is a much better proxy for real-mask
performance than clean synthetic shapes alone.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from mycv.morphology import dilate as mycv_dilate, erode as mycv_erode
from mycv.filters import convolve2d


def to255(mask01):
    return (mask01 > 0).astype(np.uint8) * 255


def apply_erosion(mask01, size=3):
    m = to255(mask01)
    kernel = np.ones((size, size), dtype=np.bool_)
    return (mycv_erode(m, kernel) > 0).astype(np.float64)


def apply_dilation(mask01, size=3):
    m = to255(mask01)
    kernel = np.ones((size, size), dtype=np.bool_)
    return (mycv_dilate(m, kernel) > 0).astype(np.float64)


def apply_salt_and_pepper(mask01, rng, prob=0.03):
    binary = mask01 > 0
    flips = rng.random(binary.shape) < prob
    out = binary.copy()
    out[flips] = ~out[flips]
    return out.astype(np.float64)


def apply_boundary_jitter(mask01, rng, flip_prob=0.08):
    """Heavier version of the training-time boundary noise: flips a larger
    fraction of boundary-adjacent pixels, simulating a noisier segmentation."""
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


def apply_holes(mask01, rng, n_holes=3, max_radius=4):
    """Punch a few small circular holes into the mask (patchy segmentation)."""
    out = (mask01 > 0).copy()
    H, W = out.shape
    ys, xs = np.where(out)
    if ys.size == 0:
        return out.astype(np.float64)
    for _ in range(n_holes):
        idx = rng.integers(0, ys.size)
        cy, cx = ys[idx], xs[idx]
        r = rng.integers(1, max_radius + 1)
        yy, xx = np.mgrid[0:H, 0:W]
        hole = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        out[hole] = False
    return out.astype(np.float64)


def apply_stray_blobs(mask01, rng, n_blobs=2, max_radius=3):
    """Add a few small disconnected stray foreground blobs near the object
    (e.g. specular highlight fragments, adjacent similarly-coloured noise)."""
    out = (mask01 > 0).copy()
    H, W = out.shape
    ys, xs = np.where(out)
    if ys.size == 0:
        return out.astype(np.float64)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    for _ in range(n_blobs):
        cy = rng.integers(max(0, y1 - 5), min(H, y2 + 5))
        cx = rng.integers(max(0, x1 - 5), min(W, x2 + 5))
        r = rng.integers(1, max_radius + 1)
        yy, xx = np.mgrid[0:H, 0:W]
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        out[blob] = True
    return out.astype(np.float64)


def apply_blur_rethreshold(mask01, sigma_kernel=3):
    """Simple box-blur-then-rethreshold, approximating a slightly
    out-of-focus / low-contrast segmentation boundary."""
    m = (mask01 > 0).astype(np.float64)
    k = sigma_kernel
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    blurred = convolve2d(m, kernel, padding="same")
    return (blurred > 0.5).astype(np.float64)


def apply_low_res_quantization(mask01, factor=3):
    """Nearest-neighbour downsample-then-upsample, simulating a coarse
    native segmentation resolution (matches live_demo's resize_nearest to
    a small max_dim before component extraction)."""
    H, W = mask01.shape
    small_h, small_w = max(1, H // factor), max(1, W // factor)
    row_idx = np.linspace(0, H - 1, small_h).astype(np.int64)
    col_idx = np.linspace(0, W - 1, small_w).astype(np.int64)
    small = mask01[row_idx][:, col_idx]
    row_idx2 = np.linspace(0, small_h - 1, H).astype(np.int64)
    col_idx2 = np.linspace(0, small_w - 1, W).astype(np.int64)
    return small[row_idx2][:, col_idx2]


DEGRADATIONS = {
    "clean": lambda m, rng: m,
    "erosion": lambda m, rng: apply_erosion(m, size=3),
    "dilation": lambda m, rng: apply_dilation(m, size=3),
    "salt_pepper": lambda m, rng: apply_salt_and_pepper(m, rng, prob=0.03),
    "boundary_jitter": lambda m, rng: apply_boundary_jitter(m, rng, flip_prob=0.08),
    "holes": lambda m, rng: apply_holes(m, rng, n_holes=3, max_radius=4),
    "stray_blobs": lambda m, rng: apply_stray_blobs(m, rng, n_blobs=2, max_radius=3),
    "blur": lambda m, rng: apply_blur_rethreshold(m, sigma_kernel=3),
    "low_res": lambda m, rng: apply_low_res_quantization(m, factor=3),
}
