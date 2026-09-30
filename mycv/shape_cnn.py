"""
mycv.shape_cnn
================
A ready-to-use tiny CNN that classifies a binary object mask as
Circle / Square / Rectangle, built on `mycv.nn`.

This exists to AUGMENT `features.classify_object`'s aspect-ratio +
circularity heuristic, not replace it: the heuristic is fast, exact, and
easy to reason about, but it is fundamentally axis-aligned-bounding-box
based, so it structurally cannot recognise a ROTATED square/rectangle as
what it is (a 45-degree-rotated square has a ~1:1 axis-aligned bbox with
extent ~0.5, which the heuristic can only ever call "not filling its
box" -- it has no notion of rotation at all). The bundled CNN was trained
on rotated as well as axis-aligned shapes and correctly recognises both,
which is the concrete capability the heuristic is missing.

Architecture (fixed -- must match the bundled weights in
`mycv/models/shape_cnn.npz`):

    Conv2D(1->8, 3x3, same) -> ReLU -> MaxPool(2)      # 24x24 -> 12x12
    Conv2D(8->16, 3x3, same) -> ReLU -> MaxPool(2)      # 12x12 -> 6x6
    Flatten -> Dense(576->32) -> ReLU -> Dense(32->3)

Trained on ~4500 synthetic 24x24 binary shape crops (see
`gen_shape_dataset.py` in the mycv repo for the generator), with random
size, position jitter, rotation, and boundary noise -- reaching 100%
validation accuracy on a held-out synthetic set. Being purely synthetic,
it will not be perfect on real segmented masks; treat its prediction as
a second opinion alongside (not a replacement for) the heuristic label.

Functions
---------
build_shape_cnn   : construct a fresh (untrained) model with the fixed
                     architecture above
load_shape_cnn    : build + load the bundled (or a custom) trained weights
predict_shape     : classify a single binary mask crop
"""

import os
import numpy as np

from . import nn
from .geometry import bilinear_interpolate

CLASS_NAMES = ["Circle", "Square", "Rectangle"]
INPUT_SIZE = 24

_DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "shape_cnn.npz")

_cached_model = None


def build_shape_cnn(seed: int = 0) -> "nn.Sequential":
    """
    Construct a fresh model with the fixed architecture the bundled
    weights were trained with. Untrained (random init) unless you call
    `load_weights` on the result yourself -- most callers want
    `load_shape_cnn` instead.
    """
    return nn.Sequential([
        nn.Conv2D(1, 8, kernel_size=3, stride=1, padding="same", seed=seed + 1),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Conv2D(8, 16, kernel_size=3, stride=1, padding="same", seed=seed + 2),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Flatten(),
        nn.Dense(16 * 6 * 6, 32, seed=seed + 3),
        nn.ReLU(),
        nn.Dense(32, len(CLASS_NAMES), seed=seed + 4),
    ])


def load_shape_cnn(weights_path: str = None) -> "nn.Sequential":
    """
    Build the shape-classification model and load trained weights into it.

    Parameters
    ----------
    weights_path : str, optional -- defaults to the weights bundled with
        mycv (`mycv/models/shape_cnn.npz`). Pass your own path if you've
        retrained the model (e.g. on real captured crops rather than the
        synthetic dataset it ships with).

    Returns
    -------
    mycv.nn.Sequential -- ready for `.predict(x)`

    Raises
    ------
    FileNotFoundError if no weights file is found at `weights_path`.
    """
    path = weights_path or _DEFAULT_WEIGHTS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No shape-CNN weights found at '{path}'. The bundled weights "
            f"should ship at mycv/models/shape_cnn.npz -- if you deleted or "
            f"moved them, either restore that file or pass weights_path= "
            f"to a model you've trained yourself."
        )
    model = build_shape_cnn()
    model.load_weights(path)
    return model


def _resize_mask_to_input(mask: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """
    Crop to the tight foreground bounding box, then resize to `size` x
    `size`, PRESERVING aspect ratio (letterbox-style: scale so the longer
    side maps to `size`, then centre on a zero-padded `size` x `size`
    canvas), using bilinear interpolation (re-thresholded back to binary)
    via `geometry.bilinear_interpolate`.

    Cropping to the tight bbox happens HERE, internally, rather than being
    left to the caller -- calling this (or `predict_shape`) with an
    uncropped mask (e.g. a small object on a large mostly-background
    canvas) used to silently produce a wrong prediction, because the
    model is trained exclusively on tight, edge-touching crops (that's
    what a real object's own bounding box always looks like -- see
    `features.extract_object_metrics`). Auto-cropping here makes the
    function robust regardless of whether the caller remembered to crop
    first; cropping an already-tight mask is a harmless no-op.

    Why aspect ratio must be preserved
    ------------------------------------
    An earlier version resized H and W independently to `size` x `size`
    (i.e. warped to a fixed square) -- but a solid, axis-aligned
    RECTANGLE and a solid, axis-aligned SQUARE are then BOTH just "fully
    filled" after that warp: independently stretching each axis to fit
    destroys exactly the elongation information that distinguishes a
    rectangle from a square, and the model could only tell them apart via
    incidental rotation cues. That made it a near coin-flip on the easy,
    common case of an unrotated rectangle. Scaling both axes by the SAME
    factor (this version) keeps the true width:height ratio intact -- a
    3:1 rectangle still looks like a 3:1 rectangle in the padded canvas,
    just smaller, with the shorter axis's margin filled with background.

    Normalises the input to {0.0, 1.0} via `mask > 0`, so this works
    regardless of the caller's foreground convention -- bool
    (True/False), {0, 1}, or {0, 255} all produce the same result.
    """
    if mask.size == 0:
        return np.zeros((size, size), dtype=np.float64)

    binary_full = mask > 0
    ys, xs = np.where(binary_full)
    if ys.size == 0:
        return np.zeros((size, size), dtype=np.float64)

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    binary01 = binary_full[y1:y2, x1:x2].astype(np.float64)
    H, W = binary01.shape

    scale = size / max(H, W)
    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))

    yv, xv = np.meshgrid(
        np.linspace(0, H - 1, new_H),
        np.linspace(0, W - 1, new_W),
        indexing="ij",
    )
    resized = bilinear_interpolate(binary01, xv, yv)
    resized_binary = (resized > 0.5).astype(np.float64)

    canvas = np.zeros((size, size), dtype=np.float64)
    y_off = (size - new_H) // 2
    x_off = (size - new_W) // 2
    canvas[y_off:y_off + new_H, x_off:x_off + new_W] = resized_binary
    return canvas


def predict_shape(mask: np.ndarray, model: "nn.Sequential" = None) -> dict:
    """
    Classify a single binary object mask crop as Circle / Square / Rectangle.

    Parameters
    ----------
    mask  : np.ndarray  shape (H, W) -- a binary mask crop (e.g. the
            object's own segmented mask, NOT the full frame -- crop to
            its bounding box first, as `features.extract_object_metrics`
            already gives you via `metrics["bbox"]`)
    model : mycv.nn.Sequential, optional -- a pre-loaded model (from
            `load_shape_cnn`). If omitted, the bundled model is loaded
            once and cached at module level (so repeated calls, e.g.
            once per video frame, don't reload weights from disk).

    Returns
    -------
    dict with keys:
        'label'      : str  -- the predicted class name
        'confidence' : float -- softmax probability of the predicted class
        'probs'      : dict  -- {class_name: probability} for all classes
    Raises
    ------
    ValueError if `mask` is empty or contains no foreground pixels at all
        (`mask > 0` is all False). "No object" is not Circle, Square, or
        Rectangle -- silently classifying a blank mask would produce a
        confident-looking but meaningless prediction (an all-zero CNN
        input isn't a legitimate member of any of the three classes).
    """
    global _cached_model
    if mask.size == 0 or not np.any(mask > 0):
        raise ValueError(
            "predict_shape() requires a non-empty object mask (mask > 0 "
            "must be True somewhere) -- an empty/blank mask isn't a "
            "Circle, Square, or Rectangle. Check that the caller's "
            "segmentation actually found something before calling this."
        )

    if model is None:
        if _cached_model is None:
            _cached_model = load_shape_cnn()
        model = _cached_model

    resized = _resize_mask_to_input(mask)
    x = resized[np.newaxis, np.newaxis, :, :]   # (1, 1, size, size)
    probs = model.predict(x)[0]

    idx = int(probs.argmax())
    return {
        "label": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probs": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }
