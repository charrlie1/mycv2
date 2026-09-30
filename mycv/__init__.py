"""
mycv — Custom Computer Vision Library  v4.2.1
=============================================
A pure-NumPy image processing and computer vision library built from
mathematical first principles. No OpenCV. No SciPy. No skimage.

Modules
-------
core        : Point ops         — rgb_to_grayscale, threshold
color       : Photometrics      — histogram_equalize, rgb_to_hsv
filters     : Spatial filters   — convolve2d, sobel_edge_detection
morphology  : Binary set ops    — dilate, erode, opening, closing,
                                   grayscale_dilate, connected components
geometry    : Transformations   — rotate_image, warp_perspective, bilinear_interpolate
features    : Feature extract   — harris_corner_response, detect_harris_corners,
                                   hough_line_transform, count_hough_lines,
                                   extract_object_metrics, convex_hull,
                                   draw_bounding_box, classify_object
detection   : Detection         — match_template_ncc, match_template_ncc_multiscale,
                                   gaussian_pyramid, non_max_suppression
tracking    : Motion & tracking — compute_motion_mask, color_mask, color_mask_hue_wrap,
                                   color_mask_adaptive, calculate_centroid,
                                   TemporalSmoother, KalmanCentroidTracker,
                                   MultiObjectKalmanTracker
calibration : Projective geom.  — solve_homography_dlt, ransac_homography
streaming   : Live video (opt.) — StreamReader (requires the optional `av` package)
nn          : Neural nets       — Conv2D, MaxPool2D, Dense, ReLU, Sequential,
                                   SGD, softmax_cross_entropy (see module docstring)
shape_cnn   : Trained shape CNN — predict_shape, load_shape_cnn (Circle/Square/
                                   Rectangle classifier that AUGMENTS
                                   classify_object; bundled pretrained weights)

Changelog
---------
v4.0.0 -> v4.1.0 — corrections:
  - features.extract_object_metrics: bbox width/height are now correctly
    inclusive (+1); the exported bbox convention is now EXCLUSIVE
    (y1, x1, y2, x2) and standardised across features/detection/morphology,
    so `image[y1:y2, x1:x2]` always crops the box directly.
  - features.extract_object_metrics: now returns true `pixel_area`
    (foreground pixel count) distinct from `bbox_area` (rectangle area).
  - features.classify_object: now accepts an optional `mask` and computes
    mean colour from object pixels only, not the whole bounding box.
  - features.draw_bounding_box: rewritten for the new exclusive convention.
  - tracking.KalmanCentroidTracker.predict(): now COMMITS the predicted
    state, so repeated calls during occlusion correctly propagate the
    object forward instead of re-predicting from stale state.
  - tracking.kalman_filter_update: uses np.linalg.solve instead of an
    explicit matrix inverse, and the numerically robust Joseph-form
    covariance update by default.
  - tracking.KalmanCentroidTracker: supports real per-frame `dt`.

v4.0.0 -> v4.1.0 — new capabilities:
  - True pixel area, extent, second moments, orientation, eccentricity,
    perimeter, circularity, and solidity (via a new monotone-chain
    `convex_hull`) in `extract_object_metrics`.
  - `detect_harris_corners` (thresholded, NMS'd corner points) and
    `count_hough_lines` (peak-clustered line count).
  - `morphology.label_connected_components` + `component_properties`
    for robust multi-object colour tracking.
  - `tracking.MultiObjectKalmanTracker` for tracking several objects
    with persistent IDs.
  - Kalman measurement gating (`mahalanobis_gate`) and adaptive
    measurement noise (`confidence` argument to `update`).
  - `detection.match_template_ncc_multiscale` for scale-invariant
    template matching via the Gaussian pyramid.
  - `detection.match_template_ncc` now guards against unbounded memory
    use on large frames/templates.
  - `tracking.color_mask_hue_wrap` automatically handles hue wrap-around
    (e.g. red spanning 0/360 degrees).
  - New `mycv.calibration` module: normalised DLT homography estimation
    and RANSAC for outlier-robust homography fitting.
  - New optional `mycv.streaming` module for RTSP/HTTP/UDP video
    ingestion with reconnection and frame-dropping (requires `av`).

v4.1.1: `classify_object` gained a real Circle category, using
  `circularity >= 0.85 AND extent <= 0.88` (both signals required —
  circularity alone is unreliable for small shapes; see the function's
  docstring). New `morphology.select_component` to pick a connected
  component by nearest point or largest area — the missing link between
  a locator (e.g. a template match) and a real measurement.

v4.1.2: New `tracking.color_mask_adaptive` — hue is numerically unstable
  for near-white/gray/black colours (small denominator in the hue
  formula amplifies sensor noise into ~random hue values), so a sampled
  achromatic colour now segments via Saturation+Value thresholds instead
  of a hue window. Saturated colours are unaffected.

v4.2.0: New `mycv.nn` module — a small, pure-NumPy, from-scratch neural
  network toolkit (Conv2D, MaxPool2D/AvgPool2D, Dense, ReLU,
  softmax_cross_entropy, SGD, a Sequential container). Every layer's
  backward pass is verified against numerical finite-difference gradient
  checks (see the project's test suite). New `mycv.shape_cnn` module: a
  small trained CNN (Circle/Square/Rectangle, ~24x24 input, bundled
  pretrained weights) that AUGMENTS `classify_object`'s heuristic shape
  classification — notably, unlike the axis-aligned-bbox-based heuristic,
  the CNN correctly recognises ROTATED squares/rectangles, since it was
  trained on rotated shapes.

v4.2.1: Diagnosed via a new benchmark tool (tools/benchmark_shape_cnn.py)
  rather than guessing, per a structured data->model->evaluation plan.
  Found and fixed two real bugs in the shape_cnn training data:
  (1) squares were only trained on rotation angles [0,45), so the model
  never saw the mirror-image half of a square's rotation space (a square
  rotated 60 deg is the exact pixel-mirror of one rotated 30 deg -- CNNs
  have no built-in reflection invariance, so it confidently mis-called
  60-75 degree rotations "Rectangle"). Fixed by training on the full
  [0,90) range. (2) degradation augmentation (salt-and-pepper noise,
  holes, stray blobs, erosion/dilation, blur) was applied AFTER resizing
  to the 24x24 input, giving it no chance to be smoothed away the way it
  would be in reality (native-resolution noise -> resize) -- this alone
  took salt-and-pepper accuracy from 33% (chance level) to 85%. Retrained
  model: 96.0% on a combined rotation+degradation+small-size stress test
  (up from 82.7%). `predict_shape` now raises ValueError on an empty/
  all-background mask instead of silently predicting on a blank input.
  New `tools/degrade.py` (realistic segmentation-failure-mode functions),
  `tools/benchmark_shape_cnn.py` (confusion matrices, per-degradation and
  rotation-binned accuracy, precision/recall/F1, real-mask evaluation),
  and `tools/capture_shape_masks.py` (collects real segmented masks from
  a live camera for actual -- not synthetic-only -- evaluation; untested
  against a real camera in this environment). Model weights now ship
  with `mycv/models/shape_cnn_metadata.json` recording architecture,
  training config, and measured accuracy for reproducibility.
"""

from .core       import rgb_to_grayscale, threshold
from .color      import histogram_equalize, rgb_to_hsv
from .filters    import convolve2d, sobel_edge_detection, SOBEL_X, SOBEL_Y
from .morphology import (
    dilate, erode, opening, closing,
    grayscale_dilate,
    label_connected_components, component_properties, select_component,
)
from .geometry   import bilinear_interpolate, rotate_image, warp_perspective
from .features   import (
    harris_corner_response, detect_harris_corners,
    hough_line_transform, count_hough_lines,
    extract_object_metrics, convex_hull,
    draw_bounding_box, classify_object,
)
from .detection  import (
    match_template_ncc, find_template_matches, match_template_ncc_multiscale,
    gaussian_pyramid, non_max_suppression,
)
from .tracking   import (
    compute_motion_mask,
    color_mask, color_mask_hue_wrap, color_mask_adaptive, calculate_centroid,
    TemporalSmoother,
    kalman_filter_predict, kalman_filter_update, mahalanobis_gate,
    KalmanCentroidTracker, MultiObjectKalmanTracker,
)
from .calibration import (
    normalize_points, solve_homography_dlt, reprojection_error, ransac_homography,
)
from . import nn
from .shape_cnn import predict_shape, load_shape_cnn, build_shape_cnn
# .streaming is intentionally NOT imported here: it depends on the
# optional `av` package and importing mycv should never require it.
# Access it explicitly: `from mycv.streaming import StreamReader`.

__version__ = "4.2.1"
__author__  = "Tolu"

__all__ = [
    "rgb_to_grayscale", "threshold",
    "histogram_equalize", "rgb_to_hsv",
    "convolve2d", "sobel_edge_detection", "SOBEL_X", "SOBEL_Y",
    "dilate", "erode", "opening", "closing",
    "grayscale_dilate", "label_connected_components", "component_properties", "select_component",
    "bilinear_interpolate", "rotate_image", "warp_perspective",
    "harris_corner_response", "detect_harris_corners",
    "hough_line_transform", "count_hough_lines",
    "extract_object_metrics", "convex_hull",
    "draw_bounding_box", "classify_object",
    "match_template_ncc", "find_template_matches", "match_template_ncc_multiscale",
    "gaussian_pyramid", "non_max_suppression",
    "compute_motion_mask",
    "color_mask", "color_mask_hue_wrap", "color_mask_adaptive", "calculate_centroid",
    "TemporalSmoother",
    "kalman_filter_predict", "kalman_filter_update", "mahalanobis_gate",
    "KalmanCentroidTracker", "MultiObjectKalmanTracker",
    "normalize_points", "solve_homography_dlt", "reprojection_error", "ransac_homography",
    "nn", "predict_shape", "load_shape_cnn", "build_shape_cnn",
]
