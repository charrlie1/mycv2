# mycv — Pure-NumPy Computer Vision Library

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-green.svg)](https://numpy.org)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

**mycv** is a dependency-free computer vision library implemented entirely in pure NumPy.
No OpenCV, no scikit-image — just vectorised linear algebra, set theory, and probability,
derived from first principles.

---

## Installation

```bash
git clone https://github.com/charrlie1/mycv.git
cd mycv
pip install numpy          # core only
pip install pillow         # for the static demo
pip install pygame-ce av   # for the live demo (optional)
```

---

## Quick Start

```python
import numpy as np
import mycv

gray   = mycv.rgb_to_grayscale(rgb)
edges  = mycv.sobel_edge_detection(gray)
motion = mycv.compute_motion_mask(frame_t, frame_prev, threshold=30)
cmask  = mycv.color_mask(hsv, lower=[0,.3,.3], upper=[60,1,1])
cx, cy = mycv.calculate_centroid(cmask)

tracker = mycv.KalmanCentroidTracker()
pos     = tracker.update((cx, cy))

multi   = mycv.MultiObjectKalmanTracker()
tracks  = multi.update([(cx1,cy1),(cx2,cy2)])   # [(id,x,y), ...]

H, mask = mycv.ransac_homography(src_pts, dst_pts, inlier_threshold=3.0)
```

---

## Modules

| Module | Key functions |
|---|---|
| `core` | `rgb_to_grayscale`, `threshold` |
| `color` | `histogram_equalize`, `rgb_to_hsv` |
| `filters` | `convolve2d`, `sobel_edge_detection` |
| `morphology` | `dilate`, `erode`, `opening`, `closing`, `grayscale_dilate`, `label_connected_components`, `component_properties` |
| `geometry` | `rotate_image`, `warp_perspective`, `bilinear_interpolate` |
| `features` | `harris_corner_response`, `detect_harris_corners`, `hough_line_transform`, `count_hough_lines`, `convex_hull`, `extract_object_metrics`, `draw_bounding_box`, `classify_object` |
| `detection` | `match_template_ncc`, `match_template_ncc_multiscale`, `find_template_matches`, `gaussian_pyramid`, `non_max_suppression` |
| `tracking` | `compute_motion_mask`, `color_mask`, `color_mask_hue_wrap`, `calculate_centroid`, `TemporalSmoother`, `KalmanCentroidTracker`, `MultiObjectKalmanTracker`, `mahalanobis_gate` |
| `calibration` | `normalize_points`, `solve_homography_dlt`, `reprojection_error`, `ransac_homography` |
| `streaming` | `StreamReader` |

---

## Running the Demos

```bash
# Static pipeline — 23 output images
cp /path/to/photo.jpg examples/test_image.jpg
python examples/main.py

# Live real-time demo
python examples/live_demo.py --source synthetic --mode color   # no camera
python examples/live_demo.py --source camera   --mode motion   # webcam
python examples/live_demo.py --source video.mp4               # file
python examples/live_demo.py --source synthetic --headless     # no display
```

**Live demo keyboard controls:**
`c` colour · `m` motion · `t` template · `s` sample colour · `g/r/b/y` presets ·
`o` overlay · `p` morphology · `=/-` NCC threshold · `[/]` motion threshold ·
`←/→` template size · `q` quit

---

## Tests

```bash
pip install pytest
pytest            # all tests
pytest -q         # quiet
pytest tests/test_calibration.py   # single module
```

120+ assertions across 10 test files. Passes with NumPy only — no camera or display required.

---

## Mathematical Documentation

Complete derivations for every algorithm are in **[mycv_math_v4.pdf](mycv_math_v4.pdf)**:
BT.601 · Histogram equalisation · HSV geometry · Convolution theory · Sobel operator ·
Morphological set theory · Affine & projective geometry · Bilinear interpolation ·
Homography (convergence proof) · Harris structure tensor · Hough transform ·
NCC as Pearson correlation · Gaussian pyramid & Nyquist · IoU / Jaccard ·
uint8 underflow & the int16 fix · Centroid as expected value · Ring buffer O(1) analysis ·
Kalman filter predict/update · Mahalanobis gate · DLT homography · RANSAC.

---

## Author

**Abodunrin Charles Toluwanimi**  
Department of Electrical and Electronic Engineering  
Obafemi Awolowo University, Ile-Ife  
GitHub: [@charrlie1](https://github.com/charrlie1)

## License

[MIT](LICENSE)
