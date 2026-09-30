# mycv v4.2.1 — Documentation

Algorithm explanations and parameter details live in the docstrings of the modules listed below.

| Module | Key exports |
|---|---|
| `core` | `rgb_to_grayscale`, `threshold` |
| `color` | `histogram_equalize`, `rgb_to_hsv` |
| `filters` | `convolve2d`, `sobel_edge_detection` |
| `morphology` | `dilate`, `erode`, `opening`, `closing`, connected components |
| `geometry` | `rotate_image`, `warp_perspective`, `bilinear_interpolate` |
| `features` | Harris corners, Hough lines, shape metrics, object classification |
| `detection` | template matching, Gaussian pyramids, non-maximum suppression |
| `tracking` | motion and colour masks, Kalman trackers, temporal smoothing |
| `calibration` | normalized DLT, reprojection error, RANSAC homography |
| `streaming` | optional PyAV `StreamReader` |
| `nn` / `shape_cnn` | NumPy neural network layers and pretrained shape classifier |

See the [README](../README.md) for installation, demos, and test commands.
