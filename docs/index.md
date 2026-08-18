# mycv v4.1.0 — Documentation

Full mathematical derivations: [mycv_math_v4.pdf](../mycv_math_v4.pdf)

## Modules
| Module | Key exports |
|---|---|
| core | rgb_to_grayscale, threshold |
| color | histogram_equalize, rgb_to_hsv |
| filters | convolve2d, sobel_edge_detection |
| morphology | dilate, erode, opening, closing, label_connected_components |
| geometry | rotate_image, warp_perspective, bilinear_interpolate |
| features | harris_corner_response, detect_harris_corners, hough_line_transform, convex_hull |
| detection | match_template_ncc, gaussian_pyramid, non_max_suppression |
| tracking | compute_motion_mask, color_mask, KalmanCentroidTracker, MultiObjectKalmanTracker |
| calibration | normalize_points, solve_homography_dlt, ransac_homography |
| streaming | StreamReader |
