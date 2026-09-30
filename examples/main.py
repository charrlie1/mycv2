#!/usr/bin/env python3
"""
examples/main.py  —  static 23-step pipeline for mycv v4.2.1
Usage:
    cp /path/to/photo.jpg examples/test_image.jpg
    python examples/main.py
"""
import sys, numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mycv

HERE = Path(__file__).parent

def save(arr, name):
    Image.fromarray(arr).save(str(HERE / name))
    print(f"  ✓ {name}")

def u8(arr):
    mn, mx = float(arr.min()), float(arr.max())
    return ((arr-mn)/(mx-mn+1e-9)*255).astype(np.uint8)

def colour_labels(lmap, n):
    out = np.zeros((*lmap.shape,3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    cols = rng.integers(60,230,(n+1,3),dtype=np.uint8); cols[0]=0
    for l in range(n+1): out[lmap==l] = cols[l]
    return out

def main():
    src = HERE / "test_image.jpg"
    if not src.exists():
        print("ERROR: place test_image.jpg next to main.py"); sys.exit(1)

    print(f"\nmycv v{mycv.__version__}  |  {src.name}")
    print("="*55)
    rgb  = np.array(Image.open(src).convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]

    gray = mycv.rgb_to_grayscale(rgb);                  save(gray,   "output_01_grayscale.jpg")
    save(mycv.histogram_equalize(gray),                              "output_02_equalized.jpg")
    hsv  = mycv.rgb_to_hsv(rgb)
    save((hsv[...,2]*255).astype(np.uint8),                         "output_03_hsv_value.jpg")
    save((hsv[...,1]*255).astype(np.uint8),                         "output_04_hsv_saturation.jpg")

    ed   = mycv.sobel_edge_detection(gray)
    edge = ed["magnitude"]; Gx, Gy = ed["Gx"], ed["Gy"]
    save(edge,                                                       "output_05_edges.jpg")
    binary = mycv.threshold(edge, tau=80);              save(binary, "output_06_threshold.jpg")

    se = np.ones((3,3), dtype=np.bool_)
    save(mycv.dilate(binary, se),                                    "output_07_dilated.jpg")
    save(mycv.erode(binary, se),                                     "output_08_eroded.jpg")
    save(mycv.opening(binary, se),                                   "output_09_opened.jpg")
    save(mycv.closing(binary, se),                                   "output_10_closed.jpg")
    save(mycv.grayscale_dilate(gray, size=5),                        "output_11_gray_dilate.jpg")

    labels, n = mycv.label_connected_components(binary)
    save(colour_labels(labels, n),                                   "output_12_labeled.jpg")
    print(f"     {n} connected components")

    save(u8(mycv.harris_corner_response(Gx.astype(float),Gy.astype(float))), "output_13_harris.jpg")
    corners = mycv.detect_harris_corners(Gx.astype(float), Gy.astype(float))
    print(f"     {len(corners)} Harris corners")

    hough = mycv.hough_line_transform(binary, n_thetas=180)
    save(u8(hough["accumulator"].astype(float)),                     "output_14_hough_accum.jpg")
    print(f"     {mycv.count_hough_lines(binary)['count']} Hough lines")

    save(mycv.rotate_image(gray, 30.),                               "output_15_rotated.jpg")
    Hmat = np.array([[1.,.3,0.],[0.,1.,0.],[0.,.001,1.]])
    save(mycv.warp_perspective(gray, Hmat, gray.shape),              "output_16_homography.jpg")

    tpl = gray[:32,:32]
    save(u8(mycv.match_template_ncc(gray, tpl)),                     "output_17_ncc_map.jpg")
    pyr = mycv.gaussian_pyramid(gray, levels=4)
    save(u8(pyr[1]),                                                 "output_18_pyramid_l1.jpg")
    save(u8(pyr[2]),                                                 "output_19_pyramid_l2.jpg")

    rng   = np.random.default_rng(7)
    noise = rng.integers(-40,40,gray.shape,dtype=np.int16)
    prev  = np.clip(gray.astype(np.int16)+noise,0,255).astype(np.uint8)
    save(mycv.compute_motion_mask(gray, prev, threshold=30),         "output_20_motion_mask.jpg")

    cmask = mycv.color_mask(hsv, np.array([0.,.3,.3],dtype=np.float32),
                                  np.array([60.,1.,1.],dtype=np.float32))
    save(cmask,                                                      "output_21_color_mask.jpg")
    cx,cy = mycv.calculate_centroid(cmask)
    print(f"     colour centroid: ({cx:.1f}, {cy:.1f})")

    smoother = mycv.TemporalSmoother(5, H, W, 3)
    for _ in range(5):
        fn = rng.integers(-15,15,rgb.shape,dtype=np.int16)
        smoothed = smoother.update(np.clip(rgb.astype(np.int16)+fn,0,255).astype(np.uint8))
    save(smoothed,                                                   "output_22_temporal_smooth.jpg")

    rng2 = np.random.default_rng(0)
    Ht   = np.array([[1.2,.3,15.],[.1,1.,10.],[.001,.002,1.]])
    sp   = rng2.uniform(10,min(H,W)-10,(40,2))
    sh   = np.column_stack([sp,np.ones(40)])
    dh   = (Ht@sh.T).T; dp = dh[:,:2]/dh[:,2:3]
    os_  = rng2.uniform(0,min(H,W),(10,2))
    od   = rng2.uniform(0,min(H,W),(10,2))
    H_est,mask = mycv.ransac_homography(np.vstack([sp,os_]),np.vstack([dp,od]),
                                         n_iterations=500,inlier_threshold=3.)
    err = float(np.mean(mycv.reprojection_error(H_est, sp, dp)))
    print(f"     RANSAC: {mask.sum()}/50 inliers, reprojection error={err:.4f}px")
    save(mycv.warp_perspective(gray, H_est, gray.shape),             "output_23_calibration.jpg")

    print("\nDone — 23 images written.")

if __name__ == "__main__":
    main()
