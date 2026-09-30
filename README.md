# mycv

**A from-scratch computer vision and image processing library built with NumPy.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/array%20backend-NumPy-013243)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`mycv` brings common vision algorithms together in one small Python package: filters, morphology, shape features, template detection, object tracking, homography estimation, and a small neural-network toolkit. The algorithms are implemented with NumPy rather than relying on OpenCV, SciPy, or scikit-image. Optional demo tools use Pillow, pygame-ce, and PyAV.

> **Version:** 4.2.1 · **Python:** 3.9+ · **Core dependency:** NumPy 1.22+

## Get started

Install the library from the repository:

```bash
git clone https://github.com/charrlie1/mycv2.git
cd mycv2
python -m pip install .
```

Or install the current checkout in editable mode while developing:

```bash
python -m pip install -e ".[dev]"
```

Optional dependency groups:

| Install | Includes | Use it for |
|---|---|---|
| `python -m pip install ".[demo]"` | pygame-ce, PyAV, Pillow | Live camera/video demo, network streams, and image demo |
| `python -m pip install ".[dev]"` | pytest, pytest-cov, Black, Ruff | Tests and development tools |
| `python -m pip install ".[notebook]"` | Jupyter, Matplotlib, Pillow | Notebook exploration and plots |

The core package only needs NumPy. PyAV is needed for network streams and some video sources; a camera and a graphical display are only needed for interactive camera use.

## Quick start

Images are NumPy arrays. For the core image functions, use RGB images shaped `(height, width, 3)` and grayscale images shaped `(height, width)`, typically with `uint8` values from 0 to 255.

```python
import numpy as np
import mycv

# A small synthetic RGB image, with a bright square in the middle.
rgb = np.zeros((64, 64, 3), dtype=np.uint8)
rgb[20:44, 20:44] = (255, 255, 255)

gray = mycv.rgb_to_grayscale(rgb)
edges = mycv.sobel_edge_detection(gray)
mask = mycv.threshold(gray, tau=127)

print(gray.shape, edges.shape, np.unique(mask))
```

Compare two grayscale frames to find motion, then measure its centroid:

```python
before = np.zeros((64, 64), dtype=np.uint8)
after = before.copy()
after[20:32, 25:37] = 255

motion = mycv.compute_motion_mask(after, before, threshold=30)
centroid = mycv.calculate_centroid(motion)
print(centroid)  # (x, y), or None when the mask has no foreground pixels

tracker = mycv.KalmanCentroidTracker()
position = tracker.update(centroid) if centroid is not None else tracker.predict()
```

`mycv.MultiObjectKalmanTracker.update(detections)` accepts a list of `(x, y)` detections and returns a mapping from persistent track IDs to positions.

## What’s included

| Area | Examples |
|---|---|
| Image operations | RGB-to-grayscale conversion, thresholding, histogram equalization, convolution, Sobel edges |
| Morphology | Dilation, erosion, opening, closing, connected components, component selection and properties |
| Geometry | Image rotation, bilinear interpolation, perspective warping, homography estimation with DLT and RANSAC |
| Features | Harris corners, Hough lines, convex hulls, bounding boxes, object measurements and heuristic classification |
| Detection | Normalized cross-correlation template matching, multi-scale matching, Gaussian pyramids, non-maximum suppression |
| Tracking | Motion and HSV color masks, temporal smoothing, single- and multi-object Kalman tracking |
| Neural networks | NumPy `Conv2D`, pooling, `Dense`, `ReLU`, `Sequential`, SGD and softmax cross-entropy |
| Shape CNN | Bundled pretrained Circle/Square/Rectangle classifier to complement heuristic shape labels |
| Streaming | Optional PyAV `StreamReader` for video and network streams |

The `mycv.shape_cnn.predict_shape(mask)` classifier expects a non-empty, cropped binary mask for a single object. It predicts Circle, Square, or Rectangle; it is intended to augment the library’s geometric shape heuristic, not replace general-purpose object recognition.

For exported names and detailed algorithm notes, see the module docstrings in [`mycv/`](mycv/) and the [module index](docs/index.md).

## Run the demos

Install the demo tools first:

```bash
python -m pip install ".[demo]"
```

The static example reads `examples/test_image.jpg`; add an image at that path before running it:

```bash
python examples/main.py
```

The live demo defaults to a synthetic source, so it can be started without a camera:

```bash
python examples/live_demo.py
python examples/live_demo.py --source synthetic --headless
```

Other sources include a camera, a camera index, a video file, or a stream URL:

```bash
python examples/live_demo.py --source camera --mode motion
python examples/live_demo.py --source 0
python examples/live_demo.py --source path/to/video.mp4
python examples/live_demo.py --source "rtsp://host:554/stream"
```

Use `python examples/live_demo.py --help` to see startup options. In the interactive window, press **h** for the full controls. Useful keys include **c** for color mode, **m** for motion mode, **t** to capture a template, **k** to toggle Kalman filtering, **u** for multi-object tracking, **i** for object information, **v** for the shape CNN, and **q** or **Esc** to quit.

## Run the test suite

```bash
python -m pip install ".[dev]"
python -m pytest
```

The unit tests run without a physical camera or display. The optional streaming tests need PyAV.

## License

This project is licensed under the [MIT License](LICENSE).
