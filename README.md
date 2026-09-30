# mycv — Pure NumPy Computer Vision Library

`mycv` is a computer vision and image processing library implemented with NumPy. It includes image filters, morphology, geometric transforms, feature extraction, detection, tracking, camera calibration, and an optional live demo. It does not depend on OpenCV, SciPy, or scikit-image.

## Install

```bash
git clone https://github.com/charrlie1/mycv2.git
cd mycv2
python -m pip install .             # library core
python -m pip install ".[demo]"    # live and static demo dependencies
python -m pip install ".[dev]"     # tests and lint tools
```

The core library requires NumPy. The static demo also needs Pillow; the live GUI needs pygame-ce, and network streams need PyAV.

## Quick start

```python
import mycv

gray = mycv.rgb_to_grayscale(rgb)
edges = mycv.sobel_edge_detection(gray)
mask = mycv.compute_motion_mask(frame_t, frame_prev, threshold=30)
centroid = mycv.calculate_centroid(mask)

tracker = mycv.KalmanCentroidTracker()
position = tracker.update(centroid)
```

`mycv.MultiObjectKalmanTracker.update()` returns a mapping of track IDs to `(x, y)` positions. `mycv.reprojection_error()` returns one error per correspondence; take its mean if you need a single summary value.

## Demos

```bash
# Static pipeline (place a photo at examples/test_image.jpg)
python examples/main.py

# Live demo: synthetic input needs no camera
python examples/live_demo.py --source synthetic
python examples/live_demo.py --source synthetic --headless
python examples/live_demo.py --source camera --mode motion
python examples/live_demo.py --source video.mp4
```

The live demo supports colour and motion tracking, template matching, Kalman tracking, object metrics, and optional CNN shape classification. The pretrained shape model ships in `mycv/models/` and is included in package builds.

## Tests

```bash
python -m pytest
```

The core tests need NumPy and pytest only. They do not require a camera or display.

## License

MIT. See [LICENSE](LICENSE).
