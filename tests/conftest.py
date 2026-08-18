"""Shared pytest fixtures."""
import numpy as np, pytest

@pytest.fixture
def rng(): return np.random.default_rng(42)

@pytest.fixture
def gray_image(rng): return rng.integers(0,256,(128,128),dtype=np.uint8)

@pytest.fixture
def rgb_image(rng): return rng.integers(0,256,(128,128,3),dtype=np.uint8)

@pytest.fixture
def binary_mask():
    m=np.zeros((128,128),dtype=np.uint8); m[48:80,48:80]=255; return m

@pytest.fixture
def sobel_outputs(gray_image):
    import sys; sys.path.insert(0,"/home/claude/mycv_github")
    from mycv.filters import sobel_edge_detection
    r=sobel_edge_detection(gray_image)
    return r["Gx"].astype(float), r["Gy"].astype(float)
