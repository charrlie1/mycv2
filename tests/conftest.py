"""Shared pytest fixtures."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    from mycv.filters import sobel_edge_detection
    r=sobel_edge_detection(gray_image)
    return r["Gx"].astype(float), r["Gy"].astype(float)
