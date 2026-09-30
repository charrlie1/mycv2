"""tests/test_color.py"""
import numpy as np, pytest
from mycv.color import histogram_equalize, rgb_to_hsv

class TestHEq:
    def test_shape_dtype(self):
        r=histogram_equalize(np.zeros((50,60),dtype=np.uint8))
        assert r.shape==(50,60) and r.dtype==np.uint8
    def test_range(self):
        r=histogram_equalize(np.random.default_rng(2).integers(0,256,(50,50),dtype=np.uint8))
        assert 0<=r.min() and r.max()<=255
    def test_bad_ndim(self):
        with pytest.raises(ValueError): histogram_equalize(np.zeros((5,5,3),dtype=np.uint8))

class TestHSV:
    def test_shape_dtype(self):
        r=rgb_to_hsv(np.zeros((10,10,3),dtype=np.uint8))
        assert r.shape==(10,10,3) and r.dtype==np.float32
    def test_red_hue_zero(self):
        i=np.zeros((1,1,3),dtype=np.uint8); i[0,0]=[255,0,0]
        hsv=rgb_to_hsv(i)
        assert float(hsv[0,0,0])==pytest.approx(0.0) and float(hsv[0,0,1])==pytest.approx(1.0)
    def test_green_hue_120(self):
        i=np.zeros((1,1,3),dtype=np.uint8); i[0,0]=[0,255,0]
        assert float(rgb_to_hsv(i)[0,0,0])==pytest.approx(120.0)
    def test_blue_hue_240(self):
        i=np.zeros((1,1,3),dtype=np.uint8); i[0,0]=[0,0,255]
        assert float(rgb_to_hsv(i)[0,0,0])==pytest.approx(240.0)
    def test_sv_range(self):
        h=rgb_to_hsv(np.random.default_rng(3).integers(0,256,(20,20,3),dtype=np.uint8))
        assert h[...,1].min()>=0 and h[...,1].max()<=1
    def test_bad_shape(self):
        with pytest.raises(ValueError): rgb_to_hsv(np.zeros((5,5),dtype=np.uint8))
