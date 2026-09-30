"""tests/test_geometry.py"""
import numpy as np, pytest
from mycv.geometry import bilinear_interpolate,rotate_image,warp_perspective

class TestBilinear:
    def test_integer_exact(self):
        img=np.arange(25,dtype=float).reshape(5,5)
        assert bilinear_interpolate(img,np.array([[2.0]]),np.array([[3.0]]))[0,0]==pytest.approx(float(img[3,2]))
    def test_midpoint_avg(self):
        img=np.array([[0,100],[0,100]],dtype=float)
        assert bilinear_interpolate(img,np.array([[0.5]]),np.array([[0.0]]))[0,0]==pytest.approx(50.0)
    def test_shape(self):
        assert bilinear_interpolate(np.ones((20,30)),np.zeros((10,15)),np.zeros((10,15))).shape==(10,15)
    def test_oob_clamped(self):
        img=np.full((10,10),77.0)
        assert bilinear_interpolate(img,np.array([[-5.0]]),np.array([[0.0]]))[0,0]==pytest.approx(77.0)

class TestRotate:
    def test_shape(self): assert rotate_image(np.zeros((50,60),dtype=np.uint8),45).shape==(50,60)
    def test_dtype(self): assert rotate_image(np.zeros((30,30),dtype=np.uint8),30).dtype==np.uint8
    def test_zero_identity(self):
        img=np.random.default_rng(13).integers(0,256,(30,30),dtype=np.uint8)
        np.testing.assert_array_equal(rotate_image(img,0.0),img)
    def test_range(self):
        r=rotate_image(np.random.default_rng(14).integers(0,256,(30,30),dtype=np.uint8),37)
        assert 0<=r.min() and r.max()<=255

class TestWarp:
    def test_shape(self): assert warp_perspective(np.zeros((50,60),dtype=np.uint8),np.eye(3),(40,50)).shape==(40,50)
    def test_identity(self):
        img=np.random.default_rng(15).integers(0,256,(30,30),dtype=np.uint8)
        np.testing.assert_array_equal(warp_perspective(img,np.eye(3),(30,30)),img)
    def test_bad_H(self):
        with pytest.raises(ValueError): warp_perspective(np.zeros((10,10),dtype=np.uint8),np.eye(4),(10,10))
