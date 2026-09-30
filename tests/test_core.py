"""tests/test_core.py"""
import numpy as np, pytest
from mycv.core import rgb_to_grayscale, threshold

class TestGray:
    def test_shape(self): assert rgb_to_grayscale(np.zeros((10,10,3),dtype=np.uint8)).shape==(10,10)
    def test_dtype(self): assert rgb_to_grayscale(np.zeros((5,5,3),dtype=np.uint8)).dtype==np.uint8
    def test_black(self): assert np.all(rgb_to_grayscale(np.zeros((5,5,3),dtype=np.uint8))==0)
    def test_white(self): assert np.all(rgb_to_grayscale(np.full((5,5,3),255,dtype=np.uint8))==255)
    def test_red(self):
        i=np.zeros((1,1,3),dtype=np.uint8); i[0,0]=[255,0,0]
        assert int(rgb_to_grayscale(i)[0,0])==round(0.2989*255)
    def test_range(self):
        r=rgb_to_grayscale(np.random.default_rng(0).integers(0,256,(50,50,3),dtype=np.uint8))
        assert 0<=r.min() and r.max()<=255
    def test_bad_shape(self):
        with pytest.raises(ValueError): rgb_to_grayscale(np.zeros((5,5),dtype=np.uint8))

class TestThresh:
    def test_binary(self):
        assert set(np.unique(threshold(np.random.default_rng(1).integers(0,256,(20,20),dtype=np.uint8)))).issubset({0,255})
    def test_known(self):
        assert np.array_equal(threshold(np.array([[100,200],[50,150]],dtype=np.uint8),127),np.array([[0,255],[0,255]],dtype=np.uint8))
    def test_bad_tau(self):
        with pytest.raises(ValueError): threshold(np.zeros((5,5),dtype=np.uint8),tau=300)
