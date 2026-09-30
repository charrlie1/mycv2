"""tests/test_filters.py"""
import numpy as np, pytest
from mycv.filters import convolve2d, sobel_edge_detection, SOBEL_X, SOBEL_Y

class TestConv:
    def test_same_shape(self): assert convolve2d(np.ones((50,60)),np.ones((3,3)),"same").shape==(50,60)
    def test_valid_shape(self): assert convolve2d(np.ones((50,60)),np.ones((3,3)),"valid").shape==(48,58)
    def test_identity(self):
        img=np.random.default_rng(5).integers(0,256,(30,30)).astype(float)
        k=np.array([[0,0,0],[0,1,0],[0,0,0]],dtype=float)
        np.testing.assert_allclose(convolve2d(img,k,"same"),img,atol=1e-10)
    def test_true_conv_not_corr(self):
        img=np.zeros((7,7)); img[3,3]=1.0
        k=np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=float)
        assert float(convolve2d(img,k,"same")[3,3])==pytest.approx(5.0)
    def test_bad_input(self):
        with pytest.raises(ValueError): convolve2d(np.ones((5,5,3)),np.ones((3,3)))

class TestSobel:
    def test_keys(self): assert set(sobel_edge_detection(np.zeros((20,20),dtype=np.uint8)).keys())=={"Gx","Gy","magnitude","direction"}
    def test_zero_interior(self):
        r=sobel_edge_detection(np.full((20,20),128,dtype=np.uint8))
        assert r["magnitude"][5,5]==0
    def test_vertical_edge(self):
        img=np.zeros((20,20),dtype=np.uint8); img[:,10:]=255
        assert float(np.abs(sobel_edge_detection(img)["Gx"][:,9:12]).max())>0
    def test_kernels(self):
        assert SOBEL_X.shape==(3,3)
        np.testing.assert_array_equal(SOBEL_Y,SOBEL_X.T)
