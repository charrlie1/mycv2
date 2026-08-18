"""tests/test_morphology.py"""
import numpy as np, pytest, sys
sys.path.insert(0,"/home/claude/mycv_github")
from mycv.morphology import dilate,erode,opening,closing,grayscale_dilate,label_connected_components,component_properties

SE3=np.ones((3,3),dtype=np.bool_)
def _px(H=15,W=15,y=7,x=7): m=np.zeros((H,W),dtype=np.uint8); m[y,x]=255; return m
def _full(H=10,W=10): return np.full((H,W),255,dtype=np.uint8)
def _empty(H=10,W=10): return np.zeros((H,W),dtype=np.uint8)

class TestBinaryMorph:
    def test_dilate_expands(self):
        r=dilate(_px(15,15,7,7),SE3); assert np.all(r[6:9,6:9]==255)
    def test_erode_removes_pixel(self): assert np.all(erode(_px(),SE3)==0)
    def test_dilate_full_unchanged(self): np.testing.assert_array_equal(dilate(_full(),SE3),_full())
    def test_erode_full_unchanged(self): np.testing.assert_array_equal(erode(_full(),SE3),_full())
    def test_opening_removes_noise(self): assert np.all(opening(_px(),SE3)==0)
    def test_closing_fills_hole(self):
        img=_full(); img[5,5]=0; assert closing(img,SE3)[5,5]==255
    def test_opening_subset(self):
        img=(np.random.default_rng(8).integers(0,2,(20,20))*255).astype(np.uint8)
        assert np.all((opening(img,SE3)==255)<=(img==255))
    def test_binary_output(self):
        assert set(np.unique(dilate(_px()))).issubset({0,255})
        assert set(np.unique(erode(_full()))).issubset({0,255})

class TestGrayscaleDilate:
    def test_shape_dtype(self):
        r=grayscale_dilate(np.zeros((20,20),dtype=np.uint8),size=3)
        assert r.shape==(20,20) and r.dtype==np.uint8
    def test_peak_spreads(self):
        img=np.zeros((10,10),dtype=np.uint8); img[5,5]=200
        r=grayscale_dilate(img,size=3)
        assert r[4,4]==200 and r[6,6]==200
    def test_uniform_unchanged(self):
        img=np.full((10,10),128,dtype=np.uint8)
        np.testing.assert_array_equal(grayscale_dilate(img,size=3),img)

class TestConnectedComponents:
    def test_two_blobs(self):
        m=np.zeros((10,20),dtype=np.uint8); m[2:5,2:5]=255; m[2:5,15:18]=255
        _,n=label_connected_components(m); assert n==2
    def test_empty_zero_labels(self):
        _,n=label_connected_components(np.zeros((10,10),dtype=np.uint8)); assert n==0
    def test_single_blob(self):
        m=np.zeros((10,10),dtype=np.uint8); m[3:7,3:7]=255
        _,n=label_connected_components(m); assert n==1
    def test_background_is_zero(self):
        m=np.zeros((10,10),dtype=np.uint8); m[3:7,3:7]=255
        labels,_=label_connected_components(m); assert labels[0,0]==0
    def test_properties_area(self):
        m=np.zeros((20,20),dtype=np.uint8); m[5:10,5:10]=255
        labels,n=label_connected_components(m)
        props=component_properties(labels,n); assert props[0]["area"]==25
    def test_properties_centroid(self):
        m=np.zeros((20,20),dtype=np.uint8); m[8:12,8:12]=255
        labels,n=label_connected_components(m)
        cx,cy=component_properties(labels,n)[0]["centroid"]
        assert cx==pytest.approx(9.5,abs=0.6) and cy==pytest.approx(9.5,abs=0.6)
