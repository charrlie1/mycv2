"""tests/test_detection.py"""
import numpy as np, pytest
from mycv.detection import match_template_ncc,find_template_matches,gaussian_pyramid,non_max_suppression

class TestNCC:
    def test_shape(self): assert match_template_ncc(np.zeros((50,60),dtype=np.uint8),np.zeros((10,12),dtype=np.uint8)).shape==(41,49)
    def test_perfect_match(self):
        img=np.arange(400,dtype=float).reshape(20,20)
        assert match_template_ncc(img,img[5:10,5:10].copy()).max()==pytest.approx(1.0,abs=1e-6)
    def test_range(self):
        rng=np.random.default_rng(12)
        n=match_template_ncc(rng.integers(0,256,(40,40),dtype=np.uint8),rng.integers(0,256,(8,8),dtype=np.uint8))
        assert n.min()>=-1.0 and n.max()<=1.0
    def test_flat_zero(self):
        assert np.all(match_template_ncc(np.full((20,20),128,dtype=np.uint8),np.full((5,5),128,dtype=np.uint8))==0)
    def test_bad_input(self):
        with pytest.raises(ValueError): match_template_ncc(np.zeros((10,10,3)),np.zeros((3,3)))

class TestPyramid:
    def test_length(self): assert len(gaussian_pyramid(np.zeros((64,64),dtype=np.uint8),levels=4))==4
    def test_half_res(self): assert gaussian_pyramid(np.zeros((64,64),dtype=np.uint8),levels=2)[1].shape==(32,32)
    def test_shrinks(self):
        p=gaussian_pyramid(np.zeros((64,64),dtype=np.uint8),levels=4)
        for k in range(1,len(p)): assert p[k].shape[0]<=p[k-1].shape[0]

class TestNMS:
    def test_single(self):
        kb,_=non_max_suppression(np.array([[0,0,10,10]],dtype=float),np.array([0.9])); assert len(kb)==1
    def test_identical_keep_best(self):
        _,ks=non_max_suppression(np.tile([0,0,10,10],(5,1)).astype(float),np.array([.5,.9,.7,.3,.8]))
        assert len(ks)==1 and ks[0]==pytest.approx(0.9)
    def test_disjoint_keep_all(self):
        boxes=np.array([[0,0,10,10],[20,20,30,30],[40,40,50,50]],dtype=float)
        kb,_=non_max_suppression(boxes,np.array([.9,.8,.7])); assert len(kb)==3
    def test_empty(self):
        kb,_=non_max_suppression(np.empty((0,4)),np.empty(0)); assert len(kb)==0
