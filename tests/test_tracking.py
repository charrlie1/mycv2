"""tests/test_tracking.py"""
import numpy as np, pytest
from mycv.tracking import (compute_motion_mask,color_mask,color_mask_hue_wrap,
    calculate_centroid,TemporalSmoother,
    kalman_filter_predict,kalman_filter_update,mahalanobis_gate,
    KalmanCentroidTracker,MultiObjectKalmanTracker)

class TestMotion:
    def test_shape_dtype(self):
        a=np.zeros((50,60),dtype=np.uint8)
        r=compute_motion_mask(a,a); assert r.shape==(50,60) and r.dtype==np.uint8
    def test_identical_no_motion(self):
        a=np.full((10,10),128,dtype=np.uint8); assert np.all(compute_motion_mask(a,a,1)==0)
    def test_uint8_underflow_fixed(self):
        a=np.array([[10]],dtype=np.uint8); b=np.array([[200]],dtype=np.uint8)
        assert compute_motion_mask(a,b,threshold=100)[0,0]==255
    def test_binary(self):
        rng=np.random.default_rng(10)
        a=rng.integers(0,256,(20,20),dtype=np.uint8); b=rng.integers(0,256,(20,20),dtype=np.uint8)
        assert set(np.unique(compute_motion_mask(a,b))).issubset({0,255})
    def test_mismatch(self):
        with pytest.raises(ValueError): compute_motion_mask(np.zeros((10,10),dtype=np.uint8),np.zeros((10,11),dtype=np.uint8))

class TestColorMask:
    def _hsv(self,H,S,V):
        img=np.zeros((1,1,3),dtype=np.float32); img[0,0]=[H,S,V]; return img
    def test_inside(self): assert color_mask(self._hsv(60,.5,.5),np.array([0,0,0]),np.array([120,1,1]))[0,0]==255
    def test_outside(self): assert color_mask(self._hsv(200,.5,.5),np.array([0,0,0]),np.array([120,1,1]))[0,0]==0
    def test_hue_wrap_red(self):
        hsv=self._hsv(355,1,1); assert color_mask_hue_wrap(hsv,[350,.5,.5],[10,1,1])[0,0]==255
    def test_hue_wrap_excludes(self):
        hsv=self._hsv(120,1,1); assert color_mask_hue_wrap(hsv,[350,.5,.5],[10,1,1])[0,0]==0

class TestCentroid:
    def test_empty_sentinel(self): assert calculate_centroid(np.zeros((20,20),dtype=np.uint8))==(-1,-1)
    def test_single_pixel(self):
        m=np.zeros((20,20),dtype=np.uint8); m[10,15]=255
        cx,cy=calculate_centroid(m); assert cx==pytest.approx(15.0) and cy==pytest.approx(10.0)

class TestSmoother:
    def test_shape_dtype(self):
        s=TemporalSmoother(3,10,10,3); f=np.zeros((10,10,3),dtype=np.uint8)
        assert s.update(f).dtype==np.uint8
    def test_filled_after_n(self):
        s=TemporalSmoother(3,5,5,3); [s.update(np.zeros((5,5,3),dtype=np.uint8)) for _ in range(3)]
        assert s.is_filled
    def test_no_overflow(self):
        s=TemporalSmoother(5,5,5,1); f=np.full((5,5),200,dtype=np.uint8)
        for _ in range(5): r=s.update(f)
        assert r.max()==200
    def test_reset(self):
        s=TemporalSmoother(3,5,5,3); [s.update(np.full((5,5,3),200,dtype=np.uint8)) for _ in range(3)]
        s.reset(); assert not s.is_filled

class TestKalman:
    def _cv(self):
        F=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
        H=np.array([[1,0,0,0],[0,1,0,0]],dtype=float)
        return F,H,np.eye(4)*1e-3,np.eye(2)*1e-1,np.eye(4)
    def test_predict_shape(self):
        F,_,Q,_,P=self._cv(); sp,Pp=kalman_filter_predict(np.zeros(4),P,F,Q)
        assert sp.shape==(4,) and Pp.shape==(4,4)
    def test_update_shape(self):
        F,H,Q,R,P=self._cv(); sp,Pp=kalman_filter_predict(np.zeros(4),P,F,Q)
        su,Pu=kalman_filter_update(sp,Pp,np.array([1.,2.]),H,R)
        assert su.shape==(4,) and Pu.shape==(4,4)
    def test_gate_close(self):
        F,H,Q,R,P=self._cv(); sp,Pp=kalman_filter_predict(np.zeros(4),P,F,Q)
        assert mahalanobis_gate(sp,Pp,np.array([0.1,0.1]),H,R) <= 10
    def test_gate_far(self):
        F,H,Q,R,P=self._cv(); sp,Pp=kalman_filter_predict(np.zeros(4),P,F,Q)
        assert mahalanobis_gate(sp,Pp,np.array([1000.,1000.]),H,R) > 1
    def test_tracker_first_update(self): assert KalmanCentroidTracker().update((5.,10.))==(5.,10.)
    def test_tracker_reset(self):
        t=KalmanCentroidTracker(); t.update((5.,5.)); t.reset()
        assert t.update((99.,99.))==(99.,99.)
    def test_multi_assigns_ids(self):
        t=MultiObjectKalmanTracker(); r=t.update([(10.,20.),(50.,60.)])
        assert len(r)==2 and len(set(r))==2
        assert set(r.values()) == {(10.,20.), (50.,60.)}
