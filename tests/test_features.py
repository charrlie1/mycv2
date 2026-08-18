"""tests/test_features.py"""
import numpy as np, pytest, sys
sys.path.insert(0,"/home/claude/mycv_github")
from mycv.features import (harris_corner_response,detect_harris_corners,
    hough_line_transform,count_hough_lines,convex_hull,
    extract_object_metrics,draw_bounding_box,classify_object)

class TestHarris:
    def test_shape(self):
        assert harris_corner_response(np.zeros((30,40)),np.zeros((30,40))).shape==(30,40)
    def test_uniform_zero(self):
        np.testing.assert_allclose(harris_corner_response(np.zeros((20,20)),np.zeros((20,20))),0,atol=1e-10)
    def test_corner_positive(self):
        Gx=np.zeros((30,30)); Gy=np.zeros((30,30)); Gx[15,15]=Gy[15,15]=100
        assert harris_corner_response(Gx,Gy)[15,15]>0
    def test_edge_negative(self):
        Gx=np.zeros((30,30)); Gy=np.zeros((30,30)); Gx[15,15]=100
        assert harris_corner_response(Gx,Gy)[15,15]<0
    def test_detect_returns_array(self):
        Gx=np.zeros((40,40)); Gy=np.zeros((40,40))
        assert isinstance(detect_harris_corners(Gx,Gy),np.ndarray)
    def test_uniform_no_corners(self):
        assert len(detect_harris_corners(np.zeros((20,20)),np.zeros((20,20))))==0

class TestHough:
    def test_keys(self):
        assert set(hough_line_transform(np.zeros((20,20),dtype=np.uint8)).keys())=={"accumulator","thetas","rhos","lines"}
    def test_empty_no_votes(self):
        assert hough_line_transform(np.zeros((20,20),dtype=np.uint8))["accumulator"].sum()==0
    def test_line_detected(self):
        e=np.zeros((30,60),dtype=np.uint8); e[15,:]=255
        assert hough_line_transform(e)["accumulator"].max()>0
    def test_count_returns_int(self):
        assert isinstance(count_hough_lines(np.zeros((20,20),dtype=np.uint8)),int)

class TestConvexHull:
    def test_returns_ndarray(self):
        pts=np.random.default_rng(20).random((10,2))
        assert isinstance(convex_hull(pts),np.ndarray)
    def test_at_least_3_points(self):
        pts=np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]],dtype=float)
        assert len(convex_hull(pts))>=3

class TestObjectUtils:
    def test_empty_none(self): assert extract_object_metrics(np.zeros((20,20),dtype=np.uint8)) is None
    def test_area(self):
        m=np.zeros((20,20),dtype=np.uint8); m[5:10,5:10]=255
        assert extract_object_metrics(m)["area"]==25
    def test_classify_string(self):
        img=np.zeros((20,20,3),dtype=np.uint8); img[:,:,0]=255
        assert isinstance(classify_object(img,{"bbox":(0,0,20,20),"aspect_ratio":1.0}),str)
    def test_none_metrics_unknown(self):
        assert classify_object(np.zeros((10,10,3),dtype=np.uint8),None)=="Unknown"
    def test_draw_no_crash(self):
        img=np.zeros((30,30,3),dtype=np.uint8)
        draw_bounding_box(img,(2,2,28,28)); assert True
