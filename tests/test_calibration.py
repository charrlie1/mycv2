"""tests/test_calibration.py"""
import numpy as np, pytest
from mycv.calibration import normalize_points,solve_homography_dlt,reprojection_error,ransac_homography

def _pairs(n=20,noise=0.,seed=0):
    rng=np.random.default_rng(seed)
    H=np.array([[1.2,.3,15],[.1,1.,10],[.001,.002,1]],dtype=float)
    src=rng.uniform(10,200,(n,2)); sh=np.column_stack([src,np.ones(n)])
    dh=(H@sh.T).T; dst=dh[:,:2]/dh[:,2:3]
    if noise: dst+=rng.normal(0,noise,dst.shape)
    return src,dst,H

class TestNorm:
    def test_shapes(self):
        pts=np.random.default_rng(0).uniform(0,100,(20,2))
        pn,T=normalize_points(pts); assert pn.shape==(20,2) and T.shape==(3,3)
    def test_centroid_origin(self):
        pts=np.random.default_rng(1).uniform(0,200,(30,2))
        pn,_=normalize_points(pts)
        assert pn.mean(axis=0)==pytest.approx([0.,0.],abs=1e-8)

class TestDLT:
    def test_shape(self):
        src,dst,_=_pairs(8); assert solve_homography_dlt(src,dst).shape==(3,3)
    def test_last_elem_one(self):
        src,dst,_=_pairs(8); H=solve_homography_dlt(src,dst)
        assert H[2,2]==pytest.approx(1.,abs=1e-6)
    def test_exact_mapping(self):
        src,dst,_=_pairs(10,noise=0.)
        H=solve_homography_dlt(src,dst)
        sh=np.column_stack([src,np.ones(10)]); dh=(H@sh.T).T
        np.testing.assert_allclose(dh[:,:2]/dh[:,2:3],dst,atol=1e-4)
    def test_raises_too_few(self):
        s=np.random.default_rng(4).random((3,2)); d=np.random.default_rng(5).random((3,2))
        with pytest.raises((ValueError,np.linalg.LinAlgError)): solve_homography_dlt(s,d)

class TestReproj:
    def test_identity_zero(self):
        pts=np.random.default_rng(8).uniform(0,100,(10,2))
        errors = reprojection_error(np.eye(3),pts,pts)
        assert errors.shape == (10,)
        np.testing.assert_allclose(errors, 0., atol=1e-8)
    def test_good_lt_bad(self):
        src,dst,_=_pairs(20)
        good = reprojection_error(solve_homography_dlt(src,dst),src,dst)
        bad = reprojection_error(np.eye(3),src,dst)
        assert good.mean() < bad.mean()

class TestRANSAC:
    def test_shapes(self):
        src,dst,_=_pairs(30); H,mask=ransac_homography(src,dst)
        assert H.shape==(3,3) and mask.shape==(30,) and mask.dtype==np.bool_
    def test_finds_inliers(self):
        src,dst,_=_pairs(30,noise=.5)
        rng=np.random.default_rng(1)
        src2=np.vstack([src,rng.uniform(0,200,(10,2))]); dst2=np.vstack([dst,rng.uniform(0,200,(10,2))])
        _,mask=ransac_homography(src2,dst2,inlier_threshold=3.)
        assert mask.sum()>=15
    def test_raises_too_few(self):
        with pytest.raises(ValueError): ransac_homography(np.random.random((3,2)),np.random.random((3,2)))
