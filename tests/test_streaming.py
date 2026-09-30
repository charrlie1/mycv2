"""tests/test_streaming.py"""
import numpy as np, pytest

def test_importable(): import mycv.streaming

def test_has_av_flag():
    from mycv.streaming import _HAS_AV
    assert isinstance(_HAS_AV, bool)

def test_streamreader_exists():
    from mycv.streaming import StreamReader
    assert StreamReader is not None

try:
    import av; AV_OK=True
except ImportError:
    AV_OK=False

@pytest.mark.skipif(AV_OK, reason="av installed")
def test_no_av_raises():
    from mycv.streaming import StreamReader
    with pytest.raises((ImportError,RuntimeError)):
        StreamReader("rtsp://dummy")
