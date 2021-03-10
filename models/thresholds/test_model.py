from models import Thresholds


def test_construct_thresholds():
    """Test Thresholds constructor"""
    thresholds = Thresholds()
    assert thresholds.saturation == (100, 255)
    assert thresholds.lightness == (200, 255)
    assert thresholds.brightness == (220, 255)
    assert thresholds.sobel == (30, 100)
    assert thresholds.magnitude == (30, 100)
    assert thresholds.direction == (0.6, 1.3)
