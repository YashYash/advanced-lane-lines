"""
Test camera model
"""
import glob
import os
import cv2
import pytest
from models.camera import Camera


@pytest.fixture()
def constructor_arguments():
    """Reusable camera arguments"""
    path = os.path.dirname(os.path.abspath(__file__))
    pathname = path.replace(
        '/models/camera',
        '/camera_cal/calibration*.jpg'
    )
    images = glob.glob(pathname)
    args = {
        "uuid": "cam-1",
        "calibration_images": images,
        "pattern_size": (9, 6)
    }
    yield args


@pytest.fixture()
def calibration_camera(**kwargs):
    """Dynamic creationg of Camera models"""
    def _calibration_camera(**kwargs):
        return Camera.from_calibration_config(**kwargs)

    return _calibration_camera


def test_construct_calibration_cam(calibration_camera):
    """Test construction various types of cameras"""
    test_cases = [{
        "uuid": "cam-1",
        "calibration_images": glob.glob("camera_cal/calibration*.jpg"),
        "pattern_size": (9, 6)
    }, {
        "uuid": "cam-2",
        "calibration_images": [],
        "pattern_size": (10, 6)
    }]

    for config in test_cases:
        def run_tests():
            cam = calibration_camera(
                uuid=config["uuid"],
                calibration_images=config["calibration_images"],
                pattern_size=config["pattern_size"]
            )
            assert cam.uuid == config["uuid"]
            assert cam.pattern_size == config["pattern_size"]

        if len(config["calibration_images"]) > 0:
            run_tests()
        else:
            with pytest.raises(ValueError) as val_error:
                run_tests()
            assert "No calibration images passed" in str(val_error)


def test_calibrate(calibration_camera, constructor_arguments):
    """Test camera calibration"""
    config = constructor_arguments
    cam = calibration_camera(
        uuid=config["uuid"],
        calibration_images=config["calibration_images"],
        pattern_size=config["pattern_size"]
    )

    cam.calibrate(False)
    assert cam.get_is_calibrated() is True

    for _, fname in enumerate(config["calibration_images"]):
        calibrated_image_fname = fname.replace(
            'camera_cal',
            'camera_cal_output'
        ).replace('calibration', 'calibrated-')
        calibrated_image = cv2.imread(calibrated_image_fname)
        print(calibrated_image_fname)
        assert calibrated_image is not None
