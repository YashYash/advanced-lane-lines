"""
Export high-level apis that can be call to execute lane detection methods
"""
import glob
import numpy as np
from models import Lane
from models import Image
from models import Camera

LANE = Lane.from_uuid("cam-1-lane-1")


def calibrate_camera():
    """This method should be called everytime the
      camera_cal directory is updated
    """
    images = glob.glob('camera_cal/calibration*.jpg')
    camera = Camera.from_calibration_config(
        uuid="cam-1",
        calibration_images=images,
        pattern_size=(9, 6)
    )
    camera.calibrate(False)


def pipeline(image_name: str, frame: np.ndarray) -> np.ndarray:
    """Pipeline that handles a video stream"""
    global LANE

    image = Image.from_camera_config(image_name, "cam-1", frame)

    image.undistort()
    image.perspective_transform('birds_eye', False)

    binary_image = image.get_binary_image(
        sobel_kernel=15,
        dir_kernel=15,
        mag_kernel=9
    )

    img_name = image_name.replace(
      'video_binary_output_images',
      'output_fit_images'
    )
    image = Image.from_camera_config(
      img_name,
      "cam-1",
      binary_image
    )

    LANE.update_binary_output(image)
    LANE.get_lines()
    LANE.get_lane_polygon()
    final_output = LANE.add_polygon_to_image(frame, True)

    return final_output
