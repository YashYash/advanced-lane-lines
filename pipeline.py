"""
Export high-level apis that can be called to execute lane detection methods
"""
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def pipeline(
    image_name: str,
    frame: np.ndarray,
    show_all_views: bool
) -> np.ndarray:
    """Pipeline that handles a video stream"""
    global LANE

    image = Image.from_camera_config(image_name, "cam-1", frame)

    image.undistort()

    warped = image.perspective_transform('birds_eye', False)

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
    detected_lines = LANE.get_lines()
    LANE.get_lane_polygon()

    final_output = LANE.add_polygon_to_image(frame, True)

    if show_all_views is True:
        scale_percent = 20
        width = int(warped.shape[1] * scale_percent / 100)
        height = int(warped.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_warped = cv2.resize(
            warped, dim, interpolation=cv2.INTER_AREA
        )
        resized_detected_lines = cv2.resize(
            detected_lines, dim, interpolation=cv2.INTER_AREA
        )
        resized_detected_lines = cv2.cvtColor(
            resized_detected_lines,
            cv2.COLOR_RGB2BGR
        )

        final_img = Image.from_camera_config(
            "final_ouput",
            "cam-1",
            final_output
        )
        final_output = final_img.overlay_image(
            resized_warped,
            50,
            970,
        )
        final_output = final_img.overlay_image(
            resized_detected_lines,
            200,
            970,
        )
        image_name = image_name.replace(
            'video_binary_output_images',
            'output_images'
        )
        cv2.imwrite(image_name, final_output)

    return final_output
