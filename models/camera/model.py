"""
Camera class used to calibrate cameras and undistort images
using a calculated distortion matrix and distortion coefficients.
You only create an instance of this class if you need to calibrate a
camera. Once a camera is calibrated, pickle files will be created. These
pickle files will store the 2D ponts, 3D points, distortion matrix and the
distortion coefficients. This data is also accessible directly thorough the
class. Data persists till the instance of the class is destroyed.
"""
import pickle
import copy
from typing import List, Tuple
from constants import _OBJ_IMG_PTS_PICKLE_FILE_NAME
from constants import _DIST_MTX_COE_PICKLE_FILE_NAME
import matplotlib.pyplot as plt

from cv2 import cv2
import numpy as np


class Camera():
    """Prepare object points like (0,0,0), (1,0,0)..."""
    _objp: np.ndarray = np.array([])

    """Arrays to store object points and image points for all
       calibration_images in local memory
    """
    _objpoints: List[np.ndarray] = []  # 3d points in real world
    _imgpoints: List[np.ndarray] = []  # 2d points in image plane

    """Store matrix and coefficients in local memory to prevent
       having to re-calculate these values. Values will also be stored
       as pickle files so other modles can get access to the calibration
       config without having to recontruct this Camera class.
    """
    _distortion_matrix: np.ndarray = np.array([])
    _distortion_coefficients: np.ndarray = np.array([])

    def __init__(
        self,
        uuid: str,
        is_calibrated: bool,
        calibration_images: List[str],
        pattern_size: Tuple[int, int]
    ) -> None:
        self.uuid = uuid
        self.is_calibrated = is_calibrated
        self.calibration_images = calibration_images
        self.pattern_size = pattern_size
        self._objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:self.pattern_size[0],
                                     0: self.pattern_size[1]].T.reshape(-1, 2)

    @classmethod
    def from_calibration_config(
        cls,
        uuid: str,
        calibration_images: List[str],
        pattern_size: Tuple[int, int]
    ) -> "Camera":
        """Construct a Camera class by providing a calibration config"""
        return cls(uuid, False, calibration_images, pattern_size)

    def get_is_calibrated(self) -> bool:
        """For now always return false. In the future add logic here to check
           if pickle files are populated, return true
        """
        return self.is_calibrated

    def _update_img_obj_points(
        self,
        index: int,
        original_image: np.ndarray,
        gray_scale_img: np.ndarray,
        is_dev: bool
    ):
        found_corners, corners = cv2.findChessboardCorners(
            gray_scale_img,
            self.pattern_size,
            None
        )
        calibrated_image = copy.copy(original_image)
        if found_corners is True:
            self._objpoints.append(self._objp)
            self._imgpoints.append(corners)
            cv2.drawChessboardCorners(
                calibrated_image,
                self.pattern_size,
                corners,
                found_corners
            )
            cv2.imwrite('camera_cal_output/calibrated-' +
                        str(index)+'.jpg', calibrated_image)

            if is_dev is True:
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.set_title('Original Image')
                ax1.imshow(original_image)

                ax2.set_title('Calibrated Image')
                ax2.imshow(calibrated_image)

    def _store_obj_img_points(self):
        new_obj = {
            'objpoints': self._objpoints,
            'imgpoints': self._imgpoints
        }
        with open(
            self.uuid + '_' + _OBJ_IMG_PTS_PICKLE_FILE_NAME, 'wb'
        ) as file:
            pickle.dump(new_obj, file)

    def _store_distortion_config(self):
        new_obj = {
            'mtx': self._distortion_matrix,
            'coefficients': self._distortion_coefficients,
            'pattern_size': self.pattern_size
        }
        with open(
            self.uuid+'_'+_DIST_MTX_COE_PICKLE_FILE_NAME, 'wb'
        ) as file:
            pickle.dump(new_obj, file)

    def generate_distortion_config(self, image_shape: Tuple[int, int]):
        """This method should only be called if the distortion matrix and
           coefficients have not been calculated or the calibration images
           have changed
        """
        dist_pickle = pickle.load(
            open(self.uuid + '_'+_OBJ_IMG_PTS_PICKLE_FILE_NAME, "rb")
        )
        objpoints = dist_pickle["objpoints"]
        imgpoints = dist_pickle["imgpoints"]
        _, mtx, dist, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_shape,
            None,
            None
        )
        self._distortion_matrix = mtx
        self._distortion_coefficients = dist
        self._store_distortion_config()

    def calibrate(self, is_dev: bool):
        """Iterate over self.calibration_images and populate
           objpoints and imgpoints
        """
        gray_scale_img: np.ndarray = np.array([])
        for idx, fname in enumerate(self.calibration_images):
            img = cv2.imread(fname)
            gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self._update_img_obj_points(idx, img, gray_scale_img, is_dev)
            self._store_obj_img_points()

        if is_dev is True:
            cv2.destroyAllWindows()

        image_shape: Tuple[int, int] = (
            gray_scale_img.shape[1],
            gray_scale_img.shape[0]
        )
        self.generate_distortion_config(image_shape)
        self.is_calibrated = True
