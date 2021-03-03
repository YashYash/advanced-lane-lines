import pickle
from typing import List, Tuple

import cv2
import numpy as np


class Camera():
    """Is the camera calibrated"""
    is_calibrated: bool

    """Name of the pickle file that will cache the current calibration object
       and image points.
    """
    _OBJ_IMG_PTS_PICKLE_FILE_NAME: str = 'obj_img_points.p'

    """Prepare object points like (0,0,0), (1,0,0)...
    """
    _objp: np.ndarray = []

    """Arrays to store object points and image points for all
       calibration_images in local memory
    """
    _objpoints: List[float] = []  # 3d points in real world
    _imgpoints: List[float] = []  # 2d points in image plane

    """Name of the pickle file that will cache the current distortion
       matrix and coefficients. Since every camera will have a unique uuid 
       (for now I am just using the uuid field), the pickle file uuid will be prefixed with the uuid (uuid for now). When an Image class calls image.undistort, the camera uuid will be passed to let the image know which distortion config needs to be used to undistort the image.
    """
    _DIST_MTX_COE_PICKLE_FILE_NAME = 'dist_mtx_coe.p'

    """Store matrix and coefficients in local memory as well to prevent
       having to re-calculate these values
    """
    _distortion_matrix: np.ndarray = []
    _distortion_coefficients: np.ndarray = []

    def __init__(
        self,
        uuid: str,
        name: str,
        calibration_images: List[str],
        pattern_size: Tuple[int],
    ) -> None:
        self.uuid = uuid
        self.name = name
        self.calibration_images = calibration_images
        self.pattern_size = pattern_size
        self._objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:self.pattern_size[0],
                                     0: self.pattern_size[1]].T.reshape(-1, 2)

    @ classmethod
    def from_config(
        cls,
        uuid: str,
        name: str,
        calibration_images: List[str],
        pattern_size: Tuple[int]
    ) -> "Camera":
        return cls(uuid, name, calibration_images, pattern_size)

    def is_calibrated(self) -> bool:
        """For now always return false. In the future add logic here to check
           if pickle files are populated return true
        """
        return self._is_calibrated

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
        if found_corners == True:
            self._objpoints.append(self._objp)
            self._imgpoints.append(corners)
            cv2.drawChessboardCorners(
                original_image,
                self.pattern_size,
                corners,
                found_corners
            )
            cv2.imwrite('camera_cal_output/calibrated-' +
                        str(index)+'.jpg', original_image)
            if is_dev == True:
                cv2.imshow('updated image', original_image)
                cv2.waitKey(500)

    def _store_obj_img_points(self):
        new_obj = {
            'objpoints': self._objpoints,
            'imgpoints': self._imgpoints
        }
        with open(
            self.uuid + '_' + self._OBJ_IMG_PTS_PICKLE_FILE_NAME, 'wb'
        ) as f:
            pickle.dump(new_obj, f)

    def _store_distortion_config(self):
        new_obj = {
            'mtx': self._distortion_matrix,
            'coefficients': self._distortion_coefficients
        }
        with open(
            self.uuid+'_'+self._DIST_MTX_COE_PICKLE_FILE_NAME, 'wb'
        ) as f:
            pickle.dump(new_obj, f)

    def generate_distortion_config(self, image_shape: Tuple[int, int]):
        """This method should only be called if the distortion matrix and
           coefficients have not been calculated or the calibration images
           have changed
        """
        dist_pickle = pickle.load(
            open(self.uuid + '_'+self._OBJ_IMG_PTS_PICKLE_FILE_NAME, "rb")
        )
        objpoints = dist_pickle["objpoints"]
        imgpoints = dist_pickle["imgpoints"]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
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
        gray_scale_img: np.ndarray = None
        for idx, fname in enumerate(self.calibration_images):
            img = cv2.imread(fname)
            gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self._update_img_obj_points(idx, img, gray_scale_img, is_dev)
            self._store_obj_img_points()

        if is_dev == True:
            cv2.destroyAllWindows()

        self.generate_distortion_config(gray_scale_img.shape[::-1])
        self._is_calibrated = True
