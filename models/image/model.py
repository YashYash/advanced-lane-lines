"""
An instance of an Image which exposes undistort, perpective transfrom and
get_binary_image methods
"""
import pickle
from typing import Tuple
import cv2
import numpy as np

from constants import _DIST_MTX_COE_PICKLE_FILE_NAME

class Image():
    """Image class is a higher order class to implement high-level image modification
       apis
    """
    def __init__(
        self,
        image: np.ndarray,
        camera_uuid: str
    ) -> None:
        self.image = image
        self.camera_uuid = camera_uuid

    @classmethod
    def from_camera_config(
        cls,
        camera_uuid: str,
        image: np.ndarray
    ) -> "Image":
        """Create an instance of an Image by providing a Camera class reference"""
        return cls(image, camera_uuid)

    @staticmethod
    def _get_distortion_config(camera_uuid: str) -> Tuple[np.ndarray, np.ndarray]:
        distortion_config = pickle.load(open(
            camera_uuid + '_'+_DIST_MTX_COE_PICKLE_FILE_NAME,
            "rb"
        ))
        matrix = distortion_config["mtx"]
        coefficients = distortion_config["coefficients"]
        return (matrix, coefficients)

    def undistort(self) -> np.ndarray:
        """Undistort self.image by using distortion config stored in pickle file
           for self.camera.uuid
        """
        distortion_config = self._get_distortion_config(self.camera_uuid)
        matrix = distortion_config[0]
        coefficients = distortion_config[1]
        undistorted_image = cv2.undistort(
            self.image,
            matrix,
            coefficients,
            None,
            matrix
        )
        return undistorted_image
