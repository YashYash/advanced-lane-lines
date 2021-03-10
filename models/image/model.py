"""
An instance of an Image which exposes undistort, perpective transfrom and
get_binary_image methods. self.image is publicly mutable. Therefore the order
in which the class methods are called, matters. To reference the original image
that was used to construct the class, reference self._og_image (private)
"""
import pickle
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from models.thresholds import Thresholds
from constants import _DIST_MTX_COE_PICKLE_FILE_NAME


class Image():
    """Image class implements high-level apis to modify images"""
    _og_image: np.ndarray = np.array([])

    def __init__(
        self,
        name: str,
        image: np.ndarray,
        camera_uuid: str,
        thresholds: "Thresholds"
    ) -> None:
        self.name = name
        self.image = image
        self._og_image = image
        self.camera_uuid = camera_uuid
        self.thresholds = thresholds

    @classmethod
    def from_camera_config(
        cls,
        name: str,
        camera_uuid: str,
        image: np.ndarray
    ) -> "Image":
        """Create an instance of an Image by providing a
           Camera class reference
        """
        return cls(name, image, camera_uuid, Thresholds())

    @staticmethod
    def _get_distortion_config(camera_uuid: str) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        distortion_config = pickle.load(open(
            camera_uuid + '_'+_DIST_MTX_COE_PICKLE_FILE_NAME,
            "rb"
        ))
        matrix = distortion_config["mtx"]
        coefficients = distortion_config["coefficients"]
        return (matrix, coefficients)

    def undistort(self) -> np.ndarray:
        """Undistort self.image by using distortion config stored in
           pickle file for self.camera.uuid
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
        self.image = undistorted_image
        return undistorted_image

    def sobel_binary(self, orient='x', sobel_kernel=3, custom_image=None):
        """Generate binary image by caculting x and y gradient thresholds"""
        thresholds = self.thresholds
        image = self.image
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if custom_image is not None:
            channel = custom_image

        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, sobel_kernel)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, sobel_kernel)

        sobel_x_abs = np.absolute(sobel_x)
        sobel_y_abs = np.absolute(sobel_y)

        scaled_sobel_x = np.uint8(255*sobel_x_abs/np.max(sobel_x_abs))
        scaled_sobel_y = np.uint8(255*sobel_y_abs/np.max(sobel_y_abs))

        binary_output = np.zeros_like(scaled_sobel_x)
        if orient == 'y':
            binary_output = np.zeros_like(scaled_sobel_y)

        binary_output = np.zeros_like(scaled_sobel_x)
        binary_output[
            (scaled_sobel_x >= thresholds.sobel[0]) &
            (scaled_sobel_x <= thresholds.sobel[1])
        ] = 1

        if orient == 'y':
            binary_output = np.zeros_like(scaled_sobel_y)
            binary_output[
                (scaled_sobel_y >= thresholds.sobel[0]) &
                (scaled_sobel_y <= thresholds.sobel[1])
            ] = 1

        return binary_output

    def magnitude_binary(self, sobel_kernel=3) -> np.ndarray:
        """Returns binary output. Apply threshold to magnitude of gradient to remove
           noise from binary output
        """
        threshold = self.thresholds.magnitude
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(magnitude)/255
        magnitude = (magnitude/scale_factor).astype(np.uint8)

        binary_output = np.zeros_like(magnitude)
        binary_output[
            (magnitude >= threshold[0]) &
            (magnitude <= threshold[1])
        ] = 1

        return binary_output

    def direction_binary(self, sobel_kernel=15):
        """Returns binary output. Apply threshold to direction of gradient to filter
          out edges not withing set threshold
        """
        threshold = self.thresholds.direction
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)
        direction = np.arctan2(abs_sobel_y, abs_sobel_x)
        binary_output = np.zeros_like(direction)
        binary_output[(direction >= threshold[0]) &
                      (direction <= threshold[1])] = 1
        return binary_output

    def saturation_binary(self) -> np.ndarray:
        """Returns binary output. Apply threshold to image saturation."""
        thresholds = self.thresholds
        hls = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[
            (s_channel >= thresholds.saturation[0]) &
            (s_channel <= thresholds.saturation[1])
        ] = 1
        return s_binary

    def brightness_binary(self) -> np.ndarray:
        """Returns binary output. Apply threshold to LAB brighness channel"""
        thresholds = self.thresholds
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
        b_channel = lab[:, :, 2]
        b_binary = np.zeros_like(b_channel)
        b_binary[
            (b_channel >= thresholds.brightness[0]) &
            (b_channel <= thresholds.brightness[1])
        ] = 1
        return b_binary

    def lightness_binary(self) -> np.ndarray:
        """Returns binary output. Apply threshold to LAB lightness channel"""
        thresholds = self.thresholds
        hls = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        l_binary = np.zeros_like(l_channel)
        l_binary[
            (l_channel >= thresholds.lightness[0]) &
            (l_channel <= thresholds.lightness[1])
        ] = 1
        return l_binary

    def get_binary_image(
        self,
        sobel_kernel: int,
        dir_kernel: int,
        mag_kernel: int
    ) -> np.ndarray:
        """Returns binary_image genreated by combining saturation, lightness,
           brightness and sobel (abs, magnitude and directional)
        """
        grad_x_output = self.sobel_binary('x', sobel_kernel)
        grad_y_output = self.sobel_binary('y', sobel_kernel)

        mag_output = self.magnitude_binary(mag_kernel)
        dir_output = self.direction_binary(dir_kernel)
        s_binary = self.saturation_binary()
        b_binary = self.brightness_binary()
        l_binary = self.lightness_binary()

        s_channel = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)[:, :, 2]
        sx_binary = self.sobel_binary('x', sobel_kernel, s_channel)

        combined_binary = np.zeros_like(s_binary)
        combined_binary[
            (l_binary == 1) |
            (b_binary == 1) |
            (s_binary == 1) |
            (sx_binary == 1)
        ] = 1

        combined_binary[
            (grad_x_output == 1) &
            (grad_y_output == 1) &
            (mag_output == 21) &
            (dir_output == 1)
        ] = 1

        combined_binary = combined_binary * 255
        combined_binary = combined_binary.astype('uint8')
        self.image = combined_binary
        self._store_binary_image()

        return combined_binary

    def _store_binary_image(self):
        cv2.imwrite(self.name, self.image)

    def perspective_transform(
        self,
        transform_type: str,
        is_dev: bool
    ) -> np.ndarray:
        """Returns bird's eye view of the road ahead"""

        space_around_lane = 258
        car_hood_offset = 35
        height = self.image.shape[0]
        width = self.image.shape[1]
        horizon = 464

        vertices = np.array([[
            (space_around_lane, height - car_hood_offset),
            (575, horizon),
            (width - 575, horizon),
            (width - space_around_lane, height - car_hood_offset)
        ]], dtype=np.int32)

        src = np.float32(np.array([
            vertices[0][0],
            vertices[0][1],
            vertices[0][2],
            vertices[0][3]
        ]))

        dst = np.float32(np.array([
            (space_around_lane, height),
            (space_around_lane, 0),
            (width-space_around_lane, 0),
            (width-space_around_lane, height)
        ]))

        if transform_type == 'birds_eye':
            transform = cv2.getPerspectiveTransform(src, dst)
        else:
            transform = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(self.image, transform, (width, height))
        self.image = warped
        if is_dev is True:
            fname = self.name.replace('binary_outputs', 'birds_eye_view')
            cv2.imwrite(fname, warped)
            self.name.replace('birds_eye_view', 'binary_outputs')

        return warped

    def get_image(self) -> np.ndarray:
        """Get current image"""
        return self.image

    def get_og_image(self) -> np.ndarray:
        """Get og image"""
        return self._og_image

    def get_name(self) -> str:
        """Get image name"""
        return self.name

    def set_image(self, image: np.ndarray):
        """Update image"""
        self.image = image

    def get_histogram(self) -> np.ndarray:
        """Return a histogram data set quantifying the distribution or white pixels
           in the gray scale binary output image
        """
        bottom_half = self.image[self.image.shape[0]//2:, :]
        histogram = np.sum(bottom_half, axis=0)
        return histogram
