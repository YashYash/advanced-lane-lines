"""
An instance of an Image which exposes undistort, perpective transfrom and
get_binary_image methods. self.image is publicly mutable. Therefore the order
in which the class methods are called, matters. To reference the original image
that was used to construct the class, reference self._og_image (private)
"""
import pickle
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
            (scaled_sobel_x >= thresholds.sobel_x[0]) &
            (scaled_sobel_x <= thresholds.sobel_x[1])
        ] = 1

        if orient == 'y':
            binary_output = np.zeros_like(scaled_sobel_y)
            binary_output[
                (scaled_sobel_y >= thresholds.sobel_y[0]) &
                (scaled_sobel_y <= thresholds.sobel_y[1])
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
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        l_binary = np.zeros_like(l_channel)
        l_binary[
            (l_channel >= thresholds.lightness[0]) &
            (l_channel <= thresholds.lightness[1])
        ] = 1
        return l_binary

    def lab_binary(self) -> np.ndarray:
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        h_binary = np.zeros_like(h_channel)
        l_binary = np.zeros_like(l_channel)
        s_binary = np.zeros_like(s_channel)
        h_binary[
            (h_channel >= 20) &
            (h_channel <= 225)
        ] = 1
        l_binary[
            (l_channel >= 110) &
            (l_channel <= 140)
        ] = 1
        s_binary[
            (s_channel >= 170) &
            (s_channel <= 215)
        ] = 1

        return h_binary & l_binary & s_binary

    def hls_shadowed_yellow(self) -> np.ndarray:
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        h_binary = np.zeros_like(h_channel)
        l_binary = np.zeros_like(l_channel)
        s_binary = np.zeros_like(s_channel)
        h_binary[
            (h_channel >= 18) &
            (h_channel <= 63)
        ] = 1
        l_binary[
            (l_channel >= 55) &
            (l_channel <= 255)
        ] = 1
        s_binary[
            (s_channel >= 0) &
            (s_channel <= 140)
        ] = 1

        return h_binary & l_binary & s_binary

    def rgb_white(self):
        image = self.image
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        binary_output = np.zeros_like(r)
        binary_output[(r >= 210) & (g >= 200) & (b >= 190)] = 1
        return binary_output

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
        s_binary = self.saturation_binary()
        white = self.rgb_white()
        yellow = self.hls_shadowed_yellow()
        lab_binary = self.lab_binary()
        b_binary = self.brightness_binary()
        l_binary = self.lightness_binary()

        combined_binary = (
            grad_x_output |
            grad_y_output |
            s_binary |
            lab_binary |
            white |
            yellow |
            b_binary |
            l_binary
        )

        combined_binary = combined_binary * 255
        combined_binary = combined_binary.astype('uint8')
        self.image = combined_binary
        self._store_binary_image()

        return combined_binary

    def _store_binary_image(self):
        cv2.imwrite(self.name, self.image)

    @staticmethod
    def region_of_interest(image, vertices):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, [255, 0, 0])
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def hough_lines(self, image: np.ndarray):
        rho = 6
        theta = (np.pi/60)
        threshold = 200
        min_line_len = 20
        max_line_gap = 25
        gray = cv2.cvtColor(self._og_image, cv2.COLOR_RGB2GRAY)
        plt.figure()
        plt.imshow(gray)
        region_of_interest_vertices = np.array([[(200, image.shape[0]), (600, 360), (
            600, 345), (1100, image.shape[0])]], dtype=np.int32)
        cv2.polylines(gray, region_of_interest_vertices,
                      False, [255, 0, 0], 14, None, None)
        cropped_image = self.region_of_interest(
            gray,
            np.array(
                [region_of_interest_vertices],
                np.int32
            )
        )
        plt.imshow(cropped_image)
        canny = cv2.Canny(cropped_image, 100, 200)

        lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array(
            []), minLineLength=min_line_len, maxLineGap=max_line_gap)
        # If there are no lines to draw, exit.
        if lines is None:
            return
        # Make a copy of the original image.
        img = np.copy(image)
        # Create a blank image that matches the original in size.
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )
        # Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), [0, 255, 0], 3)
        # Merge the image with the lines onto the original.
        img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
        # Return the modified image.
        return img
        # return canny

        # if lines is None:
        #     return np.array([])
        # img = np.copy(cropped_image)
        # line_img = np.zeros(
        #     (
        #         img.shape[0],
        #         img.shape[1],
        #         3
        #     ),
        #     dtype=np.uint8,
        # )
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(line_img, (x1, y1), (x2, y2), [0, 0, 255], 3)
        # img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

        # return img

    def perspective_transform(
        self,
        transform_type: str,
        is_dev: bool
    ) -> np.ndarray:
        """Returns bird's eye view of the road ahead"""

        # hough_lines_image = self.hough_lines(self.image.copy())
        # # plt_img = cv2.cvtColor(hough_lines_image, cv2.COLOR_RGB2BGR)

        # plt.figure()
        # plt.imshow(hough_lines_image)

        # Regular Challenge
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

        # Harder Challenge
        # space_around_lane = 258
        # car_hood_offset = 35
        # height = self.image.shape[0]
        # width = self.image.shape[1]
        # horizon = 570

        # vertices = np.array([[
        #     (space_around_lane, height - car_hood_offset),
        #     (450, horizon),
        #     (width - 450, horizon),
        #     (width - space_around_lane, height - car_hood_offset)
        # ]], dtype=np.int32)

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

        warped = cv2.warpPerspective(
            self.image, transform, (width, height), flags=cv2.INTER_NEAREST)
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

    def overlay_image(
        self,
        overlay_image: np.ndarray,
        anchor_y: int,
        anchor_x: int
    ) -> np.ndarray:
        """Overlay an image over another
        """
        foreground = overlay_image.copy()
        background = self.image.copy()
        background_height = background.shape[0]
        background_width = background.shape[1]
        foreground_height = foreground.shape[0]
        foreground_width = foreground.shape[1]
        if foreground_height+anchor_y > background_height or foreground_width+anchor_x > background_width:
            raise ValueError(
                "The foreground image exceeds the background boundaries at this location")

        alpha = 1

        start_y = anchor_y
        start_x = anchor_x
        end_y = anchor_y + foreground_height
        end_x = anchor_x + foreground_width
        blended_portion = cv2.addWeighted(
            foreground,
            alpha,
            background[
                start_y:end_y,
                start_x:end_x, :
            ],
            1 - alpha,
            0,
            background
        )
        background[start_y:end_y, start_x:end_x, :] = blended_portion

        self.image = background

        return background
