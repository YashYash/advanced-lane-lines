"""
Instance of a Lane in a Road model. The primary purpose of this class is to
return lane lines (Line model).
"""
from typing import Tuple
from models.image import Image
import numpy as np
import cv2

M_PER_PIXEL_Y = 30. / 720
M_PER_PIXEL_X = 3.7 / 700


class Lane():
    """Lane takes a binary_ouput Image model. The main purpose is to
       calculate the curve that fits the lane lines. This will make it
       very easy to construct Line models which will be used to highlight
       the lane lines in the original image
    """
    binary_output: "Image"

    _lane_polygon: np.ndarray
    _lane_exists: bool = False
    _fitted_line_curves: np.ndarray

    num_lines: int = 0
    line_count_max: int = 10
    leftx: np.ndarray = np.array([])
    rightx: np.ndarray = np.array([])
    lefty: np.ndarray = np.array([])
    righty: np.ndarray = np.array([])
    left_fit: np.ndarray = np.array([])
    right_fit: np.ndarray = np.array([])

    def __init__(
        self,
        lane_uuid: str
    ) -> None:
        self.lane_uuid = lane_uuid

    @classmethod
    def from_uuid(
        cls,
        lane_uuid: str
    ) -> "Lane":
        """Create Lane given a lane_uuid"""
        return cls(lane_uuid)

    def update_binary_output(self, binary_output: "Image"):
        """A single instalce of the Lane model is used to process all
           video frames. Once we transition from one frame to anothe,
           the image being processed needs to be updated
        """
        self.binary_output = binary_output

    def find_lane_pixels(self) -> Tuple[int, int, int, int, np.ndarray]:
        """Return the pixels used to represent the left and right lanes"""
        image = self.binary_output.get_image()
        histogram = self.binary_output.get_histogram()
        output_image = np.dstack((image, image, image))*255

        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        n_windows = 9
        margin = 100
        minpix = 50

        window_height = np.int(image.shape[0]//n_windows)

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_indexes = []
        right_lane_indexes = []

        for window in range(n_windows):
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window*window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(
                output_image,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (0, 255, 0),
                2,
                None
            )

            cv2.rectangle(
                output_image,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (0, 255, 0),
                2,
                None
            )

            good_left_indexes = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &
                (nonzerox < win_xleft_high)
            ).nonzero()[0]

            good_right_indexes = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &
                (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_indexes.append(good_left_indexes)
            right_lane_indexes.append(good_right_indexes)

            if len(good_left_indexes) > minpix:
                leftx_current = np.int(
                    np.mean(nonzerox[good_left_indexes])
                )
            if len(good_right_indexes) > minpix:
                rightx_current = np.int(
                    np.mean(nonzerox[good_right_indexes])
                )

        left_lane_indexes = np.concatenate(left_lane_indexes)
        right_lane_indexes = np.concatenate(right_lane_indexes)

        leftx = nonzerox[left_lane_indexes]
        lefty = nonzeroy[left_lane_indexes]
        rightx = nonzerox[right_lane_indexes]
        righty = nonzeroy[right_lane_indexes]

        if (lefty.size == 0 or righty.size == 0):
            if(lefty.size == 0 or righty.size == 0):
                self._lane_exists = False
                return leftx, lefty, rightx, righty, output_image
        else:
            self.righty = righty
            self.rightx = rightx
            self.lefty = lefty
            self.leftx = leftx
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

        self._lane_exists = True
        return leftx, lefty, rightx, righty, output_image

    def fit_polynomial(self) -> np.ndarray:
        """Determine the polynomial constants for the left and right
           lane lines by fitting an equation to their curves
        """
        image = self.binary_output.get_image()
        birds_eye_image = np.dstack((image, image, image))
        leftx, lefty, rightx, righty, output_image = self.find_lane_pixels()
        margin = 100
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        nonzero = output_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_indexes = (
            (
                nonzerox > (
                    left_fit[0] * (nonzerox ** 2) +
                    left_fit[1] * nonzerox + left_fit[2] - margin
                )
            ) &
            (
                nonzerox < (
                    left_fit[0] * (nonzerox ** 2) +
                    left_fit[1] * nonzerox + left_fit[2] + margin
                )
            )
        )

        right_lane_indexes = (
            (
                nonzerox > (
                    right_fit[0] * (nonzerox ** 2) +
                    right_fit[1] * nonzerox + right_fit[2] - margin
                )
            ) &
            (
                nonzerox < (
                    right_fit[0] * (nonzerox ** 2) +
                    right_fit[1] * nonzerox + right_fit[2] + margin
                )
            )
        )

        leftx = nonzerox[left_lane_indexes]
        rightx = nonzerox[right_lane_indexes]
        lefty = nonzeroy[left_lane_indexes]
        righty = nonzeroy[right_lane_indexes]

        ploty = np.linspace(0, output_image.shape[0] - 1, output_image.shape[0])
        try:
            left_fitx = (
                (left_fit[0] * ploty ** 2) +
                (left_fit[1] * ploty) +
                left_fit[2]
            )
            right_fitx = (
                (right_fit[0] * ploty ** 2) +
                (right_fit[1] * ploty) +
                right_fit[2]
            )
        except TypeError:
            print('Function failed to fit a line')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        window_img = np.zeros_like(output_image)
        output_image[
            nonzeroy[left_lane_indexes],
            nonzerox[left_lane_indexes]
        ] = [255, 0, 0]
        output_image[
            nonzeroy[right_lane_indexes],
            nonzerox[right_lane_indexes]
        ] = [0, 0, 255]

        left_line_window1 = np.array([
            np.transpose(np.vstack([left_fitx-margin, ploty]))
        ])
        left_line_window2 = np.array([
            np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))
        ])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([
            np.transpose(np.vstack([right_fitx-margin, ploty]))
        ])
        right_line_window2 = np.array([
            np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))
        ])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(
            window_img, np.int_([left_line_pts]), (0, 255, 0), None, None, None
        )
        cv2.fillPoly(
            window_img,
            np.int_([right_line_pts]),
            (0, 255, 0), None, None, None
        )

        result = cv2.addWeighted(window_img, 1, output_image, 0.5, 0)
        result = cv2.addWeighted(output_image, 1, birds_eye_image, 0.5, 0)

        self._fitted_line_curves = result

        name = self.binary_output.get_name()
        self._store_image(name, self._fitted_line_curves)

        return result

    @staticmethod
    def _store_image(name: str, image: np.ndarray):
        cv2.imwrite(name, image)

    def estimated_lane(self):
        """Once we feel confident that the fit is accurate, we can just
           use the stored left_fit and right_fit to find the lane
        """
        binary_warped = self.binary_output.get_image()
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        left_lane_indexes = (
            (nonzerox > (
                self.left_fit[0]*(nonzeroy**2) +
                self.left_fit[1]*nonzeroy +
                self.left_fit[2] - margin
            )) & (
                nonzerox < (
                    (self.left_fit[0]*(nonzeroy**2)) +
                    (self.left_fit[1]*nonzeroy) +
                    (self.left_fit[2] + margin)
                )
            )
        )

        right_lane_indexes = (
            (nonzerox > (
                self.right_fit[0]*(nonzeroy**2) +
                self.right_fit[1]*nonzeroy +
                self.right_fit[2] - margin
            )) & (
                nonzerox < (
                    (self.right_fit[0]*(nonzeroy**2)) +
                    (self.right_fit[1]*nonzeroy) +
                    (self.right_fit[2] + margin)
                )
            )
        )

        leftx = nonzerox[left_lane_indexes]
        lefty = nonzeroy[left_lane_indexes]
        rightx = nonzerox[right_lane_indexes]
        righty = nonzeroy[right_lane_indexes]

        if lefty.size == 0 or righty.size == 0:
            if self.lefty.size == 0 or self.righty.size == 0:
                self._lane_exists = False
                return out_img
        else:
            self.righty = righty
            self.rightx = rightx
            self.lefty = lefty
            self.leftx = leftx
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

        self._lane_exists = True
        return out_img

    def get_lines(self) -> np.ndarray:
        """Return an image displaying detected lane lines"""
        if self.num_lines < self.line_count_max:
            self.num_lines += 1
            return self.fit_polynomial()
        return self.estimated_lane()

    def get_lane_polygon(self) -> np.ndarray:
        """Return image of the lane filled. This image will
           overlay the original image (video frame)
        """
        image = self.binary_output.get_image()

        left_fitx = (
            self.left_fit[0]*self.lefty**2 +
            self.left_fit[1]*self.lefty + self.left_fit[2]
        )
        right_fitx = (
            self.right_fit[0]*self.righty**2 +
            self.right_fit[1]*self.righty +
            self.right_fit[2]
        )

        image_zeros = np.zeros_like(image).astype(np.uint8)
        polygon_image = np.dstack((image_zeros, image_zeros, image_zeros))
        pts_left = np.array([
            np.flipud(np.transpose(np.vstack([left_fitx, self.lefty])))
        ])
        pts_right = np.array([
            np.transpose(np.vstack([right_fitx, self.righty]))
        ])
        pts = np.hstack((pts_left, pts_right))

        cv2.polylines(
            polygon_image,
            np.int_([pts]),
            isClosed=False,
            color=(34, 198, 238),
            thickness=40
        )

        cv2.fillPoly(polygon_image, np.int_([pts]), (138, 43, 226))

        self._lane_polygon = polygon_image

        return polygon_image

    def add_polygon_to_image(
        self,
        image: np.ndarray,
        show_data: bool
    ) -> np.ndarray:
        """Render the lane polygon on top of the original video frame"""
        polygon_image = Image.from_camera_config(
            "polygon",
            "cam-1",
            self._lane_polygon
        )
        warped_filled_lane = polygon_image.perspective_transform(
            'driver_pov', False
        )

        final_output = cv2.addWeighted(image, 1, warped_filled_lane, 0.5, 0)

        if show_data is True:
            curvature_radius = self.get_curvate_radius()
            curvature_text = (
                'Curvature: ' +
                ' {:0.6f}'.format(curvature_radius) +
                'm '
            )
            cv2.putText(
                final_output,
                curvature_text,
                (40, 70),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        final_output_name = self.binary_output.get_name()
        final_output_name = final_output_name.replace(
            'fit_lines_curves',
            'final_output'
        )
        final_output_name = final_output_name.replace(
            'output_fit_images',
            'output_images'
        )
        self._store_image(final_output_name, final_output)

        return final_output

    def get_curvate_radius(self) -> int:
        """Get the lane lines curvature"""
        if len(self.rightx) == 0 & len(self.righty) == 0:
            return 0

        left_fit_cr = np.polyfit(
            (self.lefty*M_PER_PIXEL_Y),
            (self.leftx*M_PER_PIXEL_X),
            2
        )
        right_fit_cr = np.polyfit(
            (self.righty*M_PER_PIXEL_Y),
            (self.rightx*M_PER_PIXEL_X),
            2
        )

        left_curverad = ((1 + (2*left_fit_cr[0]*np.max(self.lefty) + left_fit_cr[1])**2)**1.5) / (np.absolute(2*left_fit_cr[0]))
        right_curverad = ((1 + (2*right_fit_cr[0]*np.max(self.lefty) + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])

        return int((left_curverad + right_curverad)/2)

    def get_distance_from_center(self) -> int:
        """Calculate the distance from the center of the lane"""
        # distance = 'Distance to middle: ' + ' {:0.6f}'.format(self._distance_to_middle()) + 'm '
        # cv2.putText(out_img, distance, (40,120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        return 0
