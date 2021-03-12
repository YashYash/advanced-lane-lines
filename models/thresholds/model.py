"""
Get and set image gradient threholds. This class has been created to
strong type the various thresholds that can be set. Since it is
not straightforward to create dictionaries with enum keys, this
is the alternative. When constructed, thresholds are defaulted.
An Image model will always be the parent. You can and should
access thresholds through the parent Image.
Ex. image.thresholds.set_thresholds(...)
"""

from typing import Tuple


class Thresholds():
    """Get and set image gradient threholds"""

    saturation: Tuple[int, int]
    lightness: Tuple[int, int]
    brightness: Tuple[int, int]
    sobel: Tuple[int, int]
    magnitude: Tuple[int, int]
    direction: Tuple[float, float]

    def __init__(
        self,
    ) -> None:
        self.saturation = (100, 255)
        self.lightness = (200, 255)
        self.brightness = (220, 255)
        self.sobel_x = (50, 225)
        self.sobel_y = (100, 255)
        self.magnitude = (30, 100)
        self.direction = (0.6, 1.3)

    def set_thresholds(
        self,
        saturation: Tuple[int, int],
        lightness: Tuple[int, int],
        brightness: Tuple[int, int],
        sobel_x: Tuple[int, int],
        sobel_y: Tuple[int, int],
        magnitude: Tuple[int, int],
        direction: Tuple[int, int]
    ) -> None:
        """Update thresholds. If None, value will remain unchanged"""
        if saturation is not None:
            self.saturation = saturation
        if lightness is not None:
            self.lightness = lightness
        if brightness is not None:
            self.brightness = brightness
        if sobel_x is not None:
            self.sobel_x = sobel_x
        if sobel_y is not None:
            self.sobel_y = sobel_y
        if magnitude is not None:
            self.magnitude = magnitude
        if dir is not None:
            self.direction = direction

    def reset_thresholds(self) -> None:
        """Reset all thresholds back to default"""
        self.saturation = (100, 255)
        self.lightness = (200, 255)
        self.brightness = (220, 255)
        self.sobel_x = (50, 225)
        self.sobel_y = (10, 100)
        self.magnitude = (30, 100)
        self.direction = (0.6, 1.3)
