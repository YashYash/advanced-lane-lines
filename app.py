"""
Export high-level apis that can be call to execute lane detection methods
"""
import glob
from models import Camera

images = glob.glob('camera_cal/calibration*.jpg')
camera = Camera.from_calibration_config("cam-1", False, images, (9,6))
print("Hello")
