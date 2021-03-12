# Advanced Lane Finding Project

[//]: # (Image References)

[final]: ./examples/final.jpg "Final"
[distorted]: ./camera_cal/calibration1.jpg "Distorted"
[undistorted]: ./calibration_test/calibrated-1.jpg "Undistorted"
[og_image]: ./test_images/test1.jpg "Original Image"
[birds_eye_view]: ./birds_eye_view/test1.jpg "Bird's eye view Image"
[binary_output]: ./binary_outputs/test1.jpg "Binary Output"
[stepped_windows]: ./fit_lines_curves/test1.jpg "Stepped windows"
[final_output]: ./final_output/test1.jpg "Final Output"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[pipeline]: ./data_pipeline.jpeg "Pipeline"

### Goal:
##### Given a video streamed from a camera placed at the front of the car looking forward, detect the lane lines and highlight the lane. Calculate the curvature and how far the car is from the middle of the lane. If anyone would like to reference my solution, feel free to hit me up as well. I would be down to help you out. The current solution I have does not work well with really dark shadows, video frames with a lot of sunlight, really sharp turns and video frames that do not have a clear left or right lane lines. The current solution does do a decent job at find the lanes in the first half of the harder challenge video. Hope to improve my current solution by using deep learning later on.

### Current Result:
#### Here's a link [to the final output mp4 file](https://cronoz-assets.s3-us-west-2.amazonaws.com/udacity/output.mp4)

#### Here's a link [to the harder challenge output mp4 file](https://cronoz-assets.s3-us-west-2.amazonaws.com/udacity/harder_challenge_video_output.mp4)


![alt text][final]

### Steps:
* Calibrate the camera using chessboard calibration images
* Process the video stream. For each of the video's frames, the following steps are executed:
  - Undistort the image using the distortion matrix and coefficents calculated in step 1
  - Generate a bird's eye view image
  - Generate a combined binary output image by filtering using sobel (direction and magnitude), saturation, brightness and lightness thresholds
  - Generate a histogram to determine where the nonzero (white) pixels are in the combined binary output. These pixels will represent the left and right lane lines
  - Create windows from the bottom up tracing each lane line. For each new window use the new average middle point of the current window to shift the next windows.
  - Curved lane lines have a polynomial equation. By using the center point in each of windows for each of the lanes, we can determine the coefficients for the fit of the lines' curves
  - Now that we have the equations of the lines, we can create a polygon by filling the area between the two lines. This polygon will represent the lane.
  - Warp the polygon back to the perspective of the original image
  - Overlay the warped polygon on the orginal image to highlight the lane
  - While the above logic runs for each frame, after 10 frames, we start to reuse the equations of the line to predict the polygons the following frames will return. After 10 windows, the equations are reused to fit assume the rest of the curved line. Each iteration the latest coefficients and verified by seeing if nonzero points still land in the window. If they do not the fit for the frame is recalculated, and will once again attempt to reuse the new equation after another 10 frames.

## Pipeline Stages Summarized
![alt text][pipeline]

### Camera Calibration
#### the /camera_cal directory stores the chessboard images that cv2 will use to calibrate the camera's distortion. Camera.calibrate iterates over each of the images in /camera_cal and genrates the camera_cal_output directory which holds the original images with the cornerns highlighted. Camera.calibrate stores the distortion matrix and the distortion coefficients in cam-1_dist_mtx_coe.p pickle file. This config is used by Image classes that have an undistort member method. To test the Camera calibration, you can do any of the following:
* In notebook.ipynb run the "Calibrate Camera" block
* Run calibrate_camera in pipeline.py
* Run the calibration_camera test in models/camera/test_model.py

### Pipeline
#### The easiest way to see how the pipeline works would be to refer to notebook.ipynb. Here are the different cells in the jupyter notebook.
* Data Pipeline:
  - Displays a visaul representation of the data pipline
* Configure os.path to import local modules:
  - Allows for local imports.
* Calibrate Camera
  - Use `camera_cal/*` images to get the distortion matrix and coefficients which would be used to distort and undistort frames obtained from the same camera
  - The notebook visualizes the result of the calibration by displaying an image of the chess board with the corner verticies highlighted. The calibration is correct if the drawn corner points align perfectly with the distorted images.
  - The very last image displayed in the notebook for this cell displays an example of an undistorted next to the original image.
* Undistort > Warp > Binary Threshold
  - Convert `test_images/*` to bird's eye view binary threshold images
  - Running this notebook will visualize the various stages each frame went through to get to the final output.
  - The image is stored under `binary_outputs/`
* Find lane lines in single image
  - Using stepped windows, calculate the equations of the left and right lane line curves.
  - Store the images displaying the stepped windows for each frame under `ouput_fit_images/`
  - Fill area between left and right lines to create a polygon
  - Warp the polygon image back to the perspective of the original image
  - Overlay the warped polygon on top of the original image
  - Store the final image under `final_output/`
* Process Video Stream
  - Iterate over each `project_video.mp4`'s frames
  - For each frame run video_pipeline
  - video_pipeline calls `pipeline()` in `pipeline.py`
  - pipeline() runs all the steps above for each video frame. The only additional logic is that the Lane model is constructed only once. Every time a new frame is processed, Lane.binary_output is updated to the new frame and the same process runs again on the next frame. This continues till the last frame.
  - Once all frames have been processed and stored under `output_images/`, the images are used to generate `output.mp4`

## Models
### Camera
Camera class used to calibrate cameras and undistort images
using a calculated distortion matrix and distortion coefficients.
You only create an instance of this class if you need to calibrate a
camera. Once a camera is calibrated, pickle files will be created. These
pickle files will store the 2D points, 3D points, distortion matrix and the
distortion coefficients. This data is also accessible directly thorough the
class. Data persists till the instance of the class is destroyed.
### Image
An instance of an Image which exposes undistort, perpective transfrom and
get_binary_image methods. self.image is publicly mutable. Therefore the order
in which the class methods are called, matters. To reference the original image
that was used to construct the class, reference self._og_image (private)
### Thresholds
Get and set image gradient threholds. This class has been created to
strong type the various thresholds that can be set. Since it is
not straightforward to create dictionaries with enum keys, this
is the alternative. When constructed, thresholds are defaulted.
An Image model will always be the parent. You can and should
access thresholds through the parent Image.
Ex. image.thresholds.set_thresholds(...)
### Lane
Lane takes a binary_ouput Image model. The main purpose is to
calculate the curve that fits the lane lines. This will make it
very easy to construct Line models which will be used to highlight
the lane lines in the original image
## Udacity Rubric

### Camera Calibration


Refer to the Camera class `models/camera/model.py`. This class constructs a model that exposes the `Camera.calibrate` method. Images under `camera_cal` are used to create and store the distortion matrix and coefficients. First the points in 3D and 2D space are determined by calling cv2.findChessboard. The points are then used to calculate the distortion matrix and coefficients, which are also stored in a seperate pickle file. These values are used by the `Image` class by implementing an `Image.undistort` method. `Image.undistort` uses the clibration config in `cam-1_dist_mtx_coe.p` created by `Camera.calibrate`.
#### From Distorted
![alt text][distorted]
### To Undistorted
![alt text][undistorted]

### Pipeline for test image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][og_image]

In order to generate the binary_output below, first the image was warped to the bird's eye view persepective. This was accomplished by using the distortion matrix and coefficients to warp the image using src and destination coordinates. A trapezoid was created and then warped to the dimensions of the image as a rectange. After warping the image, a combination of direction gradient, sobel magnitude and sobel direction (x, y) thresholds were applied. Followed by saturation, brightness and lightness thresholds. First the lightness, brightness, saturation and sobel x thresholds were applied. Thes thresholds were able to filter out everything other than the white and yellow lanes. Finally a second sobel x gradient threshold, magnitude and directional gradients were used to remove additional noise from the image. Below you can see an exaple of the bird's eye view and it's binary_output
![alt text][birds_eye_view]
![alt text][binary_output]
Once the binary_outputs are generated, the lanes are ready to be detected. This is accomplished by generating a histogram to determine where the nonzero (white) pixels are in the combined binary output. These pixels will represent the left and right lane lines. Then windows are created from the bottom up tracing each lane line. For each new window use the new average middle point of the current window to shift the next windows. Curved lane lines have a polynomial equation. By using the center point in each of the windows for each of the lanes, we can determine the coefficients for the fits of the lines' curves. The equations of the lines can be used to create a polygon by filling the area between the two lines. This polygon will represent the lane. Finally warp the polygon back to the perspective of the original image and overlay the warped polygon on the orginal image to highliht the lane. Below you can see the stepped windows followed by the final output image
![alt text][stepped_windows]
![alt text][final_output]

### Pipeline (video)

#### The pipeline is run for each frame of the video. You can see the pipeline in `pipline.py`. This method does all of the above for each frame. A single `Lane` class is used globally. After 10 frames, instead of calculating the equations for the left and right lines, the equations calculated from the previous frame are reused. If the reused frames do not find any nonzero points that land in the windows, the equations for the lines is calculated again.

Here's a link [to the final output mp4 file](https://cronoz-assets.s3-us-west-2.amazonaws.com/udacity/output.mp4)

Here's a link [to the harder challenge output mp4 file](https://cronoz-assets.s3-us-west-2.amazonaws.com/udacity/harder_challenge_video_output.mp4)

---

### Discussion

#### There are number of improvements that can be made to the current pipline:
- Each Lane class should have Lane.lines which would be instances of Line classes. Each Line would hold the coefficients of their curves. This would allow for further abstraction and fewer large blocks of code that can be found in the Lane class.
- The current pipline breaks if it is not able to find either the left or right lanes due to various reasons. There are a few changes that could be made to handle these scenarios:
    - As long as a single lane is detected, the width of the lane can be calculated and used to simulate where the opposite lane would be by shifting over the lane that was found to the left or right.
    - Playing around more with the gradient thresholds to determine how to prevent change in lighting from impacting the detection of lane lines.
    - To further speed up lane detection, deep learning CNN's can be used to detect lanes in images. The computer vision data along with the deep learning classification can be used to detect lane lines very reliably in all lighting and road conditions.
