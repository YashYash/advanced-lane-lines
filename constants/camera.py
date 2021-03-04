"""Name of the pickle file that will cache the current calibration object
    and image points.
"""
_OBJ_IMG_PTS_PICKLE_FILE_NAME: str = 'obj_img_points.p'

"""Name of the pickle file that will cache the current distortion
    matrix and coefficients. Since every camera will have a unique uuid
    (for now I am just using the uuid field), the pickle file uuid will be prefixed with the uuid (uuid for now). When an Image class calls image.undistort, the camera uuid will be passed to let the image know which distortion config needs to be used to undistort the image.
"""
_DIST_MTX_COE_PICKLE_FILE_NAME = 'dist_mtx_coe.p'
