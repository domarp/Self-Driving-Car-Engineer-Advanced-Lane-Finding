### Import modules
import pickle
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def camera_calibration(nx, ny, input_filepath, draw_calibrated = True):
    """ Function to calibrate camera
    
    Args:
        nx (int) : Corners along rows
        ny (int) : Corners along columns
        input_filepath (str) : Path to chesboard images for calibration
        draw_calibrated (boolean) : Write output calibrated images, 1: Write output
        
    Returns:
        Dictionary of camera calibration metrics (as returned by cv2.calibrateCamera()
    
    """   
    
    reference_object_points = np.array(np.mgrid[0:nx,0:ny, 0:1],dtype = np.float32).T.reshape(-1,3)
    calibration_images = glob.glob(input_filepath)
    
    image_points = []
    object_points = []
    
    for image_name in calibration_images:
        image = cv2.imread(image_name)
        imshape = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)
        if ret:
            image_points.append(corners)
            object_points.append(reference_object_points)
            if draw_calibrated:
                cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
                cv2.imwrite('../output_images/camera_cal/' + image_name.split('/')[-1][:-4] + '_chessboard_corners.jpg', image)
        else:
            pass
            # print("No corners found for image", image_name)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (imshape[1],imshape[0]), None, None)
    return({'ret' : ret, 'mtx' : mtx, 'dist' : dist, 'rvecs' : rvecs, 'tvecs' : tvecs})


def image_undistortion(distorted_image, nx, ny, camera_calibration_params,write_undistorted = False, 
                       write_file_name = None):
    """Returns Undistorted Image
    
    Args:
        distorted_image (array) : Distorted image
        nx (int) : Corners along rows
        ny (int) : Corners along columns
        camera_calibration_params (dict) : disctionary of Camera Calibration Parameters 
        write_undistorted (boolean) : Write the undistorted image
        write_file_name (str) : If write_undistored = True, then the filename (folder path is already specified) of the output image (Sample value: 'test1.jpg')
        
    Returns:
        Array of undistorted image
    """
    undistort_image = cv2.undistort(distorted_image, camera_calibration_params['mtx'], camera_calibration_params['dist'], 
                                    None, camera_calibration_params['mtx'])
    if write_undistorted:
        if write_file_name != None:
            cv2.imwrite('../output_images/undistorted_images/' + write_file_name.split('/')[-1][:-4] + '_undistorted.jpg', undistort_image)
        else:
            print("Provide filename")
            return 0
    return undistort_image 


def show_image(image, isBinary = False):
    """ Display Image"""
    if isBinary:
        f, ax1 = plt.subplots(1, 1, figsize=(10,5))
        ax1.imshow(image, cmap = 'gray')
    else:
        f, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        
def create_binary_image(undistorted_image, mask_points, write_binary = False, write_file_name = None):
    """
    Returns a binary image after applying thresholds on Sobel gradients 
    of appropriate color transforms of undistorted images
    
    Args
        undistorted_image (array) : Undistorted Image array
        write_binary (boolean) : Write the Binary image
        write_file_name (str) : If write_binary = True, then the filename (folder path is already specified) of the output image (Sample value: 'test1.jpg')
        
     Returns
         Binary image array after color transforms and gradient based thresholding
    
    """
    # Define Thresholds
    thresh_min = 20#20
    thresh_max = 100#150
    s_thresh_min = 200#150
    s_thresh_max = 255#200

    
    hls = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize = 5)
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    l_channel = hls[:,:,1]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = 5)
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    lxbinary = np.zeros_like(scaled_sobel)
    lxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1    
    
    combined_binary = np.zeros_like(s_channel)
    combined_binary[((sxbinary==1) | (lxbinary==1))] = 1
    mask = np.zeros_like(s_channel)
    cv2.fillPoly(mask, np.int_([mask_points]), 1)
    combined_binary_mask = np.bitwise_and(combined_binary, mask)
    
    if write_binary:
        if write_file_name != None:
            cv2.imwrite('../output_images/binary_images/' + write_file_name.split('/')[-1][:-4] + '_binary.jpg', combined_binary*255)
        else:
            print("Provide filename")
            return 0
    
    return combined_binary_mask



def create_or_inverse_perspective(image, source_points, destination_points, inverse = False,
                                 write_transformed = False, write_file_name = None):
    """ Returns perspective transformed image or inverse transform as specified by the inverse flag
    
    Args
        image (array) : Image array
        source_points (array): Source points for perspective transformation; used as destination_points in case inverse = True
        destination_points (array) : Destination points for perspective transformation; used as source_points in case inverse = True
        inverse (boolean) : Create inverse perspective transform
        write_transformed (boolean) : Write the Binary image
        write_file_name (str) : If write_transformed = True, then the filename (folder path is already specified) of the output image (Sample value: 'test1.jpg')
        
    Returns
        If inverse = False, warped image array else unwarped image array
    """

    img_size = (image.shape[1], image.shape[0])
    if inverse:
        inverse_M = cv2.getPerspectiveTransform(destination_points, source_points)
        transformed = cv2.warpPerspective(image, inverse_M, img_size, flags=cv2.INTER_NEAREST)
    else:
        M = cv2.getPerspectiveTransform(source_points, destination_points)
        transformed = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_NEAREST)  
        
    if write_transformed:
        if write_file_name != None:
            cv2.imwrite('../output_images/warped_images/' + write_file_name.split('/')[-1][:-4] + '_warped.jpg', transformed*255)
        else:
            print("Provide filename")
            return 0
        
    return transformed


def draw_line(image, start, end):
    """ Draw a line on the image
    
    Args
        image (array) : Image array
        start (array) : Array with line start coordinates
        end (array) : Array with line end coordinates
        
    Returns
        Image array with line layer
    """
    img = cv2.line(image, (start[0], start[1]), (end[0], end[1]), [0, 0, 255], 5)
    return img

    
def draw_polygon(image, edges, write_polygon = False, write_file_name = None):
    """ Draw a polygon on the image
    
    Args
        image (array) : Image array
        edges (array) : Array with Edge coordinates
        write_polygon (boolean) : Write the image with polygon
        write_file_name (str) : If write_polygon = True, then the filename (folder path is already specified) of the output image (Sample value: 'test1.jpg')
        
    Returns
        Image array with polygon layer
    """
    img = image.copy()
    draw_line(img, edges[0], edges[1])
    draw_line(img, edges[1], edges[2])
    draw_line(img, edges[2], edges[3])
    draw_line(img, edges[3], edges[0])
    if write_polygon:
        if write_file_name != None:
            cv2.imwrite('../output_images/warped_images/' + write_file_name.split('/')[-1][:-4] + '_polygon.jpg',img)
        else:
            print("Provide filename")
            return 0

    return img


def fill_poly(undistorted, warped, ploty, left_fit, right_fit, source_points, destination_points):
    """ Fill polygon with specific shade to highlight region of interest
    
    Args
        undistorted (array) : Undistorted Image array
        warped (Array) : Warped image array
        ploty (Array) : As defined previously
        left_fit (Array) : coefficients for polynomial fit of left lane
        right_fit (Array) : coefficients for polynomial fit of right lane
        source_points (array): Source points for perspective transformation; used as destination_points in case inverse = True
        destination_points (array) : Destination points for perspective transformation; used as source_points in case inverse = True
        
    Returns
        Image array with region of interest as a shaded polygon 
    """
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = create_or_inverse_perspective(color_warp, source_points, destination_points, True) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    return(result)


def create_final_frame(unwarped_image, radii, dist_to_center, write_file = False, write_file_name = None):
    """ Create the final output frame
    
    Args
        unwarped_image (array) : Inverse perspective transformed image
        radii (float) : Radius of curvature of detected road
        dist_to_center (float) : Distance of lane center to camera center
        write_file (boolean) : Write the image 
        write_file_name (str) : If write_file = True, then the filename (folder path is already specified) of the output image (Sample value: 'test1.jpg')
        
    Returns
        Final output with detected region of interest highlighted and annotations of radius of curvature and distance to center 
    """
    img = unwarped_image.copy()
    cv2.putText(img, f"Radius : {radii:.3f}m",
           org = (200,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color = [255, 255, 255], thickness = 5)
    cv2.putText(img, f"Distance to center : {dist_to_center:.3f}m", org = (100,120), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color = [255, 255, 255], thickness = 5)
    
    if write_file:
        if write_file_name != None:
            cv2.imwrite('../output_images/final_frames/' + write_file_name.split('/')[-1][:-4] + '_final.jpg', img)
        else:
            print("Provide filename")
            return 0
        
    return img


def image_sharpen(image):
    """ Sharpen the input image
    
    Args
        image (Array) : Image array to be sharpened
        
    Returns
        Sharpened image
    """
    blurred_original_image = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred_original_image, -0.5, 0.0)
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened
    



