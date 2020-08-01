### Import modules
import pickle
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_lane_pixels(binary_warped, leftx_base = None, rightx_base = None):
    """Find Lane Pixels"""
    nwindows = 10
    margin = 70
    minpix = 20
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    histogram = np.sum(binary_warped[1*binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]//2)
    if leftx_base == None:
        leftx_base = np.argmax(histogram[margin:midpoint-margin])
        rightx_base = np.argmax(histogram[midpoint+margin:-margin]) + midpoint
        
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
        
    leftx_current = leftx_base
    rightx_current = rightx_base
        
    left_lane_inds = []
    right_lane_inds = []
    left_shift = 0
    right_shift = 0
        
    for window in range(nwindows):
        if window == 1:
            leftx_base = leftx_current
            rightx_base = rightx_current
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
            
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.median(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: 
            rightx_current = np.int(np.median(nonzerox[good_right_inds]))
        
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass
        
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img, leftx_base, rightx_base


def fit_polynomial(binary_warped, leftx_base = None, rightx_base = None, ploty = None, left_fit = None, right_fit = None,
                   write_file = False, write_file_name = None):
    """ Fit lane polynomial """
    min_pix_replot = 8000
    leftx, lefty, rightx, righty, out_img, leftx_base, rightx_base = find_lane_pixels(binary_warped, leftx_base, rightx_base)

    if ((left_fit==None) | (leftx.shape[0] > min_pix_replot)):
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        pass
    if ((right_fit==None)| (rightx.shape[0] > min_pix_replot)):
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        pass
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    out_img_plotted = out_img.copy()
    cv2.polylines(out_img_plotted, [np.column_stack((np.int_(left_fitx), np.int_(ploty)))], False, [255,255,0], 5)
    cv2.polylines(out_img_plotted, [np.column_stack((np.int_(right_fitx), np.int_(ploty)))], False, [255,255,0], 5)

    if write_file:
        if write_file_name != None:
            cv2.imwrite('../output_images/warped_lanes/' + write_file_name.split('/')[-1][:-4] + '_warped.jpg', cv2.cvtColor(out_img_plotted, cv2.COLOR_RGB2BGR))
        else:
            print("Provide filename")
            return 0

    return out_img, out_img_plotted, ploty, left_fit, right_fit, leftx_base, rightx_base


def measure_curvature_real(ploty, ym_per_pix, xm_per_pix, left_fit, right_fit):
    """ Calculates the curvature of polynomial functions in meters """
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]) 
    
    return (left_curverad+ right_curverad)//2


def measure_distance_to_center(imshape, ploty, ym_per_pix, xm_per_pix, left_fit, right_fit):
    """ Calculates the distance of camera center to lane center in meters """
    y_eval = np.max(ploty)
    
    left = np.polyval(left_fit, y_eval)
    right = np.polyval(right_fit, y_eval)
    center = imshape[1]/2
    dist_to_center = (center - (left + right)/2)*xm_per_pix
    return dist_to_center




        
    


