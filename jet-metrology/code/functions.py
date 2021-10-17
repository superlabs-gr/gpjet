#-*- coding: utf-8 -*-
"""
CREATED ON:  03.08.2021
AUTHOR:      @THANOS_OIKON
DESCRIPTION: Functions to use in main
"""


#importing libraries
import numpy as np
import cv2 as cv
import time
import math
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def nothing(x):
    """
    Dummy function for trackbars
    :param x: position of trackbar
    :return: -
    """
    pass

def Create_Tracking_Mask(img, mask_area_1, mask_area_2, mask_area_3, mask_area_4,is_scaled=False):
    """
    Creating mask for object detection
    :param img: image to apply mask on
    :param is_scaled: -
    :param mask_area_1: mask 1 coordinates
    :param mask_area_2: mask 2 coordinates
    :param mask_area_3: mask 3 coordinates
    :param mask_area_4: mask 4 coordinates
    :return:
    """
    #t1=time.perf_counter()
    mask = np.zeros_like(img)
    match_mask_color_1 = (220, 130, 0)
    match_mask_color_2 = (0, 255, 0)
    match_mask_color_3 = (255, 0, 255)
    match_mask_color_4 = (255, 0, 0)
    cv.fillPoly(mask, np.array([mask_area_1]), match_mask_color_1)
    cv.fillPoly(mask, np.array([mask_area_2]), match_mask_color_2)
    cv.fillPoly(mask, np.array([mask_area_3]), match_mask_color_3)
    cv.fillPoly(mask, np.array([mask_area_4]), match_mask_color_4)
    masked_image = cv.bitwise_and(img, mask)
    """
    if is_scaled:
        masked_image_b = cv.adaptiveThreshold(masked_image[:, :, 0], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv.THRESH_BINARY, 3, 2)
        masked_image_g = cv.adaptiveThreshold(masked_image[:, :, 1], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv.THRESH_BINARY, 3, 2)
        masked_image_r = cv.adaptiveThreshold(masked_image[:, :, 2], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv.THRESH_BINARY, 3, 2)
    else:
    """
    _, masked_image_b = cv.threshold(masked_image[:, :, 0], 50, 255, cv.THRESH_BINARY)
    _, masked_image_g = cv.threshold(masked_image[:, :, 1], 127, 255, cv.THRESH_BINARY)
    _, masked_image_r = cv.threshold(masked_image[:, :, 2], 50, 255, cv.THRESH_BINARY)

    masked_image = cv.merge((masked_image_b, masked_image_g, masked_image_r))
    #t2=time.perf_counter()

    return masked_image

def Features_Extraction_x_axis(img,d_cal_factor,stride):
    """
    Extracting deposited material thickness in frame
    :param img: image to extract deposited materials thickness
    :param d_cal_factor: camera calibration factor for distance
    :param stride: every how many pixels extract thickness
    :return: thickness
    """
    #t1 = time.perf_counter()
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # change image colorspace from BGR to GRAYSCALe
    img=cv.Canny(img,150,255,apertureSize=3,L2gradient=True) # create image's edgemap using Canny edge detector, apply either edge detection or thresholding
    #_,img= cv.threshold(img, 150, 255, cv.THRESH_BINARY) # perform thresholding, turns pixel values to 255 or 0
    thickness=[] # create empty list to store results
    for column in range (0,img.shape[1],stride):
        try:
            Column = img[:,column].tolist() # turn array of pixel values to a list
            a=Column.index(255) # find the index of the first pixel with a value 255 (white), from top to bottom
            b=len(Column) - 1 - Column[::-1].index(255) # find the index of the first pixel with a value 255 (white), from bottom to top
        except:
            th=0 # if there is no white value, then there is 0 thickness
        else:
            th = (b - a) * d_cal_factor
        thickness.append(th)
    #t2 = time.perf_counter()
    return np.array(thickness)


def Features_Extraction_y_axis(img,d_cal_factor,stride):
    """
    Extracting taylor cone's, jet's diameters, angles, areas in frame
    :param img: image to extract features from
    :param d_cal_factor: camera calibration factor for distance
    :param stride: every how many pixels extract features
    :return: diameters, angles, areas, right jet boundaries
    """
    #t1 = time.perf_counter(
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # change image colorspace from BGR to GRAYSCALe
    img=cv.Canny(img,150,255,apertureSize=3,L2gradient=True) # create image's edgemap using Canny edge detector, apply either edge detection or thresholding
    #_, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)  # perform thresholding, turns pixel values to 255 or 0
    # create empty lists to store results
    diameters = []
    angles_L = []
    angles_R = []
    areas = []
    r_boundary = []
    # performing first iteration out of for loop
    Row = img[0, :].tolist() # turn array of pixel values to a list
    a_previous = Row.index(255) # find the index of the first pixel with a value 255 (white), from left to right
    b_previous = len(Row) - 1 - Row[::-1].index(255) # find the index of the first pixel with a value 255 (white), from right to left
    diameters.append((b_previous-a_previous)*d_cal_factor)
    for row in range(stride,img.shape[0],stride):
        try:
            Row = img[row, :].tolist() # turn array of pixel values to a list
            c = Row.index(0)
            if c < 50:
                Row = img[row, c:].tolist()
            a = Row.index(255) # find the index of the first pixel with a value 255 (white), from left to right
            b = len(Row) - 1 - Row[::-1].index(255)  # find the index of the first pixel with a value 255 (white), from right to left
        except:
            diameter = 0
            angle_l = 0
            angle_r = 0
            area = 0
            b = 0
        else:
            angle_l = math.atan(((a - a_previous) / stride)) * 180 / math.pi # angles from jet's left side
            angle_r = math.atan(((b - b_previous) / stride)) * 180 / math.pi # angles from jet's right side
            diameter = (b - a) * d_cal_factor # jet's diameter
            #if angle_l >= angle_r - 20 and angle_l <= angle_r + 20:
                #diameter = diameter * math.cos(angle_l * math.pi/180.)
            area = 0
            for i in range(row-stride,row):
                area=area + (b - a) * math.pow(d_cal_factor,2)
        areas.append(area)
        diameters.append(diameter)
        angles_L.append(angle_l)
        angles_R.append(angle_r)
        areas.append(area)
        r_boundary.append(b)

    #t2=time.perf_counter()
    return np.array(diameters), np.array(angles_L),np.array(angles_R), np.array(areas), np.array(r_boundary)


def Deposition_Point_Estimation(img, corners_to_track, quality, min_distance):
    """
    Estimate deposition point based on the sharp corner that is formed at that point
    :param img: image to find deposition point on
    :param corners_to_track: how many corners to track in image
    :param quality:
    :param min_distance: if more than one corner to track, whats the difference between the corners
    :return: coordinates of corner
    """
    # t1=time.perf_counter() #***start timing
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # converting image from 3 channel (BGR) to UniColor (Grayscale)
    #img = cv.Canny(img,150,255,apertureSize=3,L2gradient=True)


    corners = cv.goodFeaturesToTrack(img, corners_to_track, quality, min_distance)  # detecting important corners
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()  # storing important corner coordinates
        # cv.circle(img,(x,y),5,(0,0,255),-1)  #un-comment in case you wanna check in a new dataset
    # cv.imshow('Important Corners Detection',img)
    # t2=time.perf_counter() #**end timing
    # print(t2-t1) #**print or store timing info
    return x, y


def Initialize_Object_Detection(img, x, y, w, h,l_boundary, u_boundary):
    """
    Initial step for object detection, at the first frame determine where the object will be detected
    :param img: image to detect object
    :param x: x coordinate for left top point of the object
    :param y: y coordinate for left top point of the object
    :param w: width of the included object
    :param h: height of the included object
    :param l_boundary: lower boundary in hsv colorspace to look for
    :param u_boundary: upper boundary in hsv colorspace to look for
    :return: roi(region of interest), roi histogram, tracking window, termination criteria
    """
    track_window=(x,y,w,h) # create initial tracking window
    roi=img[y:y+h,x:x+w] # cropping Region Of Interest
    hsv_roi=cv.cvtColor(roi,cv.COLOR_BGR2HSV) # turning 3-channel image from bgr to hsv colorspace
    # plt.imshow(hsv_roi) #| those to lines are for validation of detecting the right area
    # plt.show()          #| un-comment if working with new dataset
    mask = cv.inRange(hsv_roi, np.array(l_boundary), np.array(u_boundary))  # creating a mask
    # plt.imshow(mask)    #| those to lines are for validation of mask working right, the object to be detected should be coulorful and everything else black
    # plt.show()          #| un-comment if working with new dataset
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    term_criteria = (cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT, 10, 1)
    return roi_hist, track_window, term_criteria


def Mean_Shift_Object_Detection(masked_img, img,color,name,roi_hist,track_window,term_criteria):
    """
    Mean_Shift algorithm acts like a moving window, detecting the Region of Interest initially given,
    there is no change in window dimensions or direction, this kind of object detection algorithm is
    suggested for object in the frame whose dimensions or direction do not under-go significant change.
    In MEW Videos case those objects are the needle the taylor cone and possibly the deposited material
    :param img: image to draw on
    :param masked_img: masked image to perform detection on
    :param color: color to draw rectangle
    :param name: name of object to detect
    :param roi_hist: region of interest histogram
    :param track_window: previous tracking window
    :param term_criteria: previous termination criteria
    :return: processed image, new tracking window, new termination criteria
    """
    hsv=cv.cvtColor(masked_img,cv.COLOR_BGR2HSV) # turning 3-channel BGR to HSV colorspace
    dest=cv.calcBackProject([hsv],[0],roi_hist,[0,180],1) # performing back projection procedure, back projection tells us how well the pixels of a given image (hsv) fit the distribution of pixels in a histogram model (roi_hist).
    _,track_window=cv.meanShift(dest,track_window,term_criteria) # shifting the window based on where the features are detected in dest
    x,y,w,h=track_window # extract new coordinates from new tracking window
    img=cv.rectangle(img,(x,y),(x+w,y+h),color,1) # drawing a rectangle around the detected object
    img=cv.putText(img,name,(x+w+10,y+h),cv.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv.LINE_AA) # put object's name on rectangle's corner
    return img, track_window

def Cam_Shift_Object_Detection(masked_img, img, color, name, roi_hist, track_window, change_dir, term_criteria):
    """
    Cam_Shift algorithm acts like a moving changing window, detecting the Region of Interest initially given,
    there is change in window dimensions or direction, this kind of object detection algorithm is
    suggested for object in the frame whose dimensions or direction  under-go significant change.
    In MEW Videos case the jet is such an object. (jet lag causes the jet to deviate from vertical line)
    :param img: image to draw on
    :param masked_img: masked image to perform detection on
    :param color: color to draw rectangle
    :param name: name of object to detect
    :param roi_hist: region of interest histogram
    :param track_window: previous tracking window
    :param change_dir: rotate rectangle or not
    :param term_criteria: previous termination criteria
    :return: processed image, new tracking window, new termination criteria
    """
    hsv = cv.cvtColor(masked_img, cv.COLOR_BGR2HSV)  # turning 3-channel BGR to HSV colorspace
    dest = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)  # performing back projection procedure, back projection tells us how well the pixels of a given image (hsv) fit the distribution of pixels in a histogram model (roi_hist).
    ret, track_window = cv.CamShift(dest, track_window, term_criteria)  # shifting the window based on where the features are detected in dest
    if change_dir:
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img = cv.polylines(img, [pts], True, color, 1) # drawing a rotated rectangle around the detected object
        img = cv.putText(img, name, (pts[2][0] + 10, pts[2][1]),cv.FONT_HERSHEY_SIMPLEX,0.4, color, 1,cv.LINE_AA) # put object's name on rectangle's corner
    else:
        x, y, w, h = track_window
        img = cv.rectangle(img, (x, y), (x + w, y + h),color, 1)  # drawing a rectangle around the detected object
        img = cv.putText(img, name, (x + w + 10, y + h),cv.FONT_HERSHEY_SIMPLEX,0.4, color, 1,cv.LINE_AA) # put object's name on rectangle's corner
    return img, track_window



def Feat_Extr(img, track_windows, d_cal_factor,stride_x,stride_y):
    """
    Function extracting all features from every frame og the video
    :param img: image to extract features from
    :param track_windows:
    :param d_cal_factor: calibration factor regarding distance
    :param stride_x: every how many pixels read deposited material thickness
    :param stride_y: every how many pixels read angles, areas and diameters
    :return: angles,diameters,areas for taylor cone jet etc
    """
    dm_x,dm_y,dm_w,dm_h=track_windows[3]
    thickness=Features_Extraction_x_axis(img[dm_y:dm_y+dm_h,dm_x:dm_x+dm_w],d_cal_factor,stride_x)
    tc_x,tc_y,tc_w,tc_h=track_windows[1]
    j_x,j_y,j_w,j_h=track_windows[2]
    tcdiameter, tcangle_l, tcangle_r, tcarea, tcr_boundary = Features_Extraction_y_axis(img[tc_y:tc_y+tc_h,tc_x:tc_x+tc_w],d_cal_factor, stride_y)
    jdiameter, jangle_l, jangle_r, jarea, jr_boundary = Features_Extraction_y_axis(img[j_y:j_y+j_h,j_x:j_x+j_w],d_cal_factor, stride_y)
    diameter, angle_l, angle_r, area, r_boundary = Features_Extraction_y_axis(img[tc_y:j_y+j_h,tc_x:tc_x+tc_w],d_cal_factor, stride_y)
    return np.array(thickness), np.array(tcdiameter), np.array(tcangle_l), np.array(tcangle_r), np.array(tcarea), np.array(tcr_boundary), \
           np.array(jdiameter), np.array(jangle_l), np.array(jangle_r), np.array(jarea),  np.array(jr_boundary), \
           np.array(diameter), np.array(angle_l), np.array(angle_r), np.array(area), np.array(r_boundary)

def DepPointEst(img,window):
    """
    Find the deposition point on the image
    :param img: image to find deposition point on
    :param window: window in the image of expected deposition area
    :return: coordinates of deposition point
    """
    x,y,w,h=window
    dep_x,dep_y=Deposition_Point_Estimation(img[y:y+h,x:x+w],1,0.1,100)
    dep_x=dep_x+x
    dep_y=dep_y+y
    return dep_x,dep_y


def DistEst(img,dep_x,dep_y,track_window, d_cal_factor):
    """
    Function which takes results from object detection and deposition point estimation
    and calculates the distance between deposition point and centerline of the nozzle,
    also draws a graphic representation on the image
    :param img:  image to draw on
    :param dep_x: x coordinate of deposition point
    :param dep_y: y coordinate of deposition point
    :param track_window: nozzles tracking window
    :return: deposition distance and image with graphic representation
    """
    x, y, w, h = track_window
    dep_dist = (dep_x - (x + w / 2)) * d_cal_factor
    cv.circle(img, (dep_x, dep_y), 5, (0, 0, 255), -1) # create a red circle to denote deposition point
    cv.line(img, (x + int(w / 2), y + int(h / 2)), (x + int(w / 2), img.shape[0]), (0, 0, 255), 2) # create a line from nozzle's center to bottom
    cv.arrowedLine(img, (dep_x, dep_y), (x + int(w / 2), dep_y), (0, 130, 255), 2) # create a double arrow
    cv.arrowedLine(img, (x + int(w / 2), dep_y), (dep_x, dep_y), (0, 130, 255), 2) # create a double arrow
    return img, dep_dist


def ObjSegDet(img,roi_hists,track_windows,term_criterias,masks):
    """
    Function which performs Object Segmentation and Object detection
    :param img: image to perform segmentation and detection on
    :param roi_hists: roi histograms of [nozzle,taylor cone,jet,deposited material]
    :param track_windows: previous tracking windows of [nozzle,taylor cone, jet, deposited material]
    :param term_criterias: previous termination criteria of [nozzle, taylor cone, jet, deposited material
    :param masks: masks for segmentation
    :return: image with detection boxes, new tracking windows, termination criteria
    """
    masked_img=Create_Tracking_Mask(img,masks[0],masks[1],masks[2],masks[3])
    img,track_windows[0]=Cam_Shift_Object_Detection(masked_img,img,(220,130,0),"NEEDLE",roi_hists[0],track_windows[0],False, term_criterias[0])
    img,track_windows[1]=Cam_Shift_Object_Detection(masked_img,img,(0,255,0),"T_CONE",roi_hists[1],track_windows[1], False, term_criterias[1])
    img,track_windows[2]=Cam_Shift_Object_Detection(masked_img,img,(255,0,255),"JET",roi_hists[2],track_windows[2],True,term_criterias[2])
    #img,track_windows[3]=Cam_Shift_Object_Detection(masked_img,img,(255,0,0),"DEP_MAT",roi_hists[3],track_windows[3],False,term_criterias[3])
    return img,masked_img, track_windows

def ObjDet_Init(img,track_windows_init,masks):
    """
    Initializing object detection for the four "objects" of interest (Nozzle, Taylor Cone, Jet Deposited Material)
    :param img: image to start detecting on
    :param track_windows_init: initial regions of interest for every object
    :param masks: masks to apply for segmentation
    :return: regions of interest histograms, new tracking windows, termination criteria
    """
    # creating lists to store
    roi_hists=[]
    track_windows=[]
    term_criterias=[]
    # segmenting image
    masked_img = Create_Tracking_Mask(img, masks[0], masks[1], masks[2], masks[3])
    # initialize object detection for nozzle
    x,y,w,h=track_windows_init[0]
    roi_hist1, track_window1, term_criteria1 = Initialize_Object_Detection(masked_img, x, y, w, h, (85, 255, 255), (105, 255, 255))
    roi_hists.append(roi_hist1)
    track_windows.append(track_window1)
    term_criterias.append(term_criteria1)
    # initialize object detection for taylor cone
    x,y,w,h=track_windows_init[1]
    roi_hist2, track_window2, term_criteria2 = Initialize_Object_Detection(masked_img, x, y, w, h, (50, 255, 255), (70, 255, 255))
    roi_hists.append(roi_hist2)
    track_windows.append(track_window2)
    term_criterias.append(term_criteria2)
    # initialize object detection for jet
    x,y,w,h=track_windows_init[2]
    roi_hist3, track_window3, term_criteria3 = Initialize_Object_Detection(masked_img, x, y, w, h, (130, 255, 255), (160, 255, 255))
    roi_hists.append(roi_hist3)
    track_windows.append(track_window3)
    term_criterias.append(term_criteria3)
    # initialize object detection for deposited material
    x,y,w,h=track_windows_init[3]
    roi_hist4, track_window4, term_criteria4 = Initialize_Object_Detection(masked_img, x, y, w, h, (110, 255, 255), (140, 255, 255))
    roi_hists.append(roi_hist4)
    track_windows.append(track_window4)
    term_criterias.append(term_criteria4)

    return roi_hists,track_windows,term_criterias

def store_results(features,path):

    np.savetxt(path + "Deposited_Material_Thickness.csv", np.array(features.thickness), delimiter=',')

    np.savetxt(path + "TC_Diameters.csv", np.array(features.TCdiameters), delimiter=',')
    np.savetxt(path + "TC_Areas.csv", np.array(features.TCareas), delimiter=',')
    np.savetxt(path + "TC_Angles_Left.csv", np.array(features.TCangles_l), delimiter=',')
    np.savetxt(path + "TC_Angles_Right.csv", np.array(features.TCangles_r), delimiter=',')
    np.savetxt(path + "TC_movement.csv", np.array(features.TCr_boundary), delimiter=',')
    np.savetxt(path + "Jet_Diameters.csv", np.array(features.Jdiameters), delimiter=',')
    np.savetxt(path + "Jet_Areas.csv", np.array(features.Jareas), delimiter=',')
    np.savetxt(path + "Jet_Angles_Left.csv", np.array(features.Jangles_l), delimiter=',')
    np.savetxt(path + "Jet_Angles_Right.csv", np.array(features.Jangles_r), delimiter=',')
    np.savetxt(path + "Jet_movement.csv", np.array(features.Jr_boundary), delimiter=',')

    np.savetxt(path + "Total_Diameters.csv", np.array(features.Total_diameters), delimiter=',')
    np.savetxt(path + "Total_Areas.csv", np.array(features.Total_areas), delimiter=',')
    np.savetxt(path + "Total_Angles_Left.csv", np.array(features.Total_angles_l), delimiter=',')
    np.savetxt(path + "Total_Angles_Right.csv", np.array(features.Total_angles_r), delimiter=',')
    np.savetxt(path + "Totalmovement.csv", np.array(features.Totalr_boundary), delimiter=',')