#-*- coding: utf-8 -*-
"""
CREATED ON:  03.08.2021
AUTHOR:      @THANOS_OIKON
DESCRIPTION: main function to run code sequentially
"""

# importing libraries
import numpy
import cv2 as cv
import time
import datetime
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import functions
from functions import *

def mainSequantially():
    # creating lists to store data

    dep_dist=[]
    thicknesses=[]
    tcdiameters =[]
    tcangles_L = []
    tcangles_R =[]
    tcareas = []
    tcmovement = []

    jdiameters = []
    jangles_L = []
    jangles_R = []
    jareas = []
    jmovement = []

    diameters = []
    angles_L = []
    angles_R = []
    areas = []
    movement = []

    times=[]
    track_windows_init = []
    masks = []

    filename = 'Video S1'
    speed = ' (2890mm.min)'
    read_filepath = 'C:/Users/thano/Desktop/ComputerVisionMEW_Paper/Stored_Videos/data/' + filename + '/'
    save_filepath = 'C:/Users/thano/Desktop/ComputerVisionMEW_Paper/Results/Results/' + filename + speed + '/Sequential/'
    filepath = read_filepath + filename + speed + '.mp4'
    windows = np.loadtxt('PrinterSetUps/' + filename + ' set up.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))

    # distance calibrating factor for Video S1 and Video S2. Add more if available
    if filename == 'Video S1':
        d_cal_factor = 0.0115 * math.pow(10, -3)
    elif filename == 'Video S2':
        d_cal_factor = 0.0170 * math.pow(10, -3)
    fps = 50 # camera frames per seconds
    sp_cal_factor = d_cal_factor * fps  # speed  calibrating factor
    stride_x = 30  # stride length (in pixels) to calculate deposited material thickness
    stride_y = 3  # stride length (in pixels) to calculate diameters, angles, areas etc of jet and tylor cone


    cap = cv.VideoCapture(filepath) # creating video reader
    ret, frame = cap.read()

    dep_window = (0, int(windows[0, -1]), frame.shape[1], int(windows[1, -1] - windows[0, -1]))
    for i in range(0, 4):
        track_windows_init.append((0,int(windows[0,i]),frame.shape[1],int(windows[1,i]-windows[0,i])))
        masks.append([(0,int(windows[0,i])),(frame.shape[1],int(windows[0,i])),(frame.shape[1],int(windows[1,i])),(0,int(windows[1,i]))])
    print(frame.shape)
    # reversing image
    frame = cv.bitwise_not(frame)

    fourcc = cv.VideoWriter_fourcc(*'XVID') # codec needed for recording video results
    final_output_1=cv.VideoWriter(save_filepath + filename + " detected.mp4", fourcc, 50, (frame.shape[1],frame.shape[0])) # creating video writer
    final_output_2=cv.VideoWriter(save_filepath + filename + " segmented.mp4", fourcc, 50, (frame.shape[1],frame.shape[0])) # creating video writer

    # initialize object detection
    roi_hists, track_windows, term_criterias = ObjDet_Init(frame, track_windows_init, masks) # initializing object detection

    # extract features for first frame
    thickness, tcdiameter, tcangle_l, tcangle_r, tcarea, tcr_boundary_previous,jdiameter, jangle_l, jangle_r, jarea, jr_boundary_previous, diameter, angle_l, angle_r, area, r_boundary_previous = Feat_Extr(frame, track_windows_init, d_cal_factor, stride_x,stride_y) # extracting features from image for taylor cone, jet, and both of them together

    # find deposition point for first frame
    dep_x, dep_y = DepPointEst(frame, dep_window)
    img, dist = DistEst(frame, dep_x, dep_y, track_windows[0], d_cal_factor)

    # store deposition distance and features
    dep_dist.append(dist)
    thicknesses.append(thickness)
    tcdiameters.append(tcdiameter)
    tcangles_R.append(tcangle_r)
    tcangles_L.append(tcangle_l)
    tcareas.append(np.sum(tcarea))

    jdiameters.append(jdiameter)
    jangles_R.append(jangle_r)
    jangles_L.append(jangle_l)
    jareas.append(jarea)

    diameters.append(diameter)
    angles_R.append(angle_r)
    angles_L.append(angle_l)
    areas.append(area)


    while cap.isOpened():
        t1 = time.perf_counter()
        ret,frame = cap.read() # reading a new frame
        if ret==True:
            t2 = time.perf_counter()
            # reversing image
            frame=cv.bitwise_not(frame)

            t3 = time.perf_counter()
            # calculate diameters, angles, areas etc
            thickness, tcdiameter, tcangle_l, tcangle_r, tcarea, tcr_boundary, jdiameter, jangle_l, jangle_r, jarea, jr_boundary, diameter, angle_l, angle_r, area, r_boundary = Feat_Extr(frame, track_windows_init, d_cal_factor, stride_x,stride_y)  # extracting features from image for taylor cone, jet, and both of them together

            t4 = time.perf_counter()
            # find deposition point
            dep_x, dep_y = DepPointEst(frame, dep_window)

            t5 = time.perf_counter()
            # perform object detection and detection
            img, masked_img, track_windows = ObjSegDet(frame, roi_hists, track_windows, term_criterias, masks)  # initializing object detection
            # calculate distance of deposition point to  needle centerline and draw graphics
            img, dist = DistEst(img, dep_x, dep_y, track_windows[0], d_cal_factor)
            masked_img, dist = DistEst(masked_img, dep_x, dep_y, track_windows[0], d_cal_factor)

            t6 = time.perf_counter()
            # store results to lists
            dep_dist.append(dist)
            print(dep_dist)
            thicknesses.append(thickness)
            tcdiameters.append(tcdiameter)
            tcangles_R.append(tcangle_r)
            tcangles_L.append(tcangle_l)
            tcareas.append(np.sum(tcarea))
            tcmovement.append((tcr_boundary-tcr_boundary_previous) * sp_cal_factor)
            tcr_boundary_previous=tcr_boundary

            jdiameters.append(jdiameter)
            jangles_R.append(jangle_r)
            jangles_L.append(jangle_l)
            jareas.append(jarea)
            jmovement.append((jr_boundary-jr_boundary_previous) * sp_cal_factor)
            jr_boundary_previous=jr_boundary

            diameters.append(diameter)
            angles_R.append(angle_r)
            angles_L.append(angle_l)
            areas.append(area)
            movement.append((r_boundary-r_boundary_previous) * sp_cal_factor)
            r_boundary_previous=r_boundary

            t7 = time.perf_counter()
            cv.imshow("Feed", img) # show processed frame

            t8 = time.perf_counter()
            final_output_1.write(img) # write processed frame in video
            final_output_2.write(masked_img) # write processed frame in video

            t9 = time.perf_counter()
            times.append(np.array([t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7,t9-t8]))
            print(t9-t1)
            k = cv.waitKey(1)
            if k == 27: # press ESC to stop
                break
        else:
            break
    #release camera
    cap.release()
    #close video shower
    cv.destroyAllWindows()

    # saving results in csv files after ending video or stop recording
    np.savetxt(save_filepath + "ProcessingTime.csv", np.array(times), delimiter=',')

    np.savetxt(save_filepath + "Deposition_Distance.csv", np.array(dep_dist), delimiter=',')
    np.savetxt(save_filepath + "Deposited_Material_Thickness.csv", np.array(thickness), delimiter=',')

    np.savetxt(save_filepath + "TC_Diameters.csv", np.array(tcdiameters), delimiter=',')
    np.savetxt(save_filepath + "TC_Areas.csv", np.array(tcareas), delimiter=',')
    np.savetxt(save_filepath + "TC_Angles_Left.csv", np.array(tcangles_L), delimiter=',')
    np.savetxt(save_filepath + "TC_Angles_right.csv", np.array(tcangles_R), delimiter=',')
    np.savetxt(save_filepath + "TC_movement.csv", np.array(tcmovement), delimiter=',')


    np.savetxt(save_filepath + "Jet_Diameters.csv", np.array(jdiameters), delimiter=',')
    np.savetxt(save_filepath + "Jet_Areas.csv", np.array(jareas), delimiter=',')
    np.savetxt(save_filepath + "Jet_Angles_Left.csv", np.array(jangles_L), delimiter=',')
    np.savetxt(save_filepath + "Jet_Angles_right.csv", np.array(jangles_R), delimiter=',')
    np.savetxt(save_filepath + "Jet_movement.csv", np.array(jmovement), delimiter=',')

    np.savetxt(save_filepath + "Total_Diameters.csv", np.array(diameters), delimiter=',')
    np.savetxt(save_filepath + "Total_Areas.csv", np.array(areas), delimiter=',')
    np.savetxt(save_filepath + "Total_Angles_Left.csv", np.array(angles_L), delimiter=',')
    np.savetxt(save_filepath + "Total_Angles_right.csv", np.array(angles_R), delimiter=',')
    np.savetxt(save_filepath + "Total_movement.csv", np.array(movement), delimiter=',')




if __name__=="__main__":
    mainSequantially()
