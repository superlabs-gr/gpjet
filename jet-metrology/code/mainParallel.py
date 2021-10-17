#-*- coding: utf-8 -*-
"""
CREATED ON: 02.15.2021
AUTHOR: @thanos_oikon
DESCRIPTION:This file is the main function of the concurrent algorithm with multithreading of the Computer Vision Tool
"""

# importing libraries
import numpy as np
import cv2 as cv
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import datetime
from threading import Thread, Lock
from multiprocessing import Process, Queue

import functions
from functions import *
import classes
from classes import VideoReader
from classes import VideoShower
from classes import VideoWriter
from classes import Features
from classes import ObjSegDetData

def Multi_ObjSegDet(frames, frame0, roi_hists, track_windows, dep_window, term_criterias, masks, d_cal_factor, fps,save_filepath):
    """
    Function which performs Object Segmentation and Object detection using a different CPU core (multiprocessing)
    :param frames: queue of images to perform segmentation and detection on
    :param frame0: frame0 used for initialization of video shower and video writer
    :param roi_hists: roi histograms of [nozzle,taylor cone,jet,deposited material]
    :param track_windows: previous tracking windows of [nozzle,taylor cone, jet, deposited material]
    :param dep_window: window to look for deposition point
    :param term_criterias: previous termination criteria of [nozzle, taylor cone, jet, deposited material
    :param masks: masks for segmentation
    :param d_cal_factor: distance calibration factor of camera
    :param fps: frames per second of camera
    :param save_filepath: filepath to use for Video Storage
    """
    # creating list to store deposition distance calculated
    dep_dist=[]
    # find deposition point in first frame
    dep_x, dep_y = DepPointEst(frame0, dep_window)
    # find distance and create graphics
    frame0, dist = DistEst(frame0, dep_x, dep_y, track_windows[0], d_cal_factor)
    dep_dist.append(dist)
    # starting threads to display and store results
    video_shower = VideoShower(frame0,fps,name='Object Detection').start()
    video_writer1 = VideoWriter(frame0).start(save_filepath + "Video_detected.mp4",fps)
    video_writer2 = VideoWriter(frame0).start(save_filepath + "Video_segmented.mp4",fps)
    # entering main loop
    while True:
        # break loop condition and close all threads
        if not video_shower.started or not video_writer1.started or not video_writer2.started:
            video_shower.stop()
            video_writer1.stop()
            video_writer2.stop()
            break
        # if there is a new frame available detect objects and find deposition point
        if frames.qsize() > 0:
            t1 = time.perf_counter()
            img = frames.get() # get new frame

            t2 = time.perf_counter()
            #find deposition point
            dep_x, dep_y = DepPointEst(img, dep_window)
            t3 = time.perf_counter()
            # segment image
            masked_img = Create_Tracking_Mask(img, masks[0], masks[1], masks[2], masks[3])

            # track nozzle
            img, track_windows[0] = Cam_Shift_Object_Detection(masked_img, img, (220, 130, 0), "NEEDLE", roi_hists[0],
                                                           track_windows[0], False, term_criterias[0])
            # track taylor cone
            img, track_windows[1] = Cam_Shift_Object_Detection(masked_img, img, (0, 255, 0), "T_CONE", roi_hists[1],
                                                           track_windows[1], False, term_criterias[1])
            # track jet
            img, track_windows[2] = Cam_Shift_Object_Detection(masked_img, img, (255, 0, 255), "JET", roi_hists[2],
                                                           track_windows[2], True, term_criterias[2])
            # img,track_windows[3]=Cam_Shift_Object_Detection(masked_img,img,(255,0,0),"DEP_MAT",roi_hists[3],track_windows[3],False,term_criterias[3])
            t4 = time.perf_counter()
            # draw graphics and calculate deposition distance from nozzle
            img_det, dist = DistEst(img, dep_x, dep_y, track_windows[0], d_cal_factor)
            dep_dist.append(dist)
            # draw graphics on segmented image also
            img_seg, _ = DistEst(masked_img, dep_x, dep_y, track_windows[0], d_cal_factor)
            t5 = time.perf_counter()
            video_shower.frame = img # change to img_seg if wanna see segmented image
            t6 = time.perf_counter()
            video_writer1.frame = img_det # store video with objects detected
            video_writer2.frame = img_seg # store video with objects segmented
            t7 = time.perf_counter()
            print ('Obj Det: ',t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6)
            k = cv.waitKey(1)
            if k == 'q':  # press 'q' to stop
                video_shower.started = False
                video_writer1.started = False
                video_writer2.started = False
    # save deposition distance in a csv file
    np.savetxt(save_filepath + "Deposition Distance.csv", np.array(dep_dist), delimiter=',')



def Multi_FeatExtr(frames, features, frame0, track_windows,d_cal_factor, sp_cal_factor, stride_x,stride_y):
    """
    Function extracting all features from every frame on the video
    :param frames: queue of images to extract features from
    :param frame0: 1st video frame to calculate features from
    :param track_windows: regions of interest in the image regarding jet position, nozzle position etc
    :param d_cal_factor: calibrating factor of distance
    :param sp_cal_factor: calibrating factor of speed
    :param stride_x: every how many pixels read deposited material thickness
    :param stride_y: every how many pixels read angles, areas and diameters
    """


    # creating lists to store info
    thicknesses = []
    tcdiameters = []
    tcangles_L = []
    tcangles_R = []
    tcareas = []
    tcmovement = []

    jdiameters = []
    jangles_L = []
    jangles_R = []
    jareas = []
    jmovement = []

    totdiameters = []
    totangles_L = []
    totangles_R = []
    totareas = []
    totmovement = []
    # create a dummy image to display so we can safely exit
    dummy=np.zeros((100,500,3),dtype=np.uint8)
    dummy=cv.putText(dummy,'Running...',(5,80),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2,cv.LINE_AA)
    # extracting features from first frame
    thickness, tcdiameter, tcangle_l, tcangle_r, tcarea, tcr_boundary_previous, jdiameter, jangle_l, jangle_r, jarea, jr_boundary_previous, totdiameter, totangle_l, totangle_r, totarea, totr_boundary_previous = Feat_Extr(
        frame0, track_windows, d_cal_factor, stride_x,stride_y)  # extracting features from image for taylor cone, jet, and both of them together
    # storing results for first frame
    thicknesses.append(thickness)
    tcdiameters.append(tcdiameter)
    tcangles_R.append(tcangle_r)
    tcangles_L.append(tcangle_l)
    tcareas.append(np.sum(tcarea))

    jdiameters.append(jdiameter)
    jangles_R.append(jangle_r)
    jangles_L.append(jangle_l)
    jareas.append(jarea)

    totdiameters.append(totdiameter)
    totangles_R.append(totangle_r)
    totangles_L.append(totangle_l)
    totareas.append(totarea)
    video_shower=VideoShower(dummy,name='Features Extraction').start()
    # entering main loop
    while True:
        # break condition, need to terminate threads to
        if not video_shower.started:
            video_shower.stop()
            break
        # if there is available a new frame perform features extraction
        if frames.qsize() > 0:
            t1 = time.perf_counter()
            img = frames.get() # get new frame
            t2 = time.perf_counter()
            # extract features from frame
            thickness, tcdiameter, tcangle_l, tcangle_r, tcarea, tcr_boundary, \
            jdiameter, jangle_l, jangle_r, jarea, jr_boundary, \
            totdiameter, totangle_l, totangle_r, totarea, totr_boundary = Feat_Extr(img, track_windows, d_cal_factor,
                                                                                  stride_x, stride_y)
            t3 = time.perf_counter()
            # append results to store lists
            thicknesses.append(thickness)
            tcdiameters.append(tcdiameter)
            tcangles_R.append(tcangle_r)
            tcangles_L.append(tcangle_l)
            tcareas.append(np.sum(tcarea))
            tcmovement.append((tcr_boundary - tcr_boundary_previous) * sp_cal_factor)
            tcr_boundary_previous = tcr_boundary

            jdiameters.append(jdiameter)
            jangles_R.append(jangle_r)
            jangles_L.append(jangle_l)
            jareas.append(jarea)
            jmovement.append((jr_boundary - jr_boundary_previous) * sp_cal_factor)
            jr_boundary_previous = jr_boundary

            totdiameters.append(totdiameter)
            totangles_R.append(totangle_r)
            totangles_L.append(totangle_l)
            totareas.append(totarea)
            totmovement.append((totr_boundary - totr_boundary_previous) * sp_cal_factor)
            totr_boundary_previous = totr_boundary
            t4 = time.perf_counter()
            print('Feat_Extr: ',t2-t1, t3-t2,t4-t3)
            video_shower.frame=dummy
            k = cv.waitKey(1)
            # if 'q' pressed give order to terminate extra thread and exit while loop
            if k == 'q':
                video_shower.started = False
    # once exited the loop return results to main process to store
    features.put(Features(thicknesses,tcdiameters,tcangles_L,tcangles_R,tcareas,tcmovement,
                          jdiameters,jangles_L,jangles_R,jareas,jmovement,
                          totdiameters,totangles_L,totangles_R,totareas,totmovement))


def main_processing():

    track_windows_init = []
    masks = []

    # create queues to transfer data between processes
    frames_process_1 = Queue(maxsize=1)
    frames_process_2 = Queue(maxsize=1)
    features = Queue(maxsize=1)
    # setting some initial variables

    filename = 'Video S2'
    speed = ' (292.5mm.min)'
    read_filepath = 'C:/Users/thano/Desktop/ComputerVisionMEW_Paper/MEW_Tool/data/' + filename + '/'
    save_filepath = 'C:/Users/thano/Desktop/ComputerVisionMEW_Paper/MEW_Tool/Results/' + filename + speed + '/Parallel/'
    filepath = read_filepath + filename + speed + '.mp4'
    windows = np.loadtxt('PrinterSetUps/' + filename + ' set up.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))
    # assign calibration factors
    if filename == 'Video S1':
        d_cal_factor = 0.0115 * math.pow(10, -3)
    elif filename == 'Video S2':
        d_cal_factor=0.0170 * math.pow(10,-3) # change to 0.0115 for  Video S1 or 0.0170 for Video S2
    fps = 50
    sp_cal_factor = d_cal_factor * fps
    # stride to get features
    stride_x = 30 # for reading deposited material thickness
    stride_y = 10 # for reading Jet diameters etc

    # initialize video reader thread
    video_reader = VideoReader(src=filepath, fps=fps).start()
    frame = video_reader.read() # read first frame
    frame = cv.bitwise_not(frame) # reversing image


    dep_window = (0, int(windows[0, -1]), frame.shape[1], int(windows[1, -1] - windows[0, -1]))
    for i in range(0, 4):
        track_windows_init.append((0, int(windows[0, i]), frame.shape[1], int(windows[1, i] - windows[0, i])))
        masks.append([(0, int(windows[0, i])), (frame.shape[1], int(windows[0, i])), (frame.shape[1], int(windows[1, i])),(0, int(windows[1, i]))])

    # initialize detection
    roi_hists, track_windows, term_criterias = ObjDet_Init(frame, track_windows_init, masks)  # initializing object detection
    video_shower= VideoShower(frame,fps,name='Main Feed').start()
    video_writer=VideoWriter(frame).start(save_filepath+"Video_unprocessed.mp4",fps)
    p1 = Process(target=Multi_FeatExtr, args = (frames_process_1, features, frame, track_windows_init, d_cal_factor, sp_cal_factor, stride_x, stride_y))
    p2 = Process(target=Multi_ObjSegDet, args = (frames_process_2, frame, roi_hists, track_windows, dep_window, term_criterias, masks, d_cal_factor, fps, save_filepath))
    p1.start()
    p2.start()

    while True:
        if not video_reader.started or not video_shower.started or not video_writer.started:
            video_reader.stop()
            video_shower.stop()
            video_writer.stop()
            break
        t1 = time.perf_counter()
        frame = video_reader.read() # read new frame
        t2 = time.perf_counter()
        frame = cv.bitwise_not(frame) # reverse frame
        t3 = time.perf_counter()
        frames_process_1.put(frame) # store new frame to Queue 1 for process 1 to use
        frames_process_2.put(frame) # store new frame to Queue 2 for process 2 to use
        video_shower.frame = frame

    time.sleep(30)

    # after exiting main loop get features from Queue
    feat = features.get()

    # wait before terminating processes so that we get everything right
    p1.terminate()  # terminate process 1
    p2.terminate()  # terminate process 2

    # store results
    store_results(feat, save_filepath)



if __name__ == "__main__":
    main_processing()