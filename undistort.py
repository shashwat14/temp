# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:16:06 2016

@author: therumsticks
"""

import numpy as np
import cv2

class Camera(object):
    
    def __init__(self,calibration_file, width, height):
        self.mtx, self.dist = self.getCalibrationMatrix(calibration_file)
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(width,height),1,(width,height))
        self.x,self.y,self.w,self.h = roi
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.tracks = []
        self.track_len = 10
        self.detect_interval = 1
        self.frame_idx = 0
    
    def getCalibrationMatrix(self,calibration_file):
        mats = np.load(calibration_file)
        mtx = mats['arr_0']
        dist = mats['arr_1']
        return mtx, dist
    
    def undistort(self, img):
        img =  cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        img = img[self.y:self.y+self.h,self.x:self.x+self.w]
        return img
        
    def fetch(self):
        ret, frame = self.cap.read()
        if ret:
            return True, self.undistort(frame)
        else:
            return None, None
    def getFeatures(self, gray, detector_type):
        if detector_type == "sift":
            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray,None)
            return kp
        elif detector_type == "fast":
            features = []
            fast = cv2.FastFeatureDetector_create()
            kp = fast.detect(img, None)
            for k in kp:
                ptx,pty = (int(x) for x in k.pt)
                features.append((ptx,pty))
            return np.array(features)
        elif detector_type == "good":
            feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )            
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            return p0
            
    def track(self, old_gray, gray, p0):
        lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        return good_old, good_new
        
        
    def trackFeatures(self, old_gray, gray):
        lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict( maxCorners = 500, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
        
        if len(self.tracks) > 0:
            
            img0, img1 = old_gray, gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
            self.tracks = new_tracks

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(gray, mask = mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])
        
        return self.tracks
    
    def getPose(self):
        firstP = []
        secondP = []
        focal = np.mean(self.newcameramtx[0][0] + self.newcameramtx[1][1])
        pp = (self.newcameramtx[0][2],self.newcameramtx[1][2])
        for tr in self.tracks:
            if len(tr) == 1:
                firstP.append(tr[0])
                secondP.append(tr[-1])
        if len(firstP) > 5:
            E, mask = cv2.findEssentialMat(np.matrix(firstP), np.matrix(secondP), focal, pp, cv2.RANSAC, 0.999, 1.0)
            inliers, R, t, mask = cv2.recoverPose(E, np.matrix(firstP), np.matrix(secondP))
            return R,t
        else:
            return None,None
        
        
        
h, w = (480, 640)

camera = Camera("camparam2.npz", 640, 480)

ret, img = camera.fetch()
old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = 0
y = 0

while 1:
    ret, img = camera.fetch()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        camera.trackFeatures(old_gray, gray)
        R,t = camera.getPose()
        cv2.polylines(img, [np.int32(tr) for tr in camera.tracks], False, (0,255,0))
        cv2.imshow("Frame", img)
        ch = cv2.waitKey(1)
        if ch == ord('q'):
            break
camera.cap.release()
cv2.destroyAllWindows()











