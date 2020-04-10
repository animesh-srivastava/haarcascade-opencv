# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:45:35 2020

@author: anims
"""
import cv2

#%%
'''
various xml files can be found at OpenCV github pages, these have been downloaded and stored from their github.
visit https://github.com/opencv/opencv/tree/master/data/haarcascades for more
'''
face_c = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_c = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_c = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(grey, pic):
    faces = face_c.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(pic,(x,y),(x+w,y+h),(80,160,227),3)
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = pic[y:y+h,x:x+w]
        smile = smile_c.detectMultiScale(roi_grey,1.1,3)
        for (xs,ys,ws,hs) in smile:
            cv2.rectangle(roi_color,(xs,ys),(xs+ws,ys+hs),(32,232,35),2)
    return pic

video_capture = cv2.VideoCapture(0) #0 for internal webcam, 1 for external webcam

while True:
    _, pic = video_capture.read()
    grey = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    canvas = detect(grey,pic)
    cv2.imshow("Video Output",canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()