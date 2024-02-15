import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import aruco
import os

def detectAruco(frame,mtx,dist):
    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_4X4_1000 )

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10
    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, markerlgth, Left_mtx, Left_dist)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        print ("ids",ids)
        print ("rvec: {0}, tvec: {1}".format(rvec, tvec))
        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        return (rvec,tvec,frame)
    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(Left_frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        return (None,None,frame)
    
cap = cv2.VideoCapture(6)


while not cap.isOpened():
    print("trying to open camera")

count = 0
Left_dist =np.array([[-0.15642446 ,-0.00772512,  0.0006035,   0.00085093 , 0.02146668]]) #Computed from calibration charuko.py
Left_mtx = np.array([[704.45209545 ,  0.  ,       657.8833007 ],
 [  0.       ,  707.34043675 ,363.7955402 ],
 [  0.        ,   0.      ,     1.        ]])


Right_dist =np.array([[-0.16113003,  0.00030797,  0.00048842,  0.00089638 , 0.0177444 ]]) #Computed from calibration charuko.py
Right_mtx = np.array([[702.03209176  , 0.   ,      632.58983176],
 [  0.      ,   704.52835998,371.48590454],
 [  0.        ,   0.       ,    1.        ]])




markerlgth = 0.026
while (True):
    Left_ret,frame = cap.read()
    
    print(frame.shape)
    Left_frame = frame[:,:int(frame.shape[1]/2)]
    Left_frame=cv2.resize(Left_frame, (2560//2,720), interpolation = cv2.INTER_AREA)

    rvec,tvec,Left_frame=detectAruco(Left_frame,Left_mtx,Left_dist)

    Right_frame = frame[:,int(frame.shape[1]/2):]
    Right_frame=cv2.resize(Right_frame, (2560//2,720), interpolation = cv2.INTER_AREA)

    rvec,tvec,Right_frame=detectAruco(Right_frame,Right_mtx,Right_dist)

    # operations on the Left_frame
    
    # lists of ids and the corners belonging to each id
    
    # display the resulting Left_frame

    # Left_resized = cv2.resize(Left_frame, (1344//2,720), interpolation = cv2.INTER_AREA)
    # Right_resized = cv2.resize(Right_frame, (1344//2,720), interpolation = cv2.INTER_AREA)

    cv2.imshow('Left_frame',Left_frame)
    # cv2.imshow('Right_frame',Right_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
