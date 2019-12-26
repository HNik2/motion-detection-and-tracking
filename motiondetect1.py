# import the necessary packages

import datetime
import imutils
import numpy as np
from pykalman import KalmanFilter
from matplotlib import pyplot as plt
import cv2

# read the video file
vs = cv2.VideoCapture("/home/nikam/Bureau/data/Walk1.mpg")

# some matrix for kalman filter
Transition_Matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
Observation_Matrix=[[1,0,0,0],[0,1,0,0]]
# initialize the first frame in the video stream
firstFrame = None
# Matrix to store the objects position
measured = []

pts = []
# loop over the frames of the video
while True:
    # grab the current frame and initialize

    frame = vs.read()
    frame = frame[1]


    #to count the number of mooving object
    count = 0
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    #cv2.line(frame, (2, 206), (498, 206), (0, 0, 255), 4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 22, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 200:
            continue

        centers = None
        count += 1
        #to get objects center
        ((cx, cy), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        centers = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        radius = int(radius)
        # to draw the object centroids
        cv2.circle(frame, centers, 3, (0, 0, 255), -1)
        # compute the bounding box for the contour, draw it on the frame,
        # and update the number of mooving objects
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.plot(centers[0], centers[1], 'ob')


        pts.append(centers)
    # draw the number of mooving object and timestamp on the frame
    cv2.putText(frame, "Nombre de personnes : {}".format(count), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("gray blur", gray)
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(60) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.release()
# create numpy array to store the coordinates of the centres of the objects
a = np.asarray(pts)
print(pts)
print(a)
# save the trajectory of the mooving objects
np.save("Trajectory", a)
# load the trajectory
Measured=np.load("Trajectory.npy")

# for kalman tracker
xinit=Measured[0,0]
yinit=Measured[0,1]
vxinit=Measured[1,0]-Measured[0,0]
vyinit=Measured[1,1]-Measured[0,1]
initstate=[xinit,yinit,vxinit,vyinit]
initcovariance=1.0e-3*np.eye(4)
transistionCov=1.0e-4*np.eye(4)
observationCov=1.0e-1*np.eye(2)
kf=KalmanFilter(transition_matrices=Transition_Matrix,
            observation_matrices =Observation_Matrix,
            initial_state_mean=initstate,
            initial_state_covariance=initcovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

(filtered_state_means, filtered_state_covariances) = kf.filter(Measured)
plt.plot(Measured[:,0],Measured[:,1],'xr',label='measured')
plt.axis([0,520,360,0])

plt.plot(filtered_state_means[:,0],filtered_state_means[:,1],'ob',label='kalman output')
plt.legend(loc=2)
plt.title("Constant Velocity Kalman Filter")
plt.show()

cv2.destroyAllWindows()
