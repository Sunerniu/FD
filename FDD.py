from imutils import face_utils
import time
import dlib
from cv2 import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
import argparse

def run(name):
    flag = 0
    # initialize dlib's face detector (HOG-based) and then create the
    # facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

    # initialize the video stream and sleep for a bit, allowing the
    # camera sensor to warm up
    print("opening video stream...")
    vs = cv2.VideoCapture(name)
    # vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
    time.sleep(2.0)
    # loop over the frames from the video stream
    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        (359, 391),  # Nose tip 34
        (399, 561),  # Chin 9
        (337, 297),  # Left eye left corner 37
        (513, 301),  # Right eye right corne 46
        (345, 465),  # Left Mouth corner 49
        (453, 469)  # Right mouth corner 55
    ], dtype="double")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    EYE_AR_THRESH = 0.20
    MOUTH_AR_THRESH = 0.85
    EYE_AR_CONSEC_FRAMES = 15
    COUNTER = 0

    # grab the indexes of the facial landmarks for the mouth
    (mStart, mEnd) = (49, 68)
    while True:
        (grabbed, frame) = vs.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape
        # detect faces in the grayscale frame
        rects = detector(gray, 0)


        # loop over the face detections
        for rect in rects:
            # compute the bounding box of the face and draw it on the
            # frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            mouth = shape[mStart:mEnd]


            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # if the eyes were closed for a sufficient number of times
                # then show the warning
                if COUNTER == EYE_AR_CONSEC_FRAMES:
                    flag += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
            else:
                COUNTER = 0

            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            mouthHull = cv2.convexHull(mouth)

            if mar >= MOUTH_AR_THRESH and ear <= EYE_AR_THRESH:
                flag += 1
    if flag > 3:
        print("Yes")
    else :
        print("No")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str,
                default="",
                help="path video for testing")
args = vars(ap.parse_args())
run(args["path"])