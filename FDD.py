from imutils import face_utils
import time
import dlib
from cv2 import cv2
import argparse
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio


def run(name):
    print("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    # 使用dlib的库里带的脸部标志检测器1来获取脸部的68个点
    predictor = dlib.shape_predictor(
        './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

    # 使用cv2来打开视频文件,并且进行判断,若没有正确打开文件退出
    vs = cv2.VideoCapture(name)
    if vs.isOpened():
        print("Video file opened successfully.")
    else:
        print("Failed to open video file.")
        return 0

    time.sleep(2.0)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    eye_ar_thresh = 0.20
    # ear一段时间的判断阙值
    eye_ar_thresh_all = 0.30
    # ear总体时间的判断阙值
    mouth_ar_thresh = 0.85
    # mar的判断阙值
    counter = 0

    eye_tired_nums = 0
    # 整个过程眼睛的不正常帧数
    frame_nums = 0
    # 视频总帧数
    eye_close = 0
    # 不正常闭眼数
    yaw_nums = 0
    # 深哈欠检测

    (mStart, mEnd) = (49, 68)
    while True:
        frame_nums += 1
        (grabbed, frame) = vs.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 对图像进行灰度处理
        num_frames = vs.get(cv2.CAP_PROP_FPS)
        eye_ar_consed_frames = int(num_frames / 2) + 1
        # 视频的每秒帧数来作为不正常闭眼的判断依据

        rects = detector(gray, 0)

        for rect in rects:
            # 对于每一帧进行处理

            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            shape = predictor(gray, rect)

            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # 获取ear和mar的判断数据

            ear = (leftEAR + rightEAR) / 2.0
            # 计算ear
            mouth = shape[mStart:mEnd]
            if ear < eye_ar_thresh_all:
                eye_tired_nums += 1
            if ear < eye_ar_thresh:
                counter += 1
                if counter == eye_ar_consed_frames:
                    eye_close += 1
            else:
                counter = 0
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            # 计算mar
            if mar >= mouth_ar_thresh and ear <= eye_ar_thresh:
                yaw_nums += 1
    if yaw_nums + eye_close > 3:
        print("Yes")
        return 1
    elif eye_close >= 3:
        print("Yes")
        return 1
    elif eye_tired_nums >= frame_nums * 0.70:
        print("Yes")
        return 1
    else:
        print("No")
        return 0


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str,
                default="",
                help="path video for testing")
args = vars(ap.parse_args())
run(args["path"])
