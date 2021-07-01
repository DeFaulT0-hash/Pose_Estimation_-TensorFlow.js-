from typing import Counter
import cv2
import numpy as np
import mediapipe as mp
from numpy.core.defchararray import count
from numpy.lib.function_base import angle
from numpy.lib.type_check import imag


mp_draw = mp.solutions.drawing_utils
mp_hs = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(3, 1920)
cap.set(4, 1080)
state_r = None
counter_right = 0
state_l = None
counter_left = 0

print("Zero(0) To Reset Counter..... ")

with mp_pose.Pose(min_detection_confidence=1.0, min_tracking_confidence=1.0)as pose:

    while(cap.isOpened()):
        ret, frame = cap.read()

        flip_image = cv2.flip(frame, 1)
        if(ret == True):

            # recolouring the image to RGB
            img = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False

            # making dection with the pose.process
            result = pose.process(img)
            # print(result.pose_landmarks)

            # recolouring it into BGR(Blue,Green,Red)
            img.flags.writeable = True
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # TRY NO:1 (FOR RIGHT UPPER BODY )
            # grabing the landmarks(inside landmarks variable)
            try:
                landmarks = result.pose_landmarks.landmark
                # print(landmarks)

                # grabing the coordinates of the left side
                for co_ordinates_left in landmarks:
                    shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                    wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                    # print([shoulder_left], [elbow_left], [wrist_left])

                # passing the three arguments as [shoulder, elbow, wrist]
                angle_right = cal_angle(shoulder_left, elbow_left, wrist_left)
                # print("RIGHT HAND ANGLE :", angle_right)

                # rat = tuple(np.multiply(elbow, [1280, 720]).astype(int))
                # print(rat)
                # visualizing or print the angle on the screen
                cv2.putText(img_bgr, str(angle_right), tuple(
                    np.multiply(elbow_left, [1280, 720]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # counter logic right
                if angle_right > 160:
                    state_r = "down"
                if angle_right < 35 and state_r == "down":
                    state_r = "up"
                    counter_right += 1
                    # print("RIGHT HAND COUNT : ", counter_right)

            except:
                pass

            # rendering the box at pixel(0,0)
            cv2.rectangle(img_bgr, (0, 0), (300, 100), (245, 117, 16), -1)

            # processed data of the right elbow
            # rendering the counter variable
            cv2.putText(img_bgr, 'R_Hand_Count', (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, str(counter_right), (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # TRY NO:2 (FOR LEFT UPPER BODY )
            try:
                landmarks = result.pose_landmarks.landmark

                # grabing the right side of the upper body coordinates
                for co_ordinates_right in landmarks:

                    shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    elbow_right = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[
                        mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
                    wrist_right = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[
                        mp_pose.PoseLandmark.RIGHT_WRIST.value].y

                    # print(shoulder_right, elbow_right, wrist_right)

                angle_left = cal_angle(
                    shoulder_right, elbow_right, wrist_right)

                # print("LEFT HAND ANGLE:", angle_left)

                # visualizing or print the angle on the screen
                cv2.putText(img_bgr, str(angle_left), tuple(
                    np.multiply(elbow_right, [1280, 720]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)

                # counter logic left
                if angle_left > 160:
                    state_l = "down"
                if angle_left < 35 and state_l == "down":
                    state_l = "up"
                    counter_left += 1
                    # print("LEFT HAND COUNT : ", counter_left)

            except:
                pass

            # processed data of the right elbow
            cv2.putText(img_bgr, 'L_Hand_Count', (160, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, str(counter_left), (180, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # angle calculation function(where we can pass any 3 arguments)

            def cal_angle(a, b, c):
                a = np.array(a)  # First point
                b = np.array(b)  # mid point
                c = np.array(c)  # last point

                radian = (np.arctan2(c[1]-b[1], c[0]-b[0]) -
                          np.arctan2(a[1]-b[1], a[0]-b[0]))
                real_angle = np.abs(radian*180.0/np.pi)

                if real_angle > 180:
                    real_angle = 360-real_angle

                return real_angle

            if cv2.waitKey(1) == ord('0'):
                counter_left = 0
                counter_right = 0
                stage_right = None
                stage_left = None
                print("COUNTER RESET.....")

            # p = mp_pose.PoseLandmark
            # print(p)
            # for i in mp_pose.PoseLandmark:
                # print(i)

                # print([mp_pose.PoseLandmark.LEFT_WRIST.value])
                # print([mp_pose.PoseLandmark.LEFT_ELBOW.value])

            # for left_elbow in landmarks:
                # print([mp_pose.PoseLandmark.LEFT_ELBOW.value])

            # for left_wrist in landmarks:
                # print([mp_pose.PoseLandmark.LEFT_WRIST.value])

            # rendering it or rather drawing the pose co-ordinates
            mp_draw.draw_landmarks(
                img_bgr, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(circle_radius=2),
                mp_draw.DrawingSpec(color=(200, 66, 26),
                                    thickness=2, circle_radius=2)
            )

            # print(mp_pose.POSE_CONNECTIONS)

            cv2.imshow('output', img_bgr)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break

cap.release()
cv2.destroyAllWindows()
