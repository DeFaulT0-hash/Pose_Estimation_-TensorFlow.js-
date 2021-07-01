from operator import truediv
from typing import Counter
import cv2 as cv
import face_recognition
import numpy as np
from numpy.core.defchararray import count
from numpy.lib.function_base import append


cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 10)
cap.set(3, 1920)
cap.set(4, 1080)

img_comp = face_recognition.load_image_file('./known/VKRU1112.JPG')
img_comp_enc = face_recognition.face_encodings(img_comp)[0]

known_faces = [img_comp_enc]

face_name = ["Nilanjan"]
img_locations = []
img_encodings = []
img_name = []

pros_img = True
counter = 0
pointer = False

while True:
    # reading the cam feed
    r, fram = cap.read()
    fliped_frame = cv.flip(fram, 1)

    s_frame = cv.resize(fliped_frame, (0, 0), fx=0.25, fy=0.25)

    rgb = cv.cvtColor(s_frame, cv.COLOR_BGR2RGB)

    if pros_img:
        img_locations = face_recognition.face_locations(rgb)
        img_encodings = face_recognition.face_encodings(rgb, img_locations)

        def duplicat(final):

            n = len(final)
            temp = []*n
            pivot = 0
            final_length = 0

            for last_occure in range(0, n-1):
                if final[last_occure] != final[last_occure+1]:
                    temp[pivot] = final[last_occure]
                    pivot += 1
                temp[pivot] = final[n-1]
            final_length = len(temp)
            return final_length+1

        final_name = []
        for img_encoding in img_encodings:
            img_compare = face_recognition.compare_faces(
                known_faces, img_encoding)
            name = "Unknown"

            dist = face_recognition.face_distance(known_faces, img_encoding)
            best_match = np.argmin(dist)

            if img_compare[best_match]:
                name = face_name[best_match]
                # print(name)

            final_name.append(name)
            if final_name[0] == face_name[0]:
                pointer = True
            else:
                pointer = False
    pros_img = not pros_img

    if pointer == True:
        val = duplicat(final_name)
        print(val)

    # Display the results
    for (top, right, bottom, left), name in zip(img_locations, final_name):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv.rectangle(fliped_frame, (left, top),
                     (right, bottom), (200, 66, 26), 2)

        # Draw a label with a name below the face
        cv.rectangle(fliped_frame, (left, bottom - 35),
                     (right, bottom), (200, 66, 26), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(fliped_frame, name, (left + 6, bottom - 6),
                   font, 1.0, (255, 255, 255), 1)

    cv.imshow('defcam', fliped_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
