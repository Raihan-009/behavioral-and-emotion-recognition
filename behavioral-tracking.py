import cv2
import dlib
import numpy as np
from imutils import face_utils
import pygame

from poltmodule import LivePlot
from utils import stackImages

print("Imported Successfully!")

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

cap = cv2.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
faceLandmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

plotY = LivePlot(640,480,(20,50), invert=True)
plotMar = LivePlot(640,480,(10,50), invert=True)
ratioList = []

idList = [36,37,38,39,40,41,49,50,51,52,53,54,55,56,57,58,59,60]

sleepiness = 0
drowsiness = 0
awakeness = 0
activity = ""
color = (0,0,0)
sound = 0

def distance(Px,Py):
    displacement = np.linalg.norm(Px - Py)
    return displacement
    
def blinkingDetection(a,b,c,d,e,f):
    short_distance = distance(b,d) + distance(c,e)
    long_distance = distance(a,f)
    ratio = short_distance / (2.0*long_distance)

    if (ratio > 0.22):
        return 2, ratio
    elif (ratio > 0.18 and ratio < 0.22):
        return 1, ratio
    else:
        return 0, ratio

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    mouth_aspect_distance = abs(top_mean[1] - low_mean[1])
    #print(mouth_aspect_distance)
    if (mouth_aspect_distance > 20):
        return 3, mouth_aspect_distance
    else:
        return 4, mouth_aspect_distance

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_detector(gray)
    # print(type(face))
    # print(len(faces))
    if face:
        face = face[0]
        # for face in faces:
        #     x1 = face.left()
        #     y1 = face.top()
        #     x2 = face.right()
        #     y2 = face.bottom()

        landmarks = faceLandmarks(gray,face)
        for n in idList:
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(frame,(x,y), 2,(0,0,255), 4)

        landmarks = face_utils.shape_to_np(landmarks)
        # print(landmarks)
        
        left_eye, ratio = blinkingDetection(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
        #print(left_eye)
        right_eye = blinkingDetection(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])
        #print(right_eye)
        
        # print(ratio*100)
        ratio = int((ratio*100)+10)
        imgPlot = plotY.update(ratio)
        
        

        mar, marr = lip_distance(landmarks)
        # print(int(marr))
        
        marPlot = plotMar.update(marr)

        if (left_eye == 0 or right_eye == 0 or mar == 0):
            sleepiness += 1
            drowsiness = 0
            awakeness = 0
            if (sleepiness>6):
                activity = "Alert! Are you sleeping?"
                color = (0,0,255)
                # Play sound
                # alert_sound.play()
        elif (left_eye == 1 or right_eye ==1 or mar == 3):
            sleepiness = 0
            drowsiness +=1 
            awakeness = 0
            if (drowsiness>6):
                activity = "Hushh! You looks sleepy!"
                color = (0,0,0)
                # Play sound
                # alert_sound.play()
        elif (left_eye == 2 or right_eye ==2 and mar == 4):
            drowsiness =0
            sleepiness =0
            awakeness +=1
            if (awakeness>6):
                activity = "Looking Forward!"
                color = (0,255,0)
        elif (left_eye == 2 or right_eye ==2 and mar == 3):
            drowsiness =0
            sleepiness +=1
            awakeness  =0
            if (sleepiness>1):
                activity = "Hushh! You looks sleepy!"
                color = (0,255,0)
                # Play sound
                # alert_sound.play()
        
    else:
        # print("Error!")
        activity = "Driver Distracted!"
        color = (0,0,0)
        # Play sound
        # alert_sound.play()

    cv2.rectangle(frame, (10,5),(445,40), (255,255,255), -1 )
    cv2.putText(frame, activity, (15,30), cv2.FONT_HERSHEY_COMPLEX, 1, color ,2 )

    img = cv2.resize(frame, (640, 480))
    imgstack = stackImages([img, imgPlot], 2, 1)
    imgstack2 = stackImages([img, marPlot], 2, 1)
    cv2.imshow('Drowsiness Monitoring', imgstack)
    cv2.imshow('Yawnning Monitoring', imgstack2)
    
    
    
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
