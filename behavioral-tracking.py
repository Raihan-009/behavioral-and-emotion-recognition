import cv2
import dlib
import numpy as np
from imutils import face_utils
import pygame

print("Imported Successfully!")

nose_length_threshold = 2.05

def initialize_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def calculate_distance(Px, Py):
    displacement = np.linalg.norm(Px - Py)
    return displacement

def detect_blinking(a, b, c, d, e, f):
    short_distance = calculate_distance(b, d) + calculate_distance(c, e)
    long_distance = calculate_distance(a, f)
    ratio = short_distance / (2.0 * long_distance)

    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

def detect_lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    mouth_aspect_distance = abs(top_mean[1] - low_mean[1])

    if mouth_aspect_distance > 20:
        return 3
    else:
        return 4
    
def detect_nose_length_ratio(shape):
    # Points for nose length calculation
    nose_start = shape[27]  # Point 28
    nose_end = shape[30]  # Point 31

    # Points for other facial feature (for example, distance between eyes)
    other_feature_start = shape[32]  # Point 33
    other_feature_end = shape[35]  # Point 36

    # Calculate distances
    nose_length = calculate_distance(nose_start, nose_end)
    other_feature_length = calculate_distance(other_feature_start, other_feature_end)

    # Calculate ratio
    nose_length_ratio = nose_length / other_feature_length

    return nose_length_ratio

# Add this function to the code before the while loop


def detect_face_features(face, gray, faceLandmarks):
    landmarks = faceLandmarks(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)
    return landmarks

def check_activity(left_eye, right_eye, mar):
    global sleepiness, drowsiness, awakeness, activity, color

    if left_eye == 0 or right_eye == 0 or mar == 0:
        sleepiness += 1
        drowsiness = 0
        awakeness = 0
        if sleepiness > 6:
            activity = "Alert! Are you sleeping?"
            color = (0, 0, 255)
            if sound_enabled:
                play_alert_sound()
    elif left_eye == 1 or right_eye == 1 or mar == 3:
        sleepiness = 0
        drowsiness += 1
        awakeness = 0
        if drowsiness > 6:
            activity = "Hushh! You look sleepy!"
            color = (0, 0, 0)
            if sound_enabled:
                play_alert_sound()
    elif left_eye == 2 or right_eye == 2 and mar == 4:
        drowsiness = 0
        sleepiness = 0
        awakeness += 1
        if awakeness > 6:
            activity = "Having a safe driving!"
            color = (0, 255, 0)
    elif left_eye == 2 or right_eye == 2 and mar == 3:
        drowsiness = 0
        sleepiness += 1
        awakeness = 0
        if sleepiness > 1:
            activity = "Hushh! You look sleepy!"
            color = (0, 255, 0)
            
    nose_length_ratio = detect_nose_length_ratio(landmarks)
    # print('nose_length_ratio', nose_length_ratio)
    if nose_length_ratio > nose_length_threshold:
        activity = "Hushh! Be alert!!"
        color = (0, 0, 0)
        if sound_enabled:
            play_alert_sound()

def draw_activity_text(frame, activity, color):
    cv2.rectangle(frame, (10, 5), (445, 40), (255, 255, 255), -1)
    cv2.putText(frame, activity, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

def play_alert_sound():
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound("alert.wav")
    alert_sound.play()

cap = initialize_camera()

face_detector = dlib.get_frontal_face_detector()
faceLandmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleepiness = 0
drowsiness = 0
awakeness = 0
activity = ""
color = (0, 0, 0)

pygame.mixer.init()  # Initialize pygame mixer
sound_enabled = True  # Set to False if you want to disable sound

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    if len(faces) > 0:
        for face in faces:
            landmarks = detect_face_features(face, gray, faceLandmarks)
            left_eye = detect_blinking(landmarks[36], landmarks[37], landmarks[38],
                                        landmarks[41], landmarks[40], landmarks[39])
            right_eye = detect_blinking(landmarks[42], landmarks[43], landmarks[44],
                                         landmarks[47], landmarks[46], landmarks[45])
            mar = detect_lip_distance(landmarks)

            check_activity(left_eye, right_eye, mar)

    else:
        activity = "No Driver!"
        color = (0, 0, 0)

    draw_activity_text(frame, activity, color)

    cv2.imshow("Hola!", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
