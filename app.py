from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
from imutils import face_utils
import pygame
from deepface import DeepFace
from ultralytics import YOLO


app = Flask(__name__)

print("Imported Successfully!")

# Initialize pygame
pygame.mixer.init()

# Load the sound file
alert_sound = pygame.mixer.Sound("alert.wav")

face_detector = dlib.get_frontal_face_detector()
faceLandmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier('haar_face.xml')
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayimg, 1.1, 5)
    return faces

def draw_emotion_text(frame, face, emotion):
    a, b, c, d = face
    font_scale = 2.0  
    text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    text_position = (a + c // 2 - text_size[0] // 2, b + d + 50)  
    cv2.rectangle(frame, (text_position[0], text_position[1] - text_size[1]), 
                  (text_position[0] + text_size[0], text_position[1]), (0, 0, 0), -1)
    cv2.putText(frame, emotion, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

def play_alert_sound(emotion):
    global sad_duration, angry_duration
    
    if emotion == "sad":
        sad_duration += 1
        angry_duration = 0
    elif emotion == "angry":
        angry_duration += 1
        sad_duration = 0
    else:
        sad_duration = 0
        angry_duration = 0

    if sad_duration >= 5 or angry_duration >= 5:
        pygame.mixer.init()
        alert_sound = pygame.mixer.Sound("alert.wav")
        alert_sound.play()



def distance(Px, Py):
    displacement = np.linalg.norm(Px - Py)
    return displacement

def blinkingDetection(a, b, c, d, e, f):
    short_distance = distance(b, d) + distance(c, e)
    long_distance = distance(a, f)
    ratio = short_distance / (2.0 * long_distance)

    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

def lip_distance(shape):
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

def generate_frames_drowsiness():
    sleepiness = 0
    drowsiness = 0
    awakeness = 0
    activity = ""
    color = (0,0,0)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray)
        
        
        if len(faces) > 0:
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                landmarks = faceLandmarks(gray,face)
                #print(landmarks)
                landmarks = face_utils.shape_to_np(landmarks)
                #print(landmarks)
                
                left_eye = blinkingDetection(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
                #print(left_eye)
                right_eye = blinkingDetection(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])
                #print(right_eye)
                mar = lip_distance(landmarks)

                if (left_eye == 0 or right_eye == 0 or mar == 0):
                    sleepiness += 1
                    drowsiness = 0
                    awakeness = 0
                    if (sleepiness>6):
                        activity = "Alert! Are you sleeping?"
                        color = (0,0,255)
                        # Play sound
                        alert_sound.play()
                elif (left_eye == 1 or right_eye ==1 or mar == 3):
                    sleepiness = 0
                    drowsiness +=1 
                    awakeness = 0
                    if (drowsiness>6):
                        activity = "Hushh! You looks sleepy!"
                        color = (0,0,0)
                        # Play sound
                        alert_sound.play()
                elif (left_eye == 2 or right_eye ==2 and mar == 4):
                    drowsiness =0
                    sleepiness =0
                    awakeness +=1
                    if (awakeness>6):
                        activity = "Having a safe driving!"
                        color = (0,255,0)
                elif (left_eye == 2 or right_eye ==2 and mar == 3):
                    drowsiness =0
                    sleepiness +=1
                    awakeness  =0
                    if (sleepiness>1):
                        activity = "Hushh! You looks sleepy!"
                        color = (0,255,0)
                        # Play sound
                        alert_sound.play()
            
        else:
            print("Error!")
            activity = "Driver Distracted!"
            color = (0,0,0)
            # Play sound
            alert_sound.play()

        cv2.rectangle(frame, (10,5),(445,40), (255,255,255), -1 )
        cv2.putText(frame, activity, (15,30), cv2.FONT_HERSHEY_COMPLEX, 1, color ,2 )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def generate_frames_emotion():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open webcam")
    
    while True:
        ret, frame = cam.read()
        res = DeepFace.analyze(frame, actions=("emotion"), enforce_detection=False)
        faces = detect_faces(frame)
        
        for face in faces:
            cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255,0,0), 3)
            emotion = res[0]['dominant_emotion']
            draw_emotion_text(frame, face, emotion)
            play_alert_sound(emotion)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()


def generate_frames_classification():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8m.pt")

    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound("alert.wav")

    text_classes = []
    with open('classes.txt', "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            text_classes.append(class_name)
            
    if not cap.isOpened():
        print("Error: Couldn't open camera")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        results = model(frame, device="mps")
        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        for cls, bbox in zip(classes, bboxes):
            if cls == 67:
                (x, y, x2, y2) = bbox
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
                cv2.putText(frame, text_classes[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
                alert_sound.play()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
  

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/behavioral')
def behavioral():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion')
def emotion():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classification')
def classification():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames_classification(), mimetype='multipart/x-mixed-replace; boundary=frame')




  
    
if __name__ == '__main__':
    app.run(debug=True)
