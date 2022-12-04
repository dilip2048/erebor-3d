from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
import mediapipe as mp

# from keras.preprocessing import image
from keras.models import load_model
from keras_preprocessing.image import img_to_array

# load model
model = model_from_json(open("EmotionDetectionModel.json", "r").read())

# load weights
model.load_weights('EmotionDetectionModel.h5')

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = load_model('EmotionDetectionModel.h5')

class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, img = camera.read()
        if not success:
            break
        else:
            """Flips the image horizontally. Converts mirror image to normal"""
            img = cv2.flip(img, 1)

            """Applying face mesh model using MediaPipe"""
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Emotion detection code starts from here
            labels = []
            """Changed the color of the image to grey, because we have trained model on grey images"""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            """
            detect the faces in the image and returns a list containing the coordinates of the rectangle 
            around the face. It captures the images of all the people appearing on the camera.
            """
            all_human_faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            """Looping over all the detected faces on the camera"""
            for (x, y, w, h) in all_human_faces:

                """Creates a rectangle on top of the face"""
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                """Stores values of rectangle on top of the face and resize it to 48 by 48"""
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                """If face is detected, convert the values to 0/1 which is easier for model to predict the emotion"""
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    preds = model.predict(roi)[0]
                    print(preds.argmax())
                    label = class_labels[preds.argmax()]
                    label_position = (20, 440)

                    cv2.rectangle(img, (10, 450), (300, 390), (232, 52, 235), 2)

                    """Display the predicted emotion on the side of the face"""
                    cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_TRIPLEX, 2, (52, 217, 235), 3)

                    # cv2.putText(img, "Current Expression: " + label, (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (225,
                    # 225, 225), 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                # emotion detection code ends here

            results = mp_face_mesh.FaceMesh(refine_landmarks=True).process(img)

            # draw annotations on the image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    val_a = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    val_b = mp_drawing_styles.get_default_face_mesh_contours_style()
                    val_c = mp_drawing_styles.get_default_face_mesh_tesselation_style()

                    # draw the irises
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=val_a
                    )

                    # draw the polygons of the face mesh
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=val_c
                    )

            ##########################################################################################
            # text: output inage, text, position, font, scale, color, thickness, parameter
            img = cv2.putText(img, "Press the 'Q' key to exit the program", (20, 20),
                              cv2.FONT_HERSHEY_DUPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
            img = cv2.putText(img, "(The X window button won't work...)", (20, 40),
                              cv2.FONT_HERSHEY_DUPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
            # img = cv2.putText(img, "Current Expression: ", (20, 440),
            #                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            # img = cv2.putText(img, "Current Expression: ", (20, 440),
            #                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (225, 225, 225), 1, cv2.LINE_AA)

            # output video window
            # cv2.imshow('Erebor 3D Emotion Recognition System', img)
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break

            resized_img = cv2.resize(img, (1000, 700))

            ret, buffer = cv2.imencode('.jpg', img)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
