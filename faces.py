from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
set_session(sess)
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('../erebor/model/EmotionDetectionModel.h5')

class_labels=['Angry','Happy','Neutral','Sad','Surprise']
class DetectEmotion(object):
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def get_frame(self):
        ret,frame=self.cap.read()
        """Reads the camera frame until someone pressed Q key"""
        success, img = self.cap.read()

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
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            """Stores values of rectangle on top of the face and resize it to 48 by 48"""
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            """If face is detected, convert the values to 0/1 which is easier for model to predict the emotion"""
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)

                """Display the predicted emotion on the side of the face"""
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(img, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # emotion detection code ends here

        ##########################################################################################
        # text: output inage, text, position, font, scale, color, thickness, parameter
        img = cv2.putText(img, "Press the 'Q' key to exit the program", (20, 20),
                          cv2.FONT_HERSHEY_DUPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
        img = cv2.putText(img, "(The X window button won't work...)", (20, 40),
                          cv2.FONT_HERSHEY_DUPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
        success, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
