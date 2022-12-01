"""
We have developed an application to capture facial movement and apply it to 3D mesh.
Along with it, we have extended the application to recognize human facial emotions
to determine whether the person is happy, sad, angry, and so on. We have made use of
OpenCV library for image processing and the Keras library for deep learning.

References:
    https://www.youtube.com/watch?v=G1Uhs6NVi-M&ab_channel=edureka%21
    https://medium.com/swlh/emotion-detection-using-opencv-and-keras-771260bbd7f7

"""

import cv2
import mediapipe as mp
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

"""
The "haarcascade_frontalface_default" classifier detects the front face of a person in an image
"""
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('../erebor/model/EmotionDetectionModel.h5')

"""These are the types of emotion that we are going to detect"""
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

webcam = cv2.VideoCapture(0)

while webcam.isOpened():

    """Reads the camera frame until someone pressed Q key"""
    success, img = webcam.read()

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

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (20, 650)

            """Display the predicted emotion on the side of the face"""
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # cv2.putText(img, "Current Expression: " + label, (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 1, cv2.LINE_AA)
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

            # draw the edges of the face mesh
            # mp_drawing.draw_landmarks(
            #    image=img,
            #    landmark_list=face_landmarks,
            #    connections = mp_face_mesh.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec = val_b
            # )

            # draw the polygons of the face mesh
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=val_c
            )

            with open('framedata.txt', 'w') as f:
                f.write(str(face_landmarks))
                f.write('----------')
            f.close()

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
    cv2.imshow('Erebor 3D Emotion Recognition System', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

"""Closes the window and releases the camera frame"""
webcam.release()
cv2.destroyAllWindows()
