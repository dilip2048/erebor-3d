import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

webcam = cv2.VideoCapture(0)
##########################################################################################
while webcam.isOpened():

    success, img = webcam.read()
    img = cv2.flip(img, 1)

    # applying face mesh model using MediaPipe
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                connections = mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None,
                connection_drawing_spec = val_a
            )

            # draw the edges of the face mesh
            #mp_drawing.draw_landmarks(
            #    image=img,
            #    landmark_list=face_landmarks,
            #    connections = mp_face_mesh.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec = val_b
            #)

            # draw the polygons of the face mesh
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec = val_c
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
    img = cv2.putText(img, "Current Expression: ", (20, 440),
                      cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    img = cv2.putText(img, "Current Expression: ", (20, 440),
                      cv2.FONT_HERSHEY_DUPLEX, 0.7, (225, 225, 225), 1, cv2.LINE_AA)
    
    # output video window
    cv2.imshow('Erebor 3D Emotion Recognition System', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()