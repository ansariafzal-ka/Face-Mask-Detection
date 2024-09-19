import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('face_mask_detector.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (75, 75))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0
    return face_img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = gray_frame[y:y+h, x:x+w]
        face = preprocess_face(face)

        prediction = model.predict(face)
        mask_label = "No Mask" if prediction[0] > 0.5 else "Mask"

        label = f"{mask_label}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
