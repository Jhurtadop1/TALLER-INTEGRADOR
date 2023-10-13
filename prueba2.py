import cv2
import os
import numpy as np

# Inicializar el clasificador para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Cargar la base de datos de caras de entrenamiento
dir_faces = 'att_faces/orl_faces/JOSUE'
images, labels = [], []

for filename in os.listdir(dir_faces):
    path = os.path.join(dir_faces, filename)
    img = cv2.imread(path, 0)
    images.append(img)
    labels.append(0)  # Asigna una etiqueta a las caras de esta persona

# Crear un modelo de reconocimiento facial
model = cv2.face_LBPHFaceRecognizer.create()
model.train(images, np.array(labels))

# Lista de invitados
lista_invitados = {'JOSUE': 'Bienvenido JOSUE'}  # Agrega los nombres de las personas permitidas

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, confidence = model.predict(face)
        if confidence < 100:
            name = 'Persona ' + str(label)
            if name in lista_invitados:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, lista_invitados[name], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, 'Desconocido', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()