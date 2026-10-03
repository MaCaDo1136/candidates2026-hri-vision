import cv2
import time
import insightface
from insightface.app import FaceAnalysis
import numpy as np

cap = cv2.VideoCapture(0)
time.sleep(1)

for _ in range(10):
    cap.read()

ret, frame = cap.read()
if ret:
    cv2.imwrite('mi_foto.jpg', frame)
    print('Guardada')
else:
    print('No se pudo leer la cámara')
cap.release()

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))

img = cv2.imread('mi_foto.jpg')
faces = app.get(img)
print(f'Caras encontradas: {len(faces)}')
print(f'Tamaño del embedding: {faces[0].embedding.shape}')

np.save('mario_embedding.npy', faces[0].embedding)
