
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

model = model_from_json(open("model.json",  "r",errors='ignore').read())
model.load_weights('model_weights.h5')

emotions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

		detected_face = cv2.resize(cv2.cvtColor(img[int(y):int(y+h), int(x):int(x+w)], cv2.COLOR_BGR2GRAY), (48, 48))

		img_pixels = np.expand_dims(image.img_to_array(detected_face), axis = 0)
		img_pixels /= 255

		predictions = model.predict(img_pixels)

		cv2.putText(img, emotions[int(np.argmax(predictions))], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
