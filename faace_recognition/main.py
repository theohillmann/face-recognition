import cv2
import numpy as np
import face_recognition

imgGinobili = face_recognition.load_image_file('ginobili1.jpg')
imgGinobili = cv2.cvtColor(imgGinobili, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ginobili2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgGinobili)[0]
encodeGinobili = face_recognition.face_encodings(imgGinobili)[0]
cv2.rectangle(imgGinobili, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0))

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 0))


#comparando as imagens
results = face_recognition.compare_faces([encodeGinobili], encodeTest)
print(results)

faceDis = face_recognition.face_distance([encodeGinobili], encodeTest)
print(faceDis)

cv2.putText(imgTest, f'{results} {faceDis[0], 2}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

cv2.imshow('ginobili', imgGinobili)
cv2.imshow('ginobili Teste', imgTest)
cv2.waitKey(0)
