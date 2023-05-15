import cv2
import face_recognition

# Load the images
imgModi = face_recognition.load_image_file('Images_Attendance/modi-image-for-InUth.jpeg')
imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('Images_Attendance/modi.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detect face locations and encodings
faceLocModi = face_recognition.face_locations(imgModi)[0]
encodeModi = face_recognition.face_encodings(imgModi)[0]

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

# Draw rectangles around the faces
cv2.rectangle(imgModi, (faceLocModi[3], faceLocModi[0]), (faceLocModi[1], faceLocModi[2]), (155, 0, 255), 2)
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (155, 0, 255), 2)

# Display the images
cv2.imshow("Modi", imgModi)
cv2.imshow("Test Image", imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compare faces and calculate distance
results = face_recognition.compare_faces([encodeModi], encodeTest)
faceDis = face_recognition.face_distance([encodeModi], encodeTest)
print(results, faceDis)

# Draw text with results on the test image
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

# Display the test image with results
cv2.imshow('narendra-modi', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
