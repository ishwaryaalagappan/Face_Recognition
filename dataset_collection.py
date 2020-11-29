#This file will do only the data collection process i.e will collect only the images
import cv2, sys, numpy, os 
#haarcascade algorithm will give the inbuilt classification logic
haar_file = 'C:\Ishwarya\calendar\geeks_img\haarcascades\haarcascade_frontalface_default.xml'
#In the file Ishwarya, pictures captured will be saved
main_path="C:\Ishwarya\calendar\geeks_img\datasets\Ishwarya"
(width, height) = (130, 100)	 

#'0' is used for my webcam, 
# if you've any other camera 
# attached use '1' like this 
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 

# The program loops until it has 30 images of the face. 
count = 1
while count < 30: 
	(_, im) = webcam.read() 
    #Since haarcascade algorith uses only gray scale image, the images are converted to gray scale
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    #detectMultiScale is used to plot the face structure
	faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
	for (x, y, w, h) in faces: 
        #rectangle(image,starting point,ending point,color,thickness) will plot the rectangle for face
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = gray[y:y + h, x:x + w] 
        #the image is resized and save in the respective folder
		face_resize = cv2.resize(face, (width, height)) 
		cv2.imwrite('% s/% s.png' % (main_path, count), face_resize) 
	count += 1
	
	cv2.imshow('OpenCV', im) 
	key = cv2.waitKey(10) 
	if key == 27: 
		break
		break
webcam.release()
cv2.destroyAllWindows()