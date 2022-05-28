import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


path = 'Images'
images = []
Names = []
myList = os.listdir(path)
print(myList)
for image in myList:
	currImg = cv2.imread(f'{path}/{image}')
	#adding images to the list
	images.append(currImg)
	#getting names of the images
	Names.append(os.path.splitext(image)[0])
print(Names)

#function to find encodings
def findEncodings(KnownImages):
	encodedImages = []
	for img in KnownImages:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodedImages.append(encode)
	return encodedImages

#function to mark attendance
def markAttendance(name):
	with open('Attendance.csv','r+') as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtString = now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},{dtString}')
			
	
encodedList = findEncodings(images)
print("Encoding completed")

#capturing the live image
capturedImage = cv2.VideoCapture(0)

while True:
	success, img = capturedImage.read()
	reduced_img = cv2.resize(img,(0,0),None,0.25,0.25) #one fourth of the size
	reduced_img = cv2.cvtColor(reduced_img , cv2.COLOR_BGR2RGB)
	
	#to identify multiple locations if multiple faces are present
	facesInCurrFrame = face_recognition.face_locations(reduced_img)
	encode = face_recognition.face_encodings(reduced_img,facesInCurrFrame)
	
	for encodeFace,faceLocation in zip(encode,facesInCurrFrame):
		matches = face_recognition.compare_faces(encodedList,encodeFace)
		faceDistance = face_recognition.face_distance(encodedList,encodeFace)
		print(faceDistance)
		minIndex = np.argmin(faceDistance)
		
		if matches[minIndex]:
			name = Names[minIndex].upper()
			markAttendance(name)
			y1,x2,y2,x1 = faceLocation
			y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
			cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
			cv2.putText(img,name,(x1+6,y2+6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

	cv2.imshow('webcam',img)
	cv2.waitKey(1)

