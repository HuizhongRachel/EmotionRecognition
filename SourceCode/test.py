import numpy as np
import cv2



vc = cv2.VideoCapture(0)
cv2.namedWindow("preview")

if vc.isOpened(): # try to get the first frame
  vc.set(3,480)
  vc.set(4,360)

  rval, frame = vc.read()
else:
  rval = False

while True:
  # cv2.imshow("preview", frame)
  rval, frame = vc.read()
  

  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  

  # img = cv2.imread('webcam.jpg')
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray, 1.2, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.line(frame,(x,y),(x,y),(0,255,0),5)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
  print("one loop")
  
  cv2.imshow('preview',frame)

  key = cv2.waitKey(40)
  if key == 'q':
    break



cv2.destroyAllWindows()