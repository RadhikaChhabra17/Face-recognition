import cv2 
from random import randrange 

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0) # for video enter the path of video file in place of 0

while True:
    successful_frame_read,frame=webcam.read() #read current frame
    #tell two things successful or not and frame

    grayscaled_img = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:      
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

    cv2.imshow('Face detector',frame )
    key = cv2.waitKey(1) # cv2.waitKey(1) wait for 1 ms 

    #stop when q is pressed
    if key==81 or key==113:
        break

webcam.release() #release video capture object
