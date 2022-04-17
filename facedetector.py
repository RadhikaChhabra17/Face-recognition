import cv2 
from random import randrange # for random color of frame

# load the classifier
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# reading the image
img = cv2.imread('fd.jpeg')

#converting img to gray scale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# give (x,y,w,h)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# (top left),(bottom right),(b,g,r),width
for (x,y,w,h) in face_coordinates:      
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

#print(face_coordinates)
# it will print top left point x and y coordinate and width and height

cv2.imshow('Face detector',img )
cv2.waitKey()
                                 