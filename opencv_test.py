import picamera
import time
import cv2
import numpy as np

cam = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

while(True):
    ret, img = cam.read()
    cv2.imshow("Show FLAME Image", img)
    
    k = cv2.waitKey(1)
    if k == ord('s'):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(gray, 1.1, 3)

        if len(face) > 0:
            for rect in face:
                cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=2)
            else:
                print( "no face" )

        cv2.imwrite('detected.jpg', img)
    
    elif k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
