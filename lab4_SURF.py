import cv2
import numpy as np

#read
img = cv2.imread('scene.jpg')

#grey
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#SIFT detector
#surf = cv2.SURF(400)
surf = cv2.xfeatures2d.SURF_create(3000,1,1,True,False)
kp, des = surf.detectAndCompute(gray,None)
img2 = cv2.drawKeypoints(gray,kp,None,(255,0,0),4)
cv2.imwrite('scene_SURF.jpg',img2)
