#! /usr/bin/env python
import cv2
from threshold import thresholdModel

img=cv2.imread('top_view.jpg',-1)
img =cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
while(True):
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    th=thresholdModel(img)
    cv2.waitKey(1000)
# cv2.waitKey(0)
cv2.imwrite('top_view_out.jpg',th)