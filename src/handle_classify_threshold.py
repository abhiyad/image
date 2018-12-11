#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('image')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import math

hue_thresh = 0
sat_thresh = 100
val_thresh = 0
canny_thresh_low = 5
canny_thresh_high = 150


def hue(X):
	global hue_thresh
	hue_thresh = X
	#print("hue at %d"%X)

def sat(X):
	global sat_thresh
	sat_thresh = X
	#print("sat at %d"%X)

def val(X):
	global val_thresh
	val_thresh = X
	#print("Val at %d"%X)

def canny_low(X):
	global canny_thresh_low
	canny_thresh_low = X
	#print("canny_low_thresh at %d"%X)

def canny_high(X):
	global canny_thresh_high
	canny_thresh_high = X
	#print("canny_high_thresh at %d"%X)


################################# utility functions

def select_rgb_white(image):
    # white color mask
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    return white_mask
def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
def select_hsv_white(image):
    converted = convert_hsv(image)
    # white color mask
    lower = np.uint8([  hue_thresh, sat_thresh,   val_thresh])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    return white_mask
def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def apply_smoothing(image,kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
def detect_edges(image):
    return cv2.Canny(image, canny_thresh_low, canny_thresh_low)
def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)
def select_region(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


##################################################################




class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("segmented_image",Image,queue_size=1000)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)

  def callback(self,data):
	try:
	  cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
	  print(e)

    # (rows,cols,channels) = cv_image.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)

    # Do your processing here, see syntax as above
    # original = cv_image
    # cv_image = select_region(original)

	t1=select_hsv_white(cv_image)
	kernel = np.ones((2,2),np.uint8)
	t2 = cv2.erode(t1,kernel,iterations = 1)
	t2=select_region(t2)

	cv2.imshow("Image window", t2)


	cv2.createTrackbar('hue',"Image window",0,179,hue)
	cv2.createTrackbar('sat',"Image window",0,255,sat)
	cv2.createTrackbar('Val',"Image window",0,255,val)
	cv2.createTrackbar('canny_low_threshold',"Image window",0,99,canny_low)
	cv2.createTrackbar('canny_high_threshold',"Image window",0,99,canny_high)

	cv2.waitKey(3)

	try:
	  self.image_pub.publish(self.bridge.cv2_to_imgmsg(t2, "8UC1"))
	except CvBridgeError as e:
	  print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_segmentation_node', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
