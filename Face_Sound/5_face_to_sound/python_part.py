#========================================================================== 
# The example shows how PiCamera is looking for faces.
# Once it finds a face, it saves an image and send a singal to Arduino 
#
# Rodger (Jieliang) Luo
# April, 2018
#==========================================================================

import time
import io
import os
import sys
from cStringIO import StringIO
import random
import pipan
import picamera
import cv2
import numpy
import base64
import serial

ser = serial.Serial('/dev/ttyACM0', 9600)

find_face = False
serach_time = 0

def looking_face():

	#Create a memory stream so photos doesn't need to be saved in a file
	stream = io.BytesIO()

	with picamera.PiCamera() as camera:
	    camera.resolution = (1280, 720)
	    camera.hflip = True
	    camera.vflip = True
	    camera.capture(stream, format='jpeg')

    #Convert the picture into a numpy array
	buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

	#Now creates an OpenCV image
	image = cv2.imdecode(buff, 1)

	#downsize the image for face recognition
	image_small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

	#Load a cascade file for detecting faces
	face_cascade = cv2.CascadeClassifier('face.xml')

	#Convert to grayscale
	gray = cv2.cvtColor(image_small,cv2.COLOR_BGR2GRAY)

	#Look for faces in the image using the loaded cascade file
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	print "Found "+str(len(faces))+" face(s)"

	if len(faces) > 0:

		# If a face is found, send a meesage to Arduino
		ser.write('1')
		
		#Draw a rectangle around every found face
		for (x,y,w,h) in faces:
			cv2.rectangle(image,(x*2,y*2),(x*2+w*2,y*2+h*2),(255,255,0),2)

		cv2.imwrite('face_found.jpg',image)

		exit()

		
def main():

	raw_input("press enter to start searching")
	print 'Search Begin!'

	# Infinite Loop
	while True:
		looking_face()

if __name__ == '__main__':
    main()