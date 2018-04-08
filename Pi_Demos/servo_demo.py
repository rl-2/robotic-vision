#========================================================================== 
# The code controls camera movement and facial recognition.
# In terms of camera movement, it has two situations: subtle search and obvious search.
# In subtle search, the servos slightly change based on the previous position. 
# Once the camera finds a face, the Python sends a message to Audiuno 
# and the servos move the camera in large angles for seveal times.
# If no more faces are found, it changes back to subtle search mode.  

# The code was initially written for serving EOYS MAT 2017. 

# Written by: Rodger (Jieliang) Luo
# May 11th, 2017
#==========================================================================
import time
import io
import os
import sys
from cStringIO import StringIO
import random

import socket
import struct

import pipan
import picamera
import cv2
import numpy
import base64

import serial

# Ranges of pan and tilt servos for our purpose
# The actual physical range for pan is (80, 220), for tilt is (50, 250)
PAN_MIN_POS = 120
PAN_MAX_POS = 180

TILT_MIN_POS = 110
TILT_MAX_POS = 140

# ---- DO NOT EDIT THIS PART ----
#IP address of the server
TCP_IP = '10.0.1.16'
TCP_PORT = 9001
BUFFER_SIZE = 1024

# ser = serial.Serial('/dev/ttyACM0', 9600)

find_face = False
serach_time = 0
# ---- DO NOT EDIT THIS PART ----

def face_recognition():

	global find_face, serach_time
	
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
	face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/RobotSwarm/face.xml')

	#Convert to grayscale
	gray = cv2.cvtColor(image_small,cv2.COLOR_BGR2GRAY)

	#Look for faces in the image using the loaded cascade file
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	print "Found "+str(len(faces))+" face(s)"

	if len(faces) > 0:

		# If a face is found, send a meesage to Arduino
		ser.write('1')
		
		find_face = True
		serach_time = 0

		#Draw a rectangle around every found face
		for (x,y,w,h) in faces:
			cv2.rectangle(image,(x*2,y*2),(x*2+w*2,y*2+h*2),(255,255,0),2)

		# ---- DO NOT EDIT THIS PART ----
		#send the image to the server
		client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client_socket.connect((TCP_IP, TCP_PORT))

		# Make a file-like object out of the connection
		connection = client_socket.makefile('wb')

		try:
			connection.write(struct.pack('<L', stream.tell()))
	   		connection.flush()
	      	# Rewind the stream and send the image data over the wire
	  		stream.seek(0)
	   		connection.write(stream.read())
			# Write a length of zero to the stream to signal we're done
			connection.write(struct.pack('<L', 0))

		finally:
		    connection.close()
		    client_socket.close()
		 # ---- DO NOT EDIT THIS PART ----

	else:
		print "No face found..."
		serach_time += 1

		# If no more faces are found within three searches, changes back to subtle serach
		if(serach_time > 3):
			find_face = False
		
def main():

	#Center the camera's position
	servo = pipan.PiPan()
	servo.neutral_pan()
	servo.neutral_tilt()

	# raw_input("press enter to start searching")
	# print 'Search Begin!'
	time.sleep(5)

	# Start position of pan and tilt servo
	tilt_pos = 120
	pan_pos = 150

	# Infinite Loop
	while True:

		# Subtle serach when no face is found
		if(find_face is False):
			tilt_pos += random.randint(-10, 10)
			pan_pos += random.randint(-15, 15)

		# Obvious serach when a face is found
		else:
			tilt_pos = random.randint(TILT_MIN_POS, TILT_MAX_POS)
			pan_pos = random.randint(PAN_MIN_POS, PAN_MAX_POS)

		# make sure the servos also in the range 
		if tilt_pos > TILT_MIN_POS or tilt_pos < TILT_MAX_POS:
			servo.do_pan(pan_pos)
		else:
			servo.do_pan(150)
		
		if pan_pos > PAN_MIN_POS or pan_pos < PAN_MAX_POS:
			servo.do_tilt(tilt_pos)
		else:
			servo.do_tilt(120)

		time.sleep(5)
		# face_recognition()

if __name__ == '__main__':
    main()