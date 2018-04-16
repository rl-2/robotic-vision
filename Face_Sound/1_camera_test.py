# This code captuers an image and save on the pi
# Rodger (Jieliang) Luo, Apr 2018

import time
import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (1920, 1080)
    camera.hflip = True
    camera.vflip = True 
    camera.start_preview()
    
    # Camera warm-up time
    time.sleep(2)
    
    camera.capture('images/test.jpg')