#========================================================================== 
# Servo Exercise Starter Code 
#
# Written by: Rodger (Jieliang) Luo
# April 10th, 2018
#==========================================================================
import time
import random
import pipan

# Ranges of pan and tilt servos for our purpose
# The actual physical range for pan is (80, 220), for tilt is (50, 250)

PAN_MIN_POS = 120
PAN_MAX_POS = 180

TILT_MIN_POS = 110
TILT_MAX_POS = 140
		
def main():

	#Center the camera's position
	servo = pipan.PiPan()
	servo.neutral_pan()
	servo.neutral_tilt()

	time.sleep(2)

	# Start position of pan and tilt servo
	pan_pos = 150
	tilt_pos = 150

	# Infinite Loop
	while True:

		tilt_pos = random.randint(TILT_MIN_POS, TILT_MAX_POS)
		pan_pos = random.randint(PAN_MIN_POS, PAN_MAX_POS)

		servo.do_tilt(tilt_pos)
		servo.do_pan(pan_pos)

		time.sleep(2)

if __name__ == '__main__':
    main()