import RPi.GPIO as GPIO
import picamera
import time
from plate_recog0810 import process_and_recognize_license_plate

GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
IN_trigger_pin = 3
IN_echo_pin = 11
OUT_trigger_pin = 5
OUT_echo_pin = 13
IN_s90_pin = 12
GPIO.setup(IN_trigger_pin, GPIO.OUT)
GPIO.setup(IN_echo_pin, GPIO.IN)
GPIO.setup(OUT_trigger_pin, GPIO.OUT)
GPIO.setup(OUT_echo_pin, GPIO.IN)
GPIO.setup(IN_s90_pin, GPIO.OUT)
# Initialize the PWM for servo control
s90_pwm = GPIO.PWM(IN_s90_pin, 50)  # Frequency set to 50 Hz
s90_pwm.start(0)

def s90(angle):   # 0、90、180
    s90_pwm.ChangeDutyCycle(2.5+angle/360*20)


def send_trigger_pulse(trigger_pin):
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.0001)
    GPIO.output(trigger_pin, GPIO.LOW)

def wait_for_echo(echo_pin, value, timeout):
    count = timeout
    while GPIO.input(echo_pin) != value and count > 0:
        count = count - 1

def get_distance(trigger_pin, echo_pin):
    send_trigger_pulse(trigger_pin)
    wait_for_echo(echo_pin, GPIO.HIGH, 10000)
    start = time.time()
    wait_for_echo(echo_pin, GPIO.LOW, 10000)
    finish = time.time()
    pulse_len = finish - start
    distance_cm = pulse_len / 0.000058
    return distance_cm

def blink_led(pin):
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(0.005)
    GPIO.output(pin, GPIO.LOW)
    time.sleep(0.005)

def turn_on_green_led():
    GPIO.output(16, GPIO.HIGH)  # Turn on the green LED
    GPIO.output(15, GPIO.LOW)  # Turn off the red LED

def turn_on_red_led():
    GPIO.output(15, GPIO.HIGH)  # Turn on the red LED
    GPIO.output(16, GPIO.LOW)  # Turn off the green LED

def turn_off_red_led():
    GPIO.output(15, GPIO.LOW)  # Turn off the red LED

def turn_on_red_led_and_wait():
    GPIO.output(15, GPIO.HIGH)  # Turn on the red LED
    time.sleep(4)
    GPIO.output(15, GPIO.LOW)  # Turn off the red LED

def turn_off_green_led():
    GPIO.output(16, GPIO.LOW)  # Turn off the green LED

def process_out_sensor_distance():
    out_distance = get_distance(OUT_trigger_pin, OUT_echo_pin)

    if out_distance < 100:
        turn_on_red_led()
    else:
        turn_on_green_led()


blink_flag=True
def process_out_sensor_distance_blink():
    global blink_flag
    out_timer_start = time.time()
    while time.time() - out_timer_start < 4:
        # Blink the green LED with a shorter on-time and longer off-time
        GPIO.output(15, GPIO.HIGH)  # Turn on the green LED
        time.sleep(0.05)  # Short on-time (0.1 seconds)
        GPIO.output(15, GPIO.LOW)  # Turn off the green LED
        time.sleep(0.05)  # Longer off-time (0.4 seconds)
        if get_distance(OUT_trigger_pin, OUT_echo_pin) < 30:
            break
    turn_off_green_led()  # Turn off the green LED
    turn_on_red_led()  # Turn on the red LED

#license_plate_processed=False
while True:
    out_distance = get_distance(OUT_trigger_pin, OUT_echo_pin)
    in_distance = get_distance(IN_trigger_pin, IN_echo_pin)
    if out_distance < 30 and in_distance > 30:
        turn_off_green_led()
        turn_on_red_led()
        s90(0)
    if out_distance < 30 and in_distance < 30:
        turn_off_green_led()
        turn_on_red_led()
        s90(0)
    if out_distance > 30 and in_distance > 30:
        turn_on_green_led()
        turn_off_red_led()
        s90(0)
    if out_distance > 30 and in_distance < 30:
        turn_on_green_led()
        process_out_sensor_distance_blink()
       
        with picamera.PiCamera() as camera:
           camera.start_preview()  
           time.sleep(1)  
           camera.capture('/ALPR/imgs/car/image.jpg')  
           camera.stop_preview()  
           image_path ='ALPR/imgs/car/image.jpg'
           #image_path ='/ALPR/imgs/car/1.jpg'
          
           Num_matching=process_and_recognize_license_plate(image_path)
           print("Num_matching:",Num_matching)
              if Num_matching>=4:
                  s90(90)
                  #license_plate_processed=True
                  blink_flag=False                        
    time.sleep(0.1)

GPIO.cleanup()

