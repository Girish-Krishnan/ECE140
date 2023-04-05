import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib # import the motor library
import time

app = FastAPI()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
templates = Jinja2Templates(directory="templates")
# Face detection using Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Stepper Motor Setup
GpioPins = [18, 23, 24, 25]
# Declare a named instance of class pass a name and motor type
mymotor = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")

#min time between motor steps (ie max speed)
step_time = 0.01

# PID gain values
Kp = 0.015
Kd = 0.001
Ki = 0.001

#error values
d_error = 0
last_error = 0
sum_error = 0

################# Computer Vision #################

def get_perspective(img, location, height = 400, width = 800):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result  

def detect_faces(frame):
    detected_faces = face_cascade.detectMultiScale(frame)
    # find largest face
    largest_area = 0
    face_coord = [0,0]
            
    if len(detected_faces) != 0:
                
            start = time.time()
            for (x,y,width, height) in detected_faces:
                if width*height > largest_area:
                    largest_area = width*height
                
            for (x,y,width,height) in detected_faces:
                cv2.rectangle(frame,(x,y),(x + width, y + height),(0, 0, 255),2)
                if width*height == largest_area:
                    face_coord = [x+ width//2, y + height//2]

    error = -1 * (frame.shape[0] // 2 - face_coord[0])

    # Draw a vertical line down the center of the frame
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (0, 255, 0), 2)

    # Draw an error line showing the deviation from center along x axis
    cv2.line(frame, (frame.shape[1] // 2, frame.shape[0] // 2), (frame.shape[1] // 2 + error, frame.shape[0] // 2), (0, 0, 255), 2)

    return error, start, frame

################# Motor Control #################

def update_motor_pid(motor,GpioPins,step_time,Kp,Ki,Kd,error,sum_error,d_error, start):

    #speed gain calculated from PID gain values
    speed = Kp * error + Ki * sum_error + Kd * d_error

    #if negative speed change direction
    direction = (speed < 0)
                    
    #inverse speed set for multiplying step time (lower step time = faster speed)
    speed_inv = abs(1/(speed))

    #get delta time between loops
    delta_t = time.time() - start
    
    #calculate derivative error
    d_error = (error - last_error)/delta_t
        
    #integrated error
    sum_error += (error * delta_t)

    last_error = error
                    
    if abs(error) > 15:
        motor.motor_run(GpioPins , speed_inv * step_time, 1, direction, False, "full", .05)
                    
    else:
        #run 0 steps if within an error of 20
        motor.motor_run(GpioPins , step_time, 0, direction, False, "full", .05)

    return sum_error, d_error, last_error

################# Main Loop for Live Streaming #################

def gen_frames():
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Face detection to give error value
            error, start, frame = detect_faces(frame)

            # PID control of motor
            sum_error, d_error, last_error = update_motor_pid(mymotor,GpioPins,step_time,Kp,Ki,Kd,error,sum_error,d_error, start)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    GPIO.cleanup()