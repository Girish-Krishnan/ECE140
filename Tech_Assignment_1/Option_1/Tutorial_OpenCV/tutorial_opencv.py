#==============================================================================
"""
OpenCV Tutorial
Girish Krishnan
"""
#==============================================================================

# Import the necessary libraries

import cv2
import numpy as np

#==============================================================================

# Create a camera object
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)

#==============================================================================

# Live stream, and capture an image if spacebar is pressed
while True:
    ret, frame = cap.read()
    cv2.imshow('Live Stream', frame)
    if cv2.waitKey(1) == ord(' '):
        cv2.imwrite("frame.jpg", frame)
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

#==============================================================================

# Now, read the image
img = cv2.imread('frame.jpg')

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)

#==============================================================================

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)

#==============================================================================

# Show the red layer
red = img.copy()
red[:, :, 0] = 0 # Set blue channel to 0
red[:, :, 1] = 0 # Set green channel to 0
cv2.imshow('Red', red)
cv2.waitKey(0)

# Show the green layer
green = img.copy()
green[:, :, 0] = 0 # Set blue channel to 0
green[:, :, 2] = 0 # Set red channel to 0
cv2.imshow('Green', green)
cv2.waitKey(0)

# Show the blue layer
blue = img.copy()
blue[:, :, 1] = 0 # Set green channel to 0
blue[:, :, 2] = 0 # Set red channel to 0
cv2.imshow('Blue', blue)
cv2.waitKey(0)

#==============================================================================

# Perform Canny edge detection on the grayscale image
edges = cv2.Canny(gray, 100, 200)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

#==============================================================================

# Apply other kinds of filters
blur = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imshow('Blur', blur)
cv2.waitKey(0)

sharpen = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
cv2.imshow('Sharpen', sharpen)
cv2.waitKey(0)

#==============================================================================

# FACE DETECTION

# Face detection using Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
img_faces = img.copy()

# Draw the faces on the original image
for (x, y, w, h) in faces:
    cv2.rectangle(img_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with faces
cv2.imshow('Faces', img_faces)
cv2.waitKey(0)

#==============================================================================

# Detect black rectangles using contours and mark them on the image

# Apply thresholding
ret, thresh = cv2.threshold(gray, 127, 255, 0)

# Draw the contours on the original image
img_contours = img.copy()

contours,_ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True) [:30]

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        # Draw the contour
        cv2.drawContours(img_contours, [c], -1, (0, 255, 0), 2)
        # x,y,w,h = cv2.boundingRect(c) 
        # cv2.rectangle(img_contours, (x,y), (x+w,y+h), (0,255,0), 2)
                   
# Display the image with contours
cv2.imshow('Contours', img_contours)
cv2.waitKey(0)

#==============================================================================

# Using masks to find objects of a certain color:

# Convert image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create a mask of blue pixels in the image
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply the mask to the original image to extract only blue objects
img_blue_objects = cv2.bitwise_and(img, img, mask=mask)

# Display the image with blue objects
cv2.imshow('Blue Objects', img_blue_objects)
cv2.waitKey(0)

#==============================================================================

# Close all windows
cv2.destroyAllWindows()