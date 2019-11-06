# -*- coding: utf-8 -*-
# Face Recognition + Smile Recognition
# Importing the libraries
import cv2
 
# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load the cascade for the face.
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # Load the cascade for the smile
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # Load the cascade for the eye
# Defining the function that will do the detection
def detect(grey, frame): # Create a function that takes as input the image in black and white (grey) and the original image (frame), and that will return the same image with the detector rectangles.
    faces = face_cascade.detectMultiScale(grey, 1.3, 5) # Apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for(x,y,w,h) in faces: # For each detected face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) # Paint a rectangle around the face.
        roi_grey = grey[y:y+h, x:x+w] # Get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # Get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_grey, 1.1, 16) # Apply the detectMultiScale method to locate the smile in the image of ther face previously obtained.
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 
        smiles = smile_cascade.detectMultiScale(roi_grey, 1.7, 27) # Apply the detectMultiScale method to locate one or several eyes in the image of ther face previously obtained.
        
        for(sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,0,0),2) 
        
        
    return frame # Return the image with the detector rectangles.

# Doing some face recognition with the webcam
video_capture = cv2.VideoCapture(0);     #  0 -Internal webcam, 1- External Webcam
while True:
    _, frame = video_capture.read() # Get the last frame.
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Similar to rgb2gray in MATLAB
    canvas = detect(grey,frame) 
    cv2.imshow('Video',canvas) # Display the output.
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press the 'q' button in keyboard to stop running the program
        break
video_capture.release() # Turn the webcam off.
cv2.destroyAllWindows() # Destroy all the windows inside which the images were displayed.