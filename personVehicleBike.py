import os
import sys

import cv2
import numpy as np
from IPython import display
from openvino.runtime import Core


class nplate_detection():
    def __init__(self):
        
        #load model
        ie = Core()
        model = ie.read_model(model='models/vehicle-detection-adas-0002.xml')     
        
        self.compiled_model = ie.compile_model(model=model, device_name="GPU")
        
        self.output_layer = self.compiled_model.output(0)
        input_layer = self.compiled_model.input(0)
        self.height, self.width = list(input_layer.shape)[2:4]
        
    
    def get_boxes(self, frame, results, thresh=0.1):
        # The size of the original frame.
        h, w = frame.shape[:2]
        results = results.squeeze()
        boxes = []
       
        ttl = 0
        for idx, label, confidence, xmin, ymin, xmax, ymax in results:
            if (label == 1. or label == 2.) and confidence > thresh:
                ttl = ttl + 1
                print ("idx = ", idx, "label = ", label, " conf = ", confidence, " idx = ", ttl , " h = ", ttl , " w = ", w )        
                # Create a box with pixels coordinates from the box with normalized coordinates [0,1].            
                boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
        
        if ttl >0:
            print ("ttl vehicles = ", ttl)
        
        return boxes

    def get_points(self,box,bounds,stretch=0):
        #the NN can sometimes return negative numbers that makes no sense 
        box=[max(x,0) for x in box]
    
        #getting points of the ends of the box (stretching a bit)
        x1 = box[0] - stretch*box[2]
        x2 = box[0] + (1+stretch)*box[2]
        y1 = box[1] - stretch*box[3]
        y2 = box[1] + (1+stretch)*box[3]
    
        #make sure that after streching, we are still in the image boundaries..
        x1,x2=(int(min(max(x,0),bounds[1])) for x in (x1,x2))
        y1,y2=(int(min(max(y,0),bounds[0])) for y in (y1,y2))
        assert x1<=x2 and y1<=y2 
        return x1,x2,y1,y2

    def process_plates(self,frame, boxes):
        
        final_image=frame.copy()
        color = (0,200,0)
        for box in boxes:
            x1,x2,y1,y2=self.get_points(box,final_image.shape[:2])
            cv2.rectangle(img=final_image, pt1=(x1,y1), pt2=(x2, y2), color=color, thickness=1)
     
        return final_image   

nplate_det=nplate_detection()


# Initialize the camera capture object with the cv2.VideoCapture class.
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
else:
    # Set the resolution of the camera (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read and display frames in a loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        #enable this line to check on static image      
        #frame = cv2.imread("car04.jpg")
                
        #openvino inferencing here
        input_img = cv2.resize( src=frame, dsize=(nplate_det.width, nplate_det.height))
        #input_img = cv2.resize( src=frame, dsize=(300, 300))
        input_img = input_img[np.newaxis, ...]
        input_img=np.transpose(input_img,[0,3,1,2])
        results = nplate_det.compiled_model([input_img])[nplate_det.output_layer]
        boxes = nplate_det.get_boxes(frame=frame, results=results, thresh=0.7)
        final_output = nplate_det.process_plates(frame=frame, boxes=boxes)  
        
        # Display the resulting frame
        cv2.imshow('Camera Feed', final_output)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
