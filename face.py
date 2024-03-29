import os
import sys

import cv2
import numpy as np
from IPython import display
from openvino.runtime import Core

class emotion_detection():
    def __init__(self):
        #load model
        ie = Core()
        model = ie.read_model(model='models/emotions-recognition-retail-0003.xml')     
        self.compiled_model = ie.compile_model(model=model, device_name="GPU")
        
        self.output_layer = self.compiled_model.output(0)
        input_layer = self.compiled_model.input(0)
        self.height, self.width = list(input_layer.shape)[2:4]
emotion_det=emotion_detection()

class age_gender_detection():
    def __init__(self):
        #load model
        ie = Core()
        model = ie.read_model(model='models/age-gender-recognition-retail-0013.xml')     
        self.compiled_model = ie.compile_model(model=model, device_name="GPU")
        
        
        self.age = self.compiled_model.output(1)
        self.gender = self.compiled_model.output(0)
        input_layer = self.compiled_model.input(0)
        self.height, self.width = list(input_layer.shape)[2:4]
ag_det=age_gender_detection()

class face_detection():
    def __init__(self):
        
        #load model
        ie = Core()
        model = ie.read_model(model='models/face-detection-adas-0001.xml')        
        self.compiled_model = ie.compile_model(model=model, device_name="GPU")
        
        self.output_layer = self.compiled_model.output(0)
        input_layer = self.compiled_model.input(0)
        self.height, self.width = list(input_layer.shape)[2:4]
    
    def get_boxes(self, frame, results, thresh=0.1):
        # The size of the original frame.
        h, w = frame.shape[:2]
        results = results.squeeze()
        boxes = []
    
        for idx, label, confidence, xmin, ymin, xmax, ymax in results:
            if label==1. and confidence>thresh:
                # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
                boxes.append(
                    tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
                )
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

    def process_faces(self,frame, boxes):
        EMOTION_NAMES=['neutral','happy','sad','surprise','anger']

        final_image=frame.copy()
        color = (0,200,0)
        for box in boxes:              
            #Add another box on the final image
            x1,x2,y1,y2=self.get_points(box,final_image.shape[:2])
            cv2.rectangle(img=final_image, pt1=(x1,y1), pt2=(x2, y2), color=color, thickness=1)
        
            ##prepare input input image for emotion
            emotion_input = frame[y1:y2,x1:x2]     
        
            emotion_input = cv2.resize( src=emotion_input, dsize=(emotion_det.width, emotion_det.height))  
            emotion_input = emotion_input[np.newaxis].transpose([0,3,1,2])          
        
            #run emotion inference
            emotion_output = emotion_det.compiled_model([emotion_input])[emotion_det.output_layer]
            emotion_output = emotion_output.squeeze()
            index=np.argmax(emotion_output)        
        
            #cv2.putText(
             #   img=final_image,
              #  text=f"{' '}{EMOTION_NAMES[index]}",
               # org=(box[0] + 10, box[1] + 30),
                #fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #fontScale=frame.shape[1] / 1000,
                #color=color,
                #thickness=1,
                #lineType=cv2.LINE_AA,
            #)

            #age-gender input
            input_img=frame[y1:y2,x1:x2]
            input_img = cv2.resize(
                    src=input_img, dsize=(ag_det.width, ag_det.height),interpolation=cv2.INTER_AREA)
           
        
            input_img = input_img[np.newaxis]
            input_img=np.transpose(input_img,[0,3,1,2])
        
            #age-gender output
            output= ag_det.compiled_model([input_img])
            age,gender=output[ag_det.age],output[ag_det.gender]
        
            age=np.squeeze(age)
            age*=100
        
            gender=np.squeeze(gender)
            if (gender[0]>=0.65):
                gender='female '
            elif (gender[1]>=0.55):
                gender='male '
            else:
                gender='nb '
        
            #drawing results
            cv2.putText(
                img=final_image,
                text=f"{gender}{age:.0f}{' '}{EMOTION_NAMES[index]}", #{emotion_score:.0f}
                org=(box[0] + 10, box[1] + 30),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=frame.shape[1] / 1000,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )



        return final_image   

face_det=face_detection()


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

        #openvino inferencing here
        input_img = cv2.resize( src=frame, dsize=(face_det.width, face_det.height))
        input_img = input_img[np.newaxis, ...]
        input_img=np.transpose(input_img,[0,3,1,2])
        results = face_det.compiled_model([input_img])[face_det.output_layer]
        boxes = face_det.get_boxes(frame=frame, results=results)
        final_output = face_det.process_faces(frame=frame, boxes=boxes)  
                

        # Display the resulting frame
        cv2.imshow('Camera Feed', final_output)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
