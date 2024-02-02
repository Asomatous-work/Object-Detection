import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import filedialog

# Path to your darknet configuration and weight files
net = cv2.dnn.readNetFromDarknet("/Users/aravind/Desktop/im/actual_yolov3.cfg", "/Users/aravind/Desktop/im/actual_yolov3.weights")

# Reading class names from coco.names
class_labels = []
with open("/Users/aravind/Desktop/im/coco.names", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

def access_camera():
    cap = cv2.VideoCapture(0)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]



    while True:
        ret, frame = cap.read()
        
        if ret:
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                detections = net.forward(output_layers)

                for out in detections:
                    for detection in out:
                        scores = detection[5:]
                        class_index = np.argsort(scores)[-1]
                        confidence = scores[class_index]
                        if confidence > 0.5:
                            center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
  
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, class_labels[class_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Camera Feed", frame)
            else:
                print("Empty frame detected")
        
        if cv2.waitKey(1) == ord('q'): # Quit by pressing 'q'
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    access_camera()
