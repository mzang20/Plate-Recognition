import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import easyocr
from ultralytics import YOLO

model = YOLO('C:/Users/mingj/OneDrive/Desktop/Plate_Recognition/model.pt')

def predict(path_test_car, filename):
    # Perform prediction on the test image using the model
    results = model.predict(path_test_car)

    image = cv2.imread(path_test_car)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(['en'])

    # Extract the boxes and labels from the results
    for result in results:
        for box in result.boxes:
            # Get the coordinates of the box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get the confidence score of the prediction
            confidence = box.conf[0]

            # Draw the box on the image
            image_bgr = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the confidence score
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Crop the box from the image for OCR
            roi_bgr = gray_image[y1:y2, x1:x2]

            # Perform OCR on the cropped image
            text = reader.readtext(roi_bgr)
            if len(text) > 0:
                text = text[0][1]
            cv2.putText(image, f'{text}', (x1, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 123, 255), 2)
            
            # Store box and confidence score into predict folder
            cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
            # Store cropped box into roi folder
            cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)

            print(text)
            
    return text