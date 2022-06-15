import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y= fetch_openml('mnist_784', version=1, return_X_y=True)
classes = ['0','1','2','3','4','5','6','7','8','9']
print(pd.Series(y).value_counts())

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,train_size=0.7,random_state=0)
x_train_scale = x_train/255
x_test_scale = x_test/225

classifier = LogisticRegression(solver='saga',multi_class='multinomial').fit(x_train_scale,y_train)

y_pred= classifier.predict(x_test_scale)
accuracy = accuracy_score(y_test,y_pred)
confusion_m = confusion_matrix(y_test,y_pred)
print(accuracy)
print(confusion_m)

cam = cv2.VideoCapture(0)

while(True):
    try:
        grey= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,weight = grey.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56)) 
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(grey,upper_left,bottom_right,(0,255,0),2)
        roi = grey[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        im_pil = Image.fromarray(roi)
        img_bw = im_pil.covert('L')
        img_bw_resize = img_bw.resize((28,28),Image.ANTIALIAS)
        img_bw_resize_inverted = PIL.ImageOps.invert(img_bw_resize)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255) 
        max_pixel = np.max(image_bw_resized_inverted) 
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = classifier.predict(test_sample)
        print('predicted class',test_pred)
        cv2.imshow('frame',grey)
    except Exception as e:
        pass
cam.release()
cv2.destroyAllWindows()
