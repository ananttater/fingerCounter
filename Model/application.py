import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import model_from_json,load_model
#from keras import load_weights
from keras import layers
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils,to_categorical
from keras import layers,regularizers
from keras import backend as K

cap = cv2.VideoCapture(0)

avgmodel=load_model('averagemodel.h5')
print('loaded from disk')

#Preprocessing part for the input images
thresh=137
def binary_mapping(im):
    im=np.array(im)
    img_read= im.astype('uint8')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(im_bw,-1,kernel)
    binary_im=np.array(dst)
    
    return binary_im

upper_left = (150, 150)
bottom_right = (450, 450)
while(cap):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #frame = cv2.convertScaleAbs(frame, alpha=3, beta=-500)
    
    #Rectangle marker
    cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 4)
    rect_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    
    rect_img= np.flip(rect_img,axis=1)
    im= Image.fromarray(rect_img,'RGB')
    b_img=im
    #Resize and Resscale
    input_img= im.resize((64,64))
    input_img=binary_mapping(input_img)
    input_img=input_img/255
    input_img= input_img.reshape((1,64,64,1))
    b_img=binary_mapping(b_img)
    #Model Predictions
    res3= avgmodel.predict_classes(input_img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text= str(res3[0]+1)           #+str((result[0])+1)+ '  1) ' 
    text2='Keep your palm in the'
    text3= 'middle of the red box'

    #Display on the frame
    frame= cv2.flip(frame,1)
    cv2.putText(frame,text,(50,300), font, 4,(123,120,250),5,cv2.LINE_AA)
    cv2.putText(frame,text2,(100,50), font, 1,(123,120,250),3,cv2.LINE_AA)
    cv2.putText(frame,text3,(120,100), font, 1,(123,120,250),3,cv2.LINE_AA)
    cv2.imshow('Cropped Frame',rect_img)
    cv2.imshow('Input Frame', frame)
    cv2.imshow('Preprocessed Frame',b_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
