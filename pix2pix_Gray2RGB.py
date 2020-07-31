import numpy as np
import glob
import h5py
import cv2
from keras.preprocessing.image import load_img,img_to_array

def load_path(dirpath):

def load_img(dirpath):
    

#Load the Image
imgo = cv2.imread('D:\\meter\\meter_dataset\\meter_images_2160\\meter1\\0700\\0704a03.jpg')
height, width = imgo.shape[:2]
mask = np.zeros(imgo.shape[:2],np.uint8)

#Grab Cut the object
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#Hard Coding the Rect The object must lie within this rect.
rect = (10,10,width-30,height-30)
cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = imgo*mask[:,:,np.newaxis]

#Get the background
background = imgo - img1

#Change all pixels in the background that are not black to white
background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

#Add the background and the image
final = background + img1

#To be done - Smoothening the edges
cv2.imwrite("./maskImg.jpg",final)
