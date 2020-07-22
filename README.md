# face-detection-using-opencv
## Basic requirements:

### 1- Libraries

1.1- cv2

1.2- numpy

1.3- matplotlib

### 2-Files 
2.1- A image of a person 

2.2 - Cascade classifier haarcascade_frontalface_default.xml.

## The purpose of using:
### image
for a face detection

### Cascade classifie
contain the feature of the face 

### opencv 
1- read image.

2- feature filed in will convert the image to numpy array.

## The steps:
### Install opencv in colab
---
!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python 

import cv2

---
### To download any file, we will use here to download the image.
---

from google.colab import files

file = files.upload()

---

### import for Libraries
---

import numpy as np

import cv2

import matplotlib.pyplot as plt # to show image

from google.colab.patches import cv2_imshow # to show image


---

###

---
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img=cv2.imread('man (1).jpg') # read image

gray=cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY) 

faces =face_cascade.detectMultiScale( gray, 1.3 ,5 )

print(faces) #

---

###

---

for (x,y,w,h) in faces:

        cv2.rectangle(img ,(x,y) , (x+w,y+h),(255,0,0),2)

        roi_gray = gray [y:y+h , x:x+w]

        roi_color = img [y:y+h , x:x+w]

cv2_imshow(img) # Show image

cv2.waitKey(0) # wait for user tp press a key.

cv2.destroyAllWindows() # close window based on waitKey 

---


## citation and references :
* [the original code](https://www.youtube.com/watch?v=-Zgm3JWkM6A)
* [Cascade classifier haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)

