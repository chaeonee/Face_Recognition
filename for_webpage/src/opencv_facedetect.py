#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:16:27 2018

@author: onee
"""

import numpy as np
import cv2
 
faceCascade = cv2.CascadeClassifier('./facenet/lbpcascade_profileface.xml')
image = cv2.imread('./ChildCarev2-20180509021906.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30)
)
 
print("Face Count : {0}".format(len(faces)))
 
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
cv2.imshow("Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)