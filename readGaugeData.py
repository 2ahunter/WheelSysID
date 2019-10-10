#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:25:35 2019

@author: aahunter
"""
import numpy as np
import scipy.io as sio
import cv2
import glob
from matplotlib import pyplot as plt

def ReadMultipleImages(dir,cond):
    images = ()
    filenames = sorted(glob.glob(dir))
    for file in filenames:
        if cond in file:
            img = cv2.imread(file,0)
            images = images + (img,)
    return images

# Load images in grayscale
images = ReadMultipleImages('./gaugeCV/*.JPG','IMG')

# Load the reference image
ref_img = cv2.imread('./gaugeCV/calibration/back.JPG',0)
im = cv2.resize(ref_img, (0,0), fx=0.3, fy=0.3) 
cv2.imwrite('ref.png',im)

# Load the zero angle calibration image
zero_img = cv2.imread('./gaugeCV/calibration/zero.JPG',0)
im = cv2.resize(zero_img, (0,0), fx=0.3, fy=0.3) 
cv2.imwrite('zero.png',im)

# Circle geometry and gauge calibration data
circles = np.zeros((2,3),dtype=np.uint16)
# gauge centers and radius--determine from the reference image manually
c1_x = 1697
c1_y = 1259
c2_x = 1961
c2_y = 3187
c_r = 755

c1 = np.array([c1_x,c1_y])
c2 = np.array([c2_x,c2_y])
#needle length in pixels--deetermine manually
needle = 740
gauge1 = np.array([c1_x,c1_y,c_r])
gauge2 = np.array([c2_x,c2_y,c_r])
circles[0]=gauge1
circles[1]=gauge2

#generate a mask of the gauge shape
# circles = np.uint16(np.around(circles))
rad_offset = 75  
inner_radius = 75  
mask=np.zeros((ref_img.shape),dtype = np.uint8)
for i in circles:
    # fill the outer circle
    cv2.circle(mask,(i[0],i[1]),i[2]-rad_offset,255,-1)
    # zero the innter circle
    cv2.circle(mask,(i[0],i[1]),i[2]-(rad_offset+inner_radius),0,-1)    
    
# Gaussian blur the reference images
blur_ref = cv2.GaussianBlur(ref_img,(9,9),0 )
blur = cv2.GaussianBlur(zero_img,(9,9),0)
# subtract blur
delta_img = cv2.subtract(blur_ref,blur)
img2 = cv2.resize(delta_img, (0,0), fx=0.3, fy=0.3) 
cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.imwrite('delta_img.png',img2)

# Evaluate threshold values
thresh = 70 
maxval=255
ret, thresholded = cv2.threshold(delta_img, thresh, maxval, cv2.THRESH_BINARY);
for i in circles:
    # fill the outer circle
    cv2.circle(thresholded,(i[0],i[1]),i[2]-rad_offset,255,12 )
    # zero the innter circle
    cv2.circle(thresholded,(i[0],i[1]),i[2]-(rad_offset+inner_radius),255,12)
img2 = cv2.resize(thresholded, (0,0), fx=0.3, fy=0.3) 

cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.imwrite('threshold_img.png',img2)

delta_img = cv2.bitwise_and(thresholded,mask)    
img2 = cv2.resize(delta_img, (0,0), fx=0.3, fy=0.3) 
cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.imwrite('mask_img.png',img2)

# Calculate the zero angle for the gauges from the zero reference image
# find needles edges
edges = cv2.Canny(delta_img,50,150)
#get the contours of the edges
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# x,y,area of the centroids of the needle end
# There can be more than one contour, so need to add them all into their respective arrays
centroid1 = np.zeros((1,1,2),dtype=np.int64)
centroid2 = np.zeros((1,1,2),dtype=np.int64)
for contour in contours:
    # calculate the centroid of each contour
    sum = np.int64(np.sum(contour,axis=0))
    weight = len(contour)
    m00 = np.uint64(sum/weight)
    needle_len = np.linalg.norm(m00-c1)
    if needle_len < needle: # in first gauge
        centroid1 = np.concatenate((centroid1, contour), axis=0)
    else: # in second gauge
        centroid2 = np.concatenate((centroid2, contour), axis=0)
        
centroid1 = np.uint64(np.sum(centroid1,axis=0)/(len(centroid1)-1))
centroid2 = np.uint64(np.sum(centroid2,axis=0)/(len(centroid2)-1))
# gauge needle vectors
needle1 = centroid1 - c1
needle2 = centroid2 - c2
# calculate the angle of the needles
ang_zero1 = np.arctan2(needle1[0][1],needle1[0][0])
ang_zero2 = np.arctan2(needle2[0][1],needle2[0][0])

#gauge sensitivity in rad/mm
resolution=5/np.pi

# Get the data containers
Lateral = []
Radial = []
# Process each image in turn.  We assume that they are taken at uniform 
# intervals around the wheel
for img in images:  
    # Gaussian blur the image
    blur = cv2.GaussianBlur(img,(9,9),0)
    delta_img = cv2.subtract(blur_ref,blur)

    # thresholding 
    ret, thresholded = cv2.threshold(delta_img, thresh, maxval, cv2.THRESH_BINARY);
    # apply the mask
    delta_img = cv2.bitwise_and(thresholded,mask)    
    
    # find needles edges
    edges = cv2.Canny(delta_img,50,150)
    #get the contours
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # x,y,area of the centroids of the needle end
    # There can be more than one contour, so need to add them all into their respective arrays
    centroid1 = np.zeros((1,1,2),dtype=np.int64)
    centroid2 = np.zeros((1,1,2),dtype=np.int64)
    for contour in contours:
        # calculate the centroid of each contour
        sum = np.int64(np.sum(contour,axis=0))
        weight = len(contour)
        m00 = np.uint64(sum/weight)
        needle_len = np.linalg.norm(m00-c1)
        if needle_len < c_r: # in first gauge
            centroid1 = np.concatenate((centroid1, contour), axis=0)
        else: # in second gauge
            centroid2 = np.concatenate((centroid2, contour), axis=0)
    
    centroid1 = np.uint64(np.sum(centroid1,axis=0)/(len(centroid1)-1))
    centroid2 = np.uint64(np.sum(centroid2,axis=0)/(len(centroid2)-1))
    # gauge needle vectors
    needle1 = centroid1 - c1
    needle2 = centroid2 - c2
    # calculate the angle of the needles
    gauge1_ang = np.arctan2(needle1[0][1],needle1[0][0])
    gauge2_ang = np.arctan2(needle2[0][1],needle2[0][0])
    
    #calculate the angle from the reference image and convert to displacement
    gauge1_reading = resolution*(gauge1_ang-ang_zero1)
    gauge2_reading = resolution*(gauge2_ang-ang_zero2)
#    print('gauge 1 reading: ',gauge1_reading,' [mm]')    
#    print('gauge 2 reading: ',gauge2_reading,' [mm]')   
    
    #append to Displacements array
    Lateral.append(gauge1_reading)
    Radial.append(gauge2_reading)
    
# show the data:
fig = plt.figure(figsize=[10,10])
plt.subplot(2,1,1)
plt.title('Lateral and Radial Diplacements')
plt.plot(Lateral,label = 'lateral')
plt.ylabel('Displacement [mm]')
plt.legend()
plt.subplot(2,1,2)
plt.plot(Radial,label = 'radial')
plt.ylabel('Displacement [mm]')
plt.legend()
data = [Lateral,Radial]
sio.savemat('./gaugeCV/CV_valid.mat',mdict={'CV_valid':data})