import cv2
import numpy as np
import os
from skimage.morphology import erosion, dilation, opening, closing

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
 
def transform(pos):
# This function is used to find the corners of the object and the dimensions of the object
    pts=[]
    n=len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))
       
    sums={}
    diffs={}
    tl=tr=bl=br=0
    for i in pts:
        x=i[0]
        y=i[1]
        sum=x+y
        diff=y-x
        sums[sum]=i
        diffs[diff]=i
    sums=sorted(sums.items())
    diffs=sorted(diffs.items())
    n=len(sums)
    rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]
    #      top-left   top-right   bottom-left   bottom-right
   
    h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)     #height of left side
    h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)     #height of right side
    h=max(h1,h2)
   
    w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)     #width of upper side
    w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)     #width of lower side
    w=max(w1,w2)
    
    #Khalli el soora ma3doola msh bl gamb
    if w>h:
        temp = h
        h = w
        w = temp
   
    return int(w),int(h),rect
 


def fixOrientation(img):
    print("fixOrientation")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    # Handles Lighting
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # blur. very CPU intensive.
    edges = cv2.Canny(gray, 30, 120)
    # edges = auto_canny(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(edges, kernel)
    _, contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)
    max_area = 0
    pos = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            pos = i
    # Handling the images that are already captured while being close
    if max_area < 8000:
        finalImage = img
    else:
        peri = cv2.arcLength(pos, True)
        approx = cv2.approxPolyDP(pos, 0.02 * peri, True)


        size = img.shape
        w, h, arr = transform(approx)

        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts1 = np.float32(arr)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (w, h))
        finalImage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return finalImage
img=cv2.imread('../Salma/sama/test1.jpg')
