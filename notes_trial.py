
import cv2
# import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median

from skimage.filters import gaussian
from skimage.feature import canny


from skimage.filters import threshold_otsu

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def binarize(img):
    t=threshold_otsu(img)
    img[img>t]=255
    img[img<=t]=0
    return img

def unpackSIFTOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    # print(_octave)
    octave = _octave&0xFF
    # print(octave)
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)

    print(octave, layer, scale)
    return (octave, layer, scale)

# img1 = cv2.imread("word2.png", cv2.IMREAD_GRAYSCALE)
# print(img1.shape)
# img2= cv2.imread("notes/quarter_notes/4.png", cv2.IMREAD_GRAYSCALE)
# sift = cv2.xfeatures2d.SIFT_create()
# binarize(img1)
# binarize(img2)
# img1=255-img1
# show_images([img1,img2])
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# kp1_np=np.array(kp1)
# for kp in kp2:
#     unpackSIFTOctave(kp)
# print("########################")
# BFMatcher with default params
# bf = cv2.BFMatcher(crossCheck=False)
# matches = bf.knnMatch(des1,des2, k=2)
# matches1=bf.match(des2,des1)
# print(len(matches))
# print(matches[0][0].distance," ",matches[0][1].distance)
# print(matches[1][0].distance," ",matches[1][1].distance)
# print(len(des1)," ",len(des2))
# # Apply ratio test
# good = []
# good1=[]
# for m in matches1:
#     print(m.distance)
#
#     # if m.distance < 0.75*n.distance:
#     #     good.append([m])
#     # if n.distance < 0.75*m.distance:
#     #     good.append([])
# # cv.drawMatchesKnn expects list of lists as matches.
# matching_image1=cv2.drawMatches(img1,kp1,img2,kp2,matches1,None,flags=2)
# #matching_image = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches1,None,flags=2)
#
# show_images([matching_image1])
import socket
import struct

import os
import io
from PIL import Image
from array import array

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def readimage(path):
    count = os.stat(path).st_size / 2
    with open(path, "rb") as f:
        return bytearray(f.read())


address = ("0.0.0.0", 5000)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)
s.listen(1000)

# img = cv2.imread('./rec/tst.jpg',cv2.IMREAD_COLOR)
# print(img.shape)
#
# img=Image.open('./rec/tst.jpg')
# print(img)

client, addr = s.accept()
print ('got connected from', addr)

while True:

    buf = b''
    while len(buf)<4:
        buf += client.recv(4-len(buf))
    size = struct.unpack('!i', buf)
    #print ("receiving bytes:" , size)

    sz=size[0]


    with open('./rec/tst.jpg', 'wb') as img:
        while True:
            data = client.recv(1024)
            sz-=len(data)
            # print(len(data))

            if(sz<=0):
                break

            # data=np.array(data)
            img.write(data)

    print("writeen !!!!")
    bytes = readimage('./rec/tst.jpg')
    image = Image.open(io.BytesIO(bytes))
    image.save('./rec/tsty.jpg')

    # f = open('tst.png', 'rb')
    # image_bytes = f.read()  # b'\xff\xd8\xff\xe0\x00\x10...'
    # print(image_bytes)
    # img=np.array(image_bytes)
    # print(img)

    img = cv2.imread('./rec/tsty.jpg')
    print(img.shape)
    show_images([img])
    # image = Image.open(io.BytesIO(image_bytes))
    # image.save("tsty.png")

    # cv2.imwrite("final.png",img)

    # img=cv2.imread("tst.jpg", cv2.IMREAD_GRAYSCALE)
    # show_images([img])
    print ('received, yay!')

client.close()