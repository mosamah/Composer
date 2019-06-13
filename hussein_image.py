# segmentation
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os

def showImages(images, titles=None, mainTitle=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    if mainTitle is not None: fig.suptitle(mainTitle)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
#########################################################################################
# local thresholding (may not be used in the project)
def localThreshRows(img, w=1):
    imgLocalRows = np.zeros(img.shape, np.uint8)
    for row in range(0, img.shape[0], w):
        _, imgLocalRows[row:row + w, :] = cv2.threshold(img[row:row + w, :], 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgLocalRows
#########################################################################################
def localThreshColumns(img):
    imgLocalColumns = np.zeros(img.shape, np.uint8)
    for col in range(img.shape[1]):
        _, imgLocalColumns[:, col:col + 1] = cv2.threshold(img[:, col:col + 1], 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgLocalColumns
#########################################################################################
def median(noiseImg, w=1):
    medImg = np.copy(noiseImg)
    for x in range(w,noiseImg.shape[0]-w):
        for y in range (w,noiseImg.shape[1]-w):
            medImg[x,y] = np.median(noiseImg[x-w:x+w+1,y])#-w:y+w+1])
    return medImg
#########################################################################################
def dilate(noiseImg, w=1):
    medImg = np.copy(noiseImg)
    for x in range(w,noiseImg.shape[0]-w):
        for y in range (w,noiseImg.shape[1]-w):
            medImg[x,y] = np.max(noiseImg[x-w:x+w+1,y])#-w:y+w+1])
    return medImg
#########################################################################################
def randomColor():
    levels = range(32, 256, 32)
    return tuple(int(random.choice(levels)) for _ in range(3))
#########################################################################################
def hough(img, imgLayer = None, c = (0,255,0)):
    # note that this function is used on images from the IAM database only
    lines = cv2.HoughLinesP(image=img, rho=1, theta=np.pi / 180, threshold=50, minLineLength=int(0.1 * img.shape[1]),maxLineGap=10)
    if imgLayer is None:
        imgLayer = np.zeros((img.shape[0],img.shape[1],3),np.uint8)#np.copy(imgBin)
        imgLayer[:,:,0]=imgLayer[:,:,1]=imgLayer[:,:,2]=img
    horizontals = []  # will contain the y coordinates of the 2 points representing the line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (abs(x1 - x2) >= int(0.1 * img.shape[1] - 1)):  # horizontal lines at the top of the page
            horizontals.append([y1, y1])
            cv2.line(imgLayer,(x1,y1),(x2,y2),color=c)
    return imgLayer
##########################################################################################

#Extracting image
imgPath = "sheets/jingle1.png"#"./clean_musical_sheets/kiss-kiss-bang-bang-page-1.png"
print("(1) Exctacting image", imgPath)

imgOriginal = cv2.imread(imgPath)
showImages([imgOriginal])
# imgOriginal = imgOriginal[300:330,:,:]
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
imgGrayInverted = 255-imgGray
otsuThresh, imgOtsu=cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# otsuThresh, imgBin=cv2.threshold(imgGrayInverted,50,255,cv2.THRESH_BINARY)
imgBin = 255 - imgOtsu

# showImages([imgBin])#,median(imgBin,2),median(imgBin,3)])
# showImages([median(imgBin,1)])
# showImages([median(imgBin,3)])


imgLayerBinRGB = np.zeros((imgBin.shape[0],imgBin.shape[1],3),np.uint8)
imgLayerBinRGB[:,:,0]=imgLayerBinRGB[:,:,1]=imgLayerBinRGB[:,:,2]=imgBin

imgLayerZerosRGB = np.zeros((imgBin.shape[0],imgBin.shape[1],3),np.uint8)

imgLayerBinRGB = hough(imgBin,imgLayerBinRGB,(0,255,0))
imgLayerZerosRGB = hough(imgBin,imgLayerZerosRGB,(0,255,0))
imgRawNotesRGB = imgLayerBinRGB - imgLayerZerosRGB

imgRawNotesBinTemp = cv2.cvtColor(imgRawNotesRGB, cv2.COLOR_BGR2GRAY)


_, imgRawNotesBinTemp=cv2.threshold(imgRawNotesBinTemp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

imgDilRawNotesBinTemp = dilate(imgRawNotesBinTemp,4)
# showImages([imgBin,imgDilRawNotesBinTemp])
imgRawNotes = np.zeros(imgBin.shape,np.uint8)
# showImages([imgRawNotes,imgDilRawNotesBinTemp,imgBin])
# print(imgRawNotes.shape,imgBin.shape,imgDilRawNotesBinTemp.shape)
# print((imgDilRawNotesBinTemp[:,:].astype(bool)))
imgRawNotes[np.bitwise_and(imgDilRawNotesBinTemp.astype(bool),imgBin.astype(bool))]=255
# print(imgRawNotes)
    # return np.max(np.asarray(horizontals)) + 10
# showImages([imgLayerBinRGB])
# showImages([imgLayerZerosRGB])
# showImages([imgRawNotesRGB])

# showImages([imgMedRawNotesBinTemp])
showImages([imgBin,imgDilRawNotesBinTemp,imgRawNotes])

path = r'C:\Users\mosama\PycharmProjects\piano\outputhussein'
cv2.imwrite(os.path.join(path , 'imgRaw.jpg'), imgRawNotes)
cv2.waitKey(0)
# kernel = np.ones((2,2),np.uint8)
# imgEroded = cv2.erode(imgBin,kernel,iterations = 1)
# imgEroded = cv2.erode(imgEroded,kernel,iterations = 1)
# imgDil = cv2.dilate(imgEroded,kernel,iterations = 1)
# imgDil = cv2.dilate(imgDil,kernel,iterations = 1)
# showImages([imgBin, imgEroded, imgDil], ["imgBin", "imgEroded", "imgDil"])
#
#
# ###########################
# def findBoundingRectangleRatio(contours):
#     npCntrs = np.asarray(contours)
#     x, y, w, h = cv2.boundingRect(contours)
#     shapesIdx = np.where((h/w)>=0.5)[0]
#
#
# imgCntrs = np.zeros((imgBin.shape[0],imgBin.shape[1],3),np.uint8)#np.copy(imgBin)
# imgCntrs[:,:,0]=imgCntrs[:,:,1]=imgCntrs[:,:,2]=imgBin
#
# _, cntrs, hierarchy = cv2.findContours(imgBin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #CHAIN_APPROX_NONE
#
# #
#
#
# # for cntr in cntrs:
# # npCntrs = np.asarray(cntrs)
# shapesIdx = []
# # for cntrI in range(len(cntrs)):
# #     x, y, w, h = cv2.boundingRect(cntrs[cntrI])
# #     if w<0.1*imgBin.shape[1]:#h / w) >= 0.2:
# #         shapesIdx.append(cntrI)
#
# # for i, ctr in enumerate(cntrs):
# #     print(i)
# #     x, y, w, h = cv2.boundingRect(ctr)
# #
# #     roi = imgCntrs[y:y + h, x:x + w]
# #
# #     area = w*h
# #
# #     if 30 < area < 50:
# #         rect = cv2.rectangle(imgCntrs, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# # cv2.imshow('rect', imgCntrs)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # cv2.drawContours(imgCntrs, list(np.asarray(cntrs)[shapesIdx]), -1, (0, 255, 0), 2)
#
# cv2.drawContours(imgCntrs, cntrs, -1, (0, 255, 0), 2)
#
# showImages([imgCntrs])
#
# # for crI in range(1,ccNumFilteredBinarized):#loop on connected regions
# #     cv2.drawContours(imgCntrsFilteredBinarized, CRCntrs[crI], -1, (0,255,0), 10)
# # show_images([imgCntrsFilteredBinarized])