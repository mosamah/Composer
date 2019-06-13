
# coding: utf-8

# In[13]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

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

def randomColor():
    levels = range(32, 256, 32)
    return tuple(int(random.choice(levels)) for _ in range(3))

def median(img, h=1,w=0):
    medImg = np.copy(img)
    for y in range(h,img.shape[0]-h):
        for x in range (w,img.shape[1]-w):
            medImg[y,x] = np.median(img[y-h:y+h+1,x-w:x+w+1])
    return medImg
def medianPlus(img,h=1,w=1):
    windowWidth = max(h,w)
    windowMask = np.zeros((2*windowWidth+1,2*windowWidth+1),bool)
    windowMask[h,:] = True
    windowMask[:,w] = True

    medImg = np.copy(img)
    for y in range(windowWidth,img.shape[0]-windowWidth):
        for x in range (windowWidth,img.shape[1]-windowWidth):
            medImg[y,x] = np.median(img[y-windowWidth:y+windowWidth+1,x-windowWidth:x+windowWidth+1][windowMask])

    return medImg

def removeHorizontalNoise(img,noiseHeight=1):
    cleanImg = np.copy(img)
    for y in range(noiseHeight,img.shape[0]-noiseHeight):
        row = np.bitwise_and(np.bitwise_not(img[y+1,:]),np.bitwise_not(img[y-1,:]))
        cleanImg[y, :] = np.bitwise_and(row,img[y,:])
    cleanImg = np.bitwise_and(np.bitwise_not(cleanImg),img)
    return cleanImg

randomColors = np.zeros((2500, 3), np.uint8)
for i in range(2500):
    randomColors[i, :] = randomColor()
randomColors[0, :] = [255, 255, 255]

leftIndex, topIndex, widthIndex, heightIndex, areaIndex = range(5)

def CE(grayImg):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgCE = clahe.apply(grayImg)
    return imgCE

def removeNoiseFromAdaptiveThreshold(imgAT):
    ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(imgAT)
    ccMaxAreaIdx = (np.argpartition(-ccStats[:, areaIndex], 2))[:2]
    ccMaxArea = ccStats[ccMaxAreaIdx[1], areaIndex]
    ccRegIdx = np.where(ccStats[:, areaIndex] >= 0.5 * ccMaxArea)[0]
    ccRegIdx = ccRegIdx[ccRegIdx > 0]
    imgBin = np.zeros((imgAT.shape[0], imgAT.shape[1])).astype(np.uint8)
    for i, r in enumerate(ccRegIdx):
        imgBin[ccImg == r] = 255
    return imgBin

def getRegions(imgBin):
    horizontal = np.copy(imgBin)
    horizontalsize = int(horizontal.shape[1] / 40)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontalBase = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal = np.copy(horizontalBase)

    ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(horizontal)
    d=1
    while (ccNum-1)%5 != 0:
        horizontal=cv2.dilate(horizontal, horizontalStructure, (-1, -1))
        d += 1
        ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(horizontal)

    rawNotes = np.bitwise_and(np.bitwise_not(horizontal),imgBin)
    centerLines=[]
    for ccI in range(1,ccNum):
        x1, x2=ccStats[ccI,leftIndex],(ccStats[ccI,leftIndex]+ccStats[ccI,widthIndex]-1)
        y1 = np.average(np.where(ccImg[:,x1]==ccI)[0])
        y2 = np.average(np.where(ccImg[:, x2] == ccI)[0])
        centerLines.append([x1,y1,x2,y2])
    centerLines.sort(key=lambda line: line[1])
    regions = np.split(np.asarray(centerLines), int(len(centerLines) / 5))
    return rawNotes, regions


def getNotesImgs(regions, rawNotes):
    regionStats = []
    for rgn in regions:
        avgDist = np.average(np.average(rgn[1:, [1, 3]] - rgn[0:-1, [1, 3]], axis=0))
        # get box bounding all notes
        x1 = int(np.min(rgn[:, 0]))  # left
        y1 = int(round(np.min(rgn[0, [1, 3]]) - 2 * avgDist))  # top
        x2 = int(np.max(rgn[:, 2]))  # right
        y2 = int(round(np.max(rgn[4, [1, 3]]) + 2 * avgDist))  # bottom
        x1 = max(x1 ,0)
        y1 = max(y1 ,0)
        x2 = min(x2, rawNotes.shape[1] - 1)
        y2 = min(y2, rawNotes.shape[0] - 1)
        rgnImgRawNotes = np.copy(rawNotes[y1:y2, x1:x2])

        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(round(avgDist / 2))))
        rgnImgNotes = cv2.dilate(rgnImgRawNotes, verticalStructure, (-1, -1))
        rgnImgNotes = cv2.dilate(rgnImgNotes, verticalStructure, (-1, -1))

        ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(rgnImgNotes)

        notesIdx = np.arange(start=1, stop=ccNum, dtype=int)
        notesIdx = sorted(notesIdx, key=lambda id: ccCentroids[id, 0])
        notesImgs = []
        for i in notesIdx:
            t = ccStats[i, topIndex]  # y1
            l = ccStats[i, leftIndex]  # x1
            th = ccStats[i, topIndex] + ccStats[i, heightIndex]  # y2
            lw = ccStats[i, leftIndex] + ccStats[i, widthIndex]  # x2
            l -= 1
            lw += 1
            # t-=1
            # th+=1
            # print(x1,y1,x2,y2)
            # print(l,lw,t,th)
            noteImg = np.zeros((th - t, lw - l),np.uint8)
            roi = ccImg[t:th, l:lw]
            print(noteImg.shape,roi.shape)
            noteImg[roi == i] = 255
            notesImgs.append([noteImg, [l, t, lw, th]])
        regionStats.append([[x1, y1, x2, y2], avgDist, notesImgs])
    return regionStats

from scipy import ndimage
def getQuarterCenter(img, avgHeight):
    qNote = np.copy(img)
    kernelWidth = int(round(avgHeight/4))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelWidth, 1))
    eroded = cv2.erode(qNote, kernel, iterations=1)
    ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(eroded,connectivity=4)
    while ccNum-1>1:
        eroded = cv2.erode(eroded, kernel, iterations=1)
        ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(eroded)

    cy, cx = ndimage.measurements.center_of_mass(eroded)
    center = (int(cx), int(cy))
    return center
def x3(img):
    showImages([img],["input"])
    img = cv2.bitwise_not(img)

    imgAdaptiveThresh = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    imgBin = removeNoiseFromAdaptiveThreshold(imgAT=imgAdaptiveThresh)

    rawNotes, regions = getRegions(imgBin)
    rawNotes = removeHorizontalNoise(rawNotes)

    regionStats = getNotesImgs(regions, rawNotes)

    qup = 81;
    qdown = 78;
    hup = 0;
    hdown=0;
    w = 0;
    eup = 1;
    edown=4;
    sup = 0;
    sdown=0;
    bup = 0;
    bdown=3;
    r = 9;
    c = 9;
    v = 189;

    for rgnCrdnts, avgHeight, notes in regionStats:
        for note, noteCrdnts in notes:
            cv2.imshow('Note', note)
            keyPress = cv2.waitKey(0)
            print(keyPress);
            if keyPress == 113:   #q
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/qup.' + str(qup) + '.jpg', note)
                qup = qup + 1;
                # cv2.waitKey(0)
            if keyPress == 97:  #a
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/qdown.' + str(qdown) + '.jpg', note)
                qdown = qdown + 1;
                # cv2.waitKey(0)
            elif keyPress == 104:   #h
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/hup.' + str(hup) + '.jpg', note)
                hup = hup + 1;
                # cv2.waitKey(0)
            elif keyPress == 100:   #d
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/hdown.' + str(hdown) + '.jpg', note)
                hdown = hdown + 1;
                # cv2.waitKey(0)
            elif keyPress == 119:   #w
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/w.' + str(w) + '.jpg', note)
                w = w + 1;
                # cv2.waitKey(0)
            elif keyPress == 101:   #e
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/eup.' + str(eup) + '.jpg', note)
                eup = eup + 1;
                # cv2.waitKey(0)
            elif keyPress == 102:   #f
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/edown.' + str(edown) + '.jpg', note)
                edown = edown + 1;
                # cv2.waitKey(0)
            elif keyPress == 115:   #s
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/sup.' + str(sup) + '.jpg', note)
                sup = sup + 1;
                # cv2.waitKey(0)
            elif keyPress == 103:   #g
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/sdown.' + str(sdown) + '.jpg', note)
                sdown = sdown + 1;
                # cv2.waitKey(0)
            elif keyPress == 98:    #b
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/bup.' + str(bup) + '.jpg', note)
                bup = bup + 1;
                # cv2.waitKey(0)
            elif keyPress == 105:   #i
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/bdown.' + str(bdown) + '.jpg', note)
                bdown = bdown + 1;
                # cv2.waitKey(0)
            elif keyPress == 114:   #r
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/r.' + str(r) + '.jpg', note)
                r = r + 1;
                # cv2.waitKey(0)
            elif keyPress == 99:    #c
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/c.' + str(c) + '.jpg', note)
                c = c + 1;
                # cv2.waitKey(0)
            elif keyPress == 118:   #v
                cv2.imwrite('C:/Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/dataset/v.' + str(v) + '.jpg', note)
                v = v + 1;
                # cv2.waitKey(0)
            cv2.destroyAllWindows()

    # print(sorted(lines,key= lambda x:x[1]))
    # raw = vertical-horizontal2
    # showImages([raw])
    # cv2.imshow("horizontal", horizontal)

    verticalsize = int(rows / 70)
    print(verticalsize)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (51, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    # showImages([vertical], ["vert"])
    # cv2.imshow("vertical", vertical)

    vertical = cv2.bitwise_not(vertical)
    # cv2.imshow("vertical_bitwise_not", vertical)

    #step1
    edges = cv2.adaptiveThreshold(vertical,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,-2)
    # cv2.imshow("edges", edges)

    #step2
    kernel = np.ones((2, 2), dtype = "uint8")
    dilated = cv2.dilate(edges, kernel)
    # cv2.imshow("dilated", dilated)


    # step3
    smooth = vertical.copy()

    #step 4
    smooth = cv2.blur(smooth, (4,4))
    # cv2.imshow("smooth", smooth)

    #step 5
    (rows, cols) = np.where(img == 0)
    vertical[rows, cols] = smooth[rows, cols]

    # cv2.imshow("vertical_final", vertical)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

from Orientation import fixOrientation
imgPath = "C://Users/yousra/Documents/Senior2-Semester1/Image Processing/Project/haget hussein/musical/20181220_185159.jpg"

imgOriginal = cv2.imread(imgPath)
showImages([imgOriginal])
imgGray = fixOrientation(imgOriginal)[int(round(imgOriginal.shape[0]*0.1)):int(round(imgOriginal.shape[0]*0.9)),:]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imgGray = clahe.apply(imgGray)

x3(imgGray)





def x2(img):
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,5))
    print(horizontalStructure)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5))
    print(horizontalStructure)



    showImages([img],["input"])
    img = cv2.bitwise_not(img)
    imgAdaptiveThresh = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    imgBin = removeNoiseFromAdaptiveThreshold(imgAdaptiveThresh)


    vertical = np.copy(imgBin)



    # masked_img = cv2.bitwise_and(img, img, mask=horizontal)

    regionStats = []
    for rgn in regions:
        avgDist = np.average(np.average(rgn[1:,[1,3]]-rgn[0:-1,[1,3]],axis=0))
        #get box bounding all notes
        x1 = int(np.min(rgn[:,0]))                              #left
        y1 = int(round(np.min(rgn[0,[1,3]]) - 2 * avgDist))     #top
        x2 = int(np.max(rgn[:, 2]))                             #right
        y2 = int(round(np.max(rgn[4, [1, 3]]) + 2 * avgDist))   #bottom
        # rgnImgRawNotes = np.copy(rawNotes[y1:y2,x1:x2])
        rgnImgBin = np.copy(imgBin[y1:y2,x1:x2])
        # rgnImgOriginal = np.copy(img[y1:y2,x1:x2])
        # rgnImgAdaptiveThresh = cv2.adaptiveThreshold(rgnImgOriginal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # rgnImgCE = clahe.apply(rgnImgOriginal)
        # showImages([rgnImgBin,rgnImgOriginal,rgnImgAdaptiveThresh],["bin crop","ori","crop adap"])
        #
        # showImages([rgnImgCE,rgnImgOriginal],["ce","ori"])
        # rgnImgCEAdaptiveThresh = cv2.adaptiveThreshold(rgnImgCE, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        #                                              15, -2)
        # showImages([rgnImgBin,rgnImgAdaptiveThresh,rgnImgCEAdaptiveThresh],["bin crop","rgn adap","region ce adap"])
        #
        # ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(rgnImgCEAdaptiveThresh)
        # ccMaxAreaIdx = (np.argpartition(-ccStats[:, areaIndex], 2))[:2]
        # ccMaxArea = ccStats[ccMaxAreaIdx[1], areaIndex]
        # ccRegIdx = np.where(ccStats[:, areaIndex] >= 0.9 * ccMaxArea)[0]
        # ccRegIdx = ccRegIdx[ccRegIdx > 0]
        # rgnImgCEAdaptiveThreshNoNoise = np.zeros((rgnImgCEAdaptiveThresh.shape[0], rgnImgCEAdaptiveThresh.shape[1])).astype(np.uint8)  # todo: zawed dilate
        # for i, r in enumerate(ccRegIdx):
        #     rgnImgCEAdaptiveThreshNoNoise[ccImg == r] = 255
        # showImages([rgnImgCEAdaptiveThresh,rgnImgCEAdaptiveThreshNoNoise],["rgnImgCEAdaptiveThresh","minus noise"])

        showImages([rawNotes[y1:y2,x1:x2],rgnImgBin],["raw","bin img"])
        showImages([notes2], ["double dilation"])
        print(avgDist)
        for i in range(100):
            notes2=cv2.erode(notes2, k, (-1, -1))
            showImages([notes2], [i])
        # showImages([m1],["median"])


        # m1 = medianPlus(notes, h=2, w=1)
        # m2 = medianPlus(notes, h=2, w=0)
        # m3 = medianPlus(notes, h=2, w=2)
        # showImages([m1, m2, m3])
        # print(verticalSize)

    # print(sorted(lines,key= lambda x:x[1]))
    # raw = vertical-horizontal2
    # showImages([raw])
    # cv2.imshow("horizontal", horizontal)

    verticalsize = int(rows / 70)
    print(verticalsize)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (51, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    # showImages([vertical], ["vert"])
    # cv2.imshow("vertical", vertical)

    vertical = cv2.bitwise_not(vertical)
    # cv2.imshow("vertical_bitwise_not", vertical)

    #step1
    edges = cv2.adaptiveThreshold(vertical,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,-2)
    # cv2.imshow("edges", edges)

    #step2
    kernel = np.ones((2, 2), dtype = "uint8")
    dilated = cv2.dilate(edges, kernel)
    # cv2.imshow("dilated", dilated)


    # step3
    smooth = vertical.copy()

    #step 4
    smooth = cv2.blur(smooth, (4,4))
    # cv2.imshow("smooth", smooth)

    #step 5
    (rows, cols) = np.where(img == 0)
    vertical[rows, cols] = smooth[rows, cols]

    # cv2.imshow("vertical_final", vertical)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

