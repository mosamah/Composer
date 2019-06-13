
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

def removeHorizontalNoise(img, noiseHeight=2):

    # cleanImg = np.copy(img)
    # for i in range(1,noiseHeight+1):
    cleanImg = removeHorizontalNoiseExact(img,noiseHeight)#(cleanImg,i)

    return cleanImg

def removeHorizontalNoiseExact(img,noiseHeight=1):
    cleanImg = np.zeros(img.shape,np.uint8)
    notImg = np.bitwise_not(img)
    for y in range(0,img.shape[0]-noiseHeight-2):
        borderRows = np.bitwise_and(notImg[y,:],notImg[y+noiseHeight+1,:])
        noiseRows = img[y+1:y+noiseHeight+1,:]
        conditionNoiseTemp = (np.sum(((noiseRows==255).astype(int)),axis=0))<=noiseHeight#((np.sum(noiseRows,axis=0,dtype=int))==noiseHeight*255)
        conditionNoise = np.zeros((1,img.shape[1]),np.uint8)
        conditionNoise[0,conditionNoiseTemp]=255
        cleanImg[y + 1:y + noiseHeight + 1, :] = np.bitwise_or(np.bitwise_and(conditionNoise,borderRows),cleanImg[y + 1:y + noiseHeight + 1, :])

    cleanImg = np.bitwise_and(np.bitwise_not(cleanImg),img)
    return cleanImg


randomColors = np.zeros((100000, 3), np.uint8)
for i in range(100000):
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
def removeNoiseFromAdaptiveThreshold2(imgAT):
    ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(imgAT)
    ccMaxAreaIdx = (np.argpartition(-ccStats[:, areaIndex], 2))[:2]
    ccMaxArea = ccStats[ccMaxAreaIdx[1], areaIndex]
    ccRegIdx = np.where(ccStats[:, areaIndex] >= 0.5 * ccMaxArea)[0]
    ccRegIdx = ccRegIdx[ccRegIdx > 0]

    imgBin = np.zeros((imgAT.shape[0], imgAT.shape[1])).astype(np.uint8)
    imgBlur = np.zeros((imgAT.shape[0], imgAT.shape[1])).astype(np.uint8)
    imgThresh = np.zeros((imgAT.shape[0], imgAT.shape[1])).astype(np.uint8)
    # ccSmallIdx = np.asarray([])
    for ccI in ccRegIdx:


        t = ccStats[ccI, topIndex]  # y1
        l = ccStats[ccI, leftIndex]  # x1
        th = ccStats[ccI, topIndex] + ccStats[ccI, heightIndex]  # y2
        lw = ccStats[ccI, leftIndex] + ccStats[ccI, widthIndex]  # x2

        ccNum2, ccImg2, ccStats2, ccCentroids2 = cv2.connectedComponentsWithStats(imgAT[t:th,l:lw],connectivity=4)

        # test
        # bin = np.zeros((th-t, lw-l, 3), np.uint8)
        # bin[:, :] = randomColors[ccImg[t:th,l:lw]]

        inCCsHeightThresh = ccStats2[:,heightIndex]>=int(ccImg2.shape[0]*3/16)#todo zawed another threshold "ORing"
        inCCsWidthThresh = ccStats2[:, widthIndex] >= int(ccImg2.shape[1]*0.05)
        inCCsThresh = np.logical_or(inCCsHeightThresh,inCCsWidthThresh)
        inCCsThresh = np.where(inCCsThresh)[0]
        inCCsThresh=inCCsThresh[inCCsThresh>0]

        for inCC in inCCsThresh:
            imgBin[t: th, l: lw][ccImg2==inCC]=255
        sd = int((th-t)/8)
        kernelSize = int((th-t)/20)
        if(kernelSize%2)==0:
            kernelSize+=1
        imgBlur[t: th, l: lw] = cv2.GaussianBlur(imgBin[t: th, l: lw],(kernelSize,kernelSize),sd)
        _, imgThresh[t: th, l: lw] = cv2.threshold(imgBlur[t: th, l: lw], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # showImages([imgBin,imgBlur,imgThresh])

    return imgThresh, ccRegIdx.shape[0]
def getRegions(imgBin, rgnCount):
    horizontal = np.copy(imgBin)
    horizontalsize = int(horizontal.shape[1] / 35)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    #test
    # horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontalsize*0.5), 1))
    # horizontal = cv2.dilate(horizontal,horizontalStructure2,(-1,-1))


    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontalBase = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal = np.copy(horizontalBase)

    ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(horizontal)
    d=1
    while (ccNum-1) % 5!=0:#!= rgnCount:
        horizontal=cv2.dilate(horizontal, horizontalStructure, (-1, -1))
        d += 1
        if d>=15:
            raise Exception('Could not extract horizontal lines, dilating for '+str(d))
        ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(horizontal)


    #test
    test = np.zeros((imgBin.shape[0],imgBin.shape[1],3),np.uint8)
    test[:,:]=randomColors[ccImg]
    showImages([imgBin,test])

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

def mergeRects(corners,centers):
    x1 = min(corners[:,0])
    y1 = min(corners[:,1])
    x2 = max(corners[:,2])
    y2 = max(corners[:,3])
    areas = (corners[:,2]-corners[:,0])*(corners[:,3]-corners[:,1])
    ci = np.argmax(areas)
    return [x1, y1, x2, y2], centers[ci]

def getNotesImgs(regions, rawNotes):
    regionStats = []
    # newCCNum = 0
    for iRegion,rgn in enumerate(regions):
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
        # notesImgs = []
        notesCrdnts = []
        notesCenters = []
        notesParents = np.asarray(list(range(0, ccNum-1)))
        for i in notesIdx:
            t = ccStats[i, topIndex]  # y1
            l = ccStats[i, leftIndex]  # x1
            th = ccStats[i, topIndex] + ccStats[i, heightIndex]  # y2
            lw = ccStats[i, leftIndex] + ccStats[i, widthIndex]  # x2
            # t -=1
            # l -=1
            # th+=1
            # lw+=1
            # t  = max(0,t )
            # l  = max(0,l )
            # th = min(ccImg.shape[0]-1,th)
            # lw = min(ccImg.shape[1]-1,lw)
            noteImg = np.zeros((th - t, lw - l),np.uint8)
            roi = ccImg[t:th, l:lw]
            noteImg[roi == i] = 255
            center = getQuarterCenter(noteImg, avgDist)
            # notesImgs.append([noteImg])
            notesCrdnts.append([l, t, lw, th])
            notesCenters.append([center[0],center[1]])
        notesCrdnts = np.asarray(notesCrdnts)
        notesCenters = np.asarray(notesCenters)
        for i in range(0,ccNum-1):
            #for merging
            xDiff = notesCrdnts[i,0]-notesCrdnts[:, 2]
            distThresh=0.5*avgDist
            children = xDiff<=distThresh
            children = np.logical_and(children, xDiff >= -(distThresh+notesCrdnts[i,2]-notesCrdnts[i, 0]))
            children = np.where(children)[0]
            for child in children:
                notesParents[notesParents==notesParents[child]] = i #union

        parentsIdx = np.unique(notesParents)
        # newCCNum+=parentsIdx.shape[0]
        groupCrdnts = []
        groupCenters = []
        # newIdx = np.asarray(list(range(0,parentsIdx.shape[0])))
        newCCImg = np.zeros(ccImg.shape,int)
        for i,parent in enumerate(parentsIdx):
            groupIdx = np.where(notesParents==parent)[0] #get children
            # print(grou)
            groupIdx = np.hstack([groupIdx,np.asarray([parent])])
            # groupIdx.vstack(np.asarray([parent]))
            #hat el crdnts wel centers we append el parent
            subGroupCrdnts = notesCrdnts[groupIdx]
            subGroupCenters = notesCenters[groupIdx]

            #hat el merged
            subGroupCrdnt, subGroupCenter = mergeRects(subGroupCrdnts, subGroupCenters)
            groupCenters.append(subGroupCenter)
            groupCrdnts.append(subGroupCrdnt)
            #zabat ccimg
            for childI in groupIdx[:-1]:
                roiCCImg = ccImg[notesCrdnts[childI,1]:notesCrdnts[childI,3], notesCrdnts[childI,0]:notesCrdnts[childI,2]]
                roiNewCCImg = newCCImg[notesCrdnts[childI,1]:notesCrdnts[childI,3], notesCrdnts[childI,0]:notesCrdnts[childI,2]]
                roiNewCCImg[roiCCImg==notesIdx[childI]]=i+1





        #notesImgs contains sub note regions

        regionStats.append([[x1, y1, x2, y2], newCCImg, avgDist, np.asarray(groupCrdnts), np.asarray(groupCenters)])
    return regionStats

from scipy import ndimage
def getQuarterCenter(img, avgHeight):
    qNote = np.copy(img)#todo: law note head bas
    kernelWidth = int(round(avgHeight/4))-1
    height = qNote.shape[0]
    iter1 = 0
    while height>=0.75*qNote.shape[0]:#sometimes the one connected component is the whole note so we need to increase kernel size
        kernelWidth+=1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelWidth, 1))
        eroded = cv2.erode(qNote, kernel, iterations=1)
        ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(eroded,connectivity=4)
        iter2=1
        while ccNum-1>1:
            eroded = cv2.erode(eroded, kernel, iterations=1)
            ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(eroded)
            iter2+=1
            if iter2>=20:
                return(-1,-1)
        if ccNum==1:
            return (-1,-1)
        height = ccStats[1,heightIndex]
        iter1+=1
        if iter1>= 20:
            return (-1,-1)
    cy, cx = ndimage.measurements.center_of_mass(eroded)
    center = (int(cx), int(cy))
    return center

from Intt.stnotes import getNoteLetter
'''
c -> cleff
q -> quarter
h -> half
w -> whole
e -> eighth
b -> bar
r -> rest
x -> undefined
'''
noteTypes = ['c','q','h','w','e','b','r','x']
def x3(img):
    # showImages([img],["input"])
    img = cv2.bitwise_not(img)
    imgAdaptiveThresh = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    imgBin, rgnCount = removeNoiseFromAdaptiveThreshold2(imgAT=imgAdaptiveThresh)

    #test
    # _,imgThresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    # showImages([img,imgAdaptiveThresh,imgBin])
    # imgBin=imgThresh

    rawNotesTemp, regions = getRegions(imgBin, rgnCount)
    rawNotes = removeHorizontalNoise(rawNotesTemp,3)

    showImages([rawNotes,rawNotesTemp])
    regionStats = getNotesImgs(regions, rawNotes)

    #[[x1, y1, x2, y2], newCCImg, avgDist, groupCrdnts, groupCenters])
    for rgnCrdnts, rgnCCImg, avgHeight, maskCrdnts, maskCenters in regionStats:
        rgnImgRGB = np.zeros((rgnCrdnts[3]-rgnCrdnts[1],rgnCrdnts[2]-rgnCrdnts[0],3),np.uint8)
        rgnImgRGB[:,:]=randomColors[rgnCCImg]

        rgnImgAT = imgAdaptiveThresh[rgnCrdnts[1]:rgnCrdnts[3],rgnCrdnts[0]:rgnCrdnts[2]]
        rgnImgBin = imgBin[rgnCrdnts[1]:rgnCrdnts[3],rgnCrdnts[0]:rgnCrdnts[2]]
        showImages([rgnImgRGB])

        numMasks = maskCrdnts.shape[0]
        for maskI in range(0,numMasks):
            roi = rgnCCImg[maskCrdnts[maskI,1]:maskCrdnts[maskI,3],maskCrdnts[maskI,0]:maskCrdnts[maskI,2]]
            roi[roi==maskI+1]=255

            roi = np.bitwise_and(roi,rgnImgAT[maskCrdnts[maskI,1]:maskCrdnts[maskI,3],maskCrdnts[maskI,0]:maskCrdnts[maskI,2]])
            roi = np.bitwise_and(roi, rgnImgBin[maskCrdnts[maskI, 1]:maskCrdnts[maskI, 3],
                                        maskCrdnts[maskI, 0]:maskCrdnts[maskI, 2]])
            roi = removeHorizontalNoise(roi)
            
        # for note, noteCrdnts in notes:
        #     cv2.imshow('Note Mask', note)
        #     cv2.imshow("AT",np.bitwise_and(rgnImgAT[noteCrdnts[1]:noteCrdnts[3],noteCrdnts[0]:noteCrdnts[2]],note))
        #     cv2.imshow("Bin", np.bitwise_and(rgnImgBin[noteCrdnts[1]:noteCrdnts[3], noteCrdnts[0]:noteCrdnts[2]], note))
        #     # showImages([note])
        #     keyPress = cv2.waitKey(0)
        #     if keyPress==32 :
        #         note3d = np.zeros((note.shape[0],note.shape[1],3),np.uint8)
        #         note3d[:,:,0]=note3d[:,:,1]=note3d[:,:,2]=note
        #         center = getQuarterCenter(note,avgHeight)
        #         print(center)
        #         cv2.circle(note3d, center, 1, (0, 0, 255), 2)
        #         cv2.imshow('centr', note3d)
        #         cv2.waitKey(0)
        #     cv2.destroyAllWindows()

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
# imPath = "../Salma/sama/test1.jpg"
# imgPath = "../Salma/musical/6.jpg"
imgPath = "../Salma/jb2/4.jpg"
# imgPath = "../Salma/jb/5.jpg"
imgOriginal = cv2.imread(imgPath)
showImages([imgOriginal])
# returnedImg = fixOrientation(imgOriginal)
# cv2.imwrite('D:/Fall 2018/Image Processing/project/integration/Salma/jbSaved/out.jpg', returnedImg)
imgGray = fixOrientation(imgOriginal)[int(round(imgOriginal.shape[0]*0.1)):int(round(imgOriginal.shape[0]*0.9)),:]
showImages([imgGray])
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

