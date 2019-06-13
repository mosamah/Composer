from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from PIL import Image
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from sklearn import svm
import pickle

trainPath = "./dataset"
testpath = './testImages'
sizeAfterResize = (30, 30)  # Size of images after resize
jobs = 1
kNeighbours = 3


def show_images(images, titles=None):
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
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def gauss_and_otsu(img):
    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(img,(5,5),0)
    # _,ther = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ther = cv2.bitwise_not(ther)
    t = threshold_otsu(img)
    img[img > t] = 255
    img[img <= t] = 0
    return img



def imageToVectorYousra(image):
    # TODO: Resize and flatten
    # Can Use cv2.resize and then flatten the output
    img_w, img_h = image.size
    maxDim = np.maximum(img_w, img_h)
    background = Image.new('RGBA', (maxDim, maxDim), (0, 0, 0, 0))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(image, offset)
    image = np.array(background)
    image = cv2.resize(image, sizeAfterResize)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = gauss_and_otsu(image)
    # show_images([image])
    # print(image.shape)
    img = image.flatten()
    return img

def imageToVector(image):
    # TODO: Resize and flatten
    # Can Use cv2.resize and then flatten the output
    img_h, img_w = image.shape
    maxDim = np.maximum(img_w, img_h)
    background = np.zeros((maxDim,maxDim),np.uint8)
    bg_h, bg_w = background.shape
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background[offset[1]:offset[1]+img_h,offset[0]:offset[0]+img_w]=image
    # image = np.array(background)
    image = cv2.resize(image, sizeAfterResize)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = gauss_and_otsu(image)

    _,image=cv2.threshold(image,0,255,cv2.THRESH_OTSU,cv2.THRESH_BINARY)
    # show_images([image])

    # print(image.shape)
    img = image.flatten()
    return img
def getSampleFeature(image):

    sampleFeatures = imageToVector(image)
    # sampleFeatures = np.flaflatten(image)

    return sampleFeatures

def initTrainFeaturesYousra(trainPath):
    imagePaths = list(paths.list_images(trainPath))
    # initialize matrices and labels list
    featuresRaw = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = Image.open(imagePath, 'r')
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        rawFeatures = getSampleFeature(image)
        featuresRaw.append(rawFeatures)
        labels.append(label)

    trainFeatures = np.array(featuresRaw)
    trainLabels = np.array(labels)
    with open('trainFeatures.pkl','wb') as output:
        pickle._dump(trainFeatures,output,pickle.HIGHEST_PROTOCOL)
    with open('trainLabels.pkl','wb') as output:
        pickle._dump(trainLabels,output,pickle.HIGHEST_PROTOCOL)
    return trainFeatures,trainLabels

def initTrainFeatures(trainPath):
    imagePaths = list(paths.list_images(trainPath))
    # initialize matrices and labels list
    featuresRaw = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        rawFeatures = getSampleFeature(image)
        featuresRaw.append(rawFeatures)
        labels.append(label)

    trainFeatures = np.array(featuresRaw)
    trainLabels = np.array(labels)
    with open('trainFeatures.pkl','wb') as output:
        pickle._dump(trainFeatures,output,pickle.HIGHEST_PROTOCOL)
    with open('trainLabels.pkl','wb') as output:
        pickle._dump(trainLabels,output,pickle.HIGHEST_PROTOCOL)
    return trainFeatures,trainLabels

def trainRawFeatures():
    with open('trainFeatures.pkl','rb') as input:
        trainFeatures=pickle.load(input)
    with open('trainLabels.pkl','rb') as input:
        trainLabels=pickle.load(input)

    # print("[SVM] Mode = ")
    svm_model = svm.LinearSVC()
    # print("[SVM] Training")

    svm_model.fit(trainFeatures, trainLabels)
    return svm_model




def determineNoteSVM(testImage,svm_model):
    testImages=[]
    testImagesLables=[]
    rawFeatures = getSampleFeature(testImage)
    testImages.append(rawFeatures)
    label = testpath.split(os.path.sep)[-1].split(".")[0]
    testImagesLables.append(label)
  #  acc = svm_model.score(testImages, testImagesLables)
  #  print("[SVM] Accuracy: {:.2f}%".format(acc * 100))
    y_pred = svm_model.predict(testImages)

    return y_pred[0][0]


#training
initTrainFeatures(trainPath)
svm_model=trainRawFeatures()

#testing
testPaths = list(paths.list_images(testpath))
for (i, testpath) in enumerate(testPaths):
    imgCV = cv2.imread(testpath,cv2.IMREAD_GRAYSCALE)
    _,img = cv2.threshold(imgCV,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    # testImage = Image.open(testpath)
    # print(testImage)
    show_images([img])
    pred=determineNoteSVM(img,svm_model)
    print("found: ",pred)









# knnModel = KNeighborsClassifier(n_neighbors=kNeighbours, n_jobs=jobs)
#
# print("[KNN] Training")
# # TODO: Fit the model using trainFeatures and trainLabels
# knnModel.fit(trainFeatures, trainLabels)
# print('get Score')
#
# acc = knnModel.score(testImages, testImagesLables)
# print("[KNN] Accuracy: {:.2f}%".format(acc * 100))
#
# y_pred = knnModel.predict(testImages)
# print(y_pred)
#
# print(knnModel.predict_proba(testImages))


