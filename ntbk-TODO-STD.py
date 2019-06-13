
# coding: utf-8

# In[192]:


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


path =  "./dataset"
mode =0 # There are 3 modes for this experiment: mode 0: Raw Pixels, mode 1: Histogram, mode2: HoG
t_size = 0.05 #test_Size
r_state = 42 #Random Seed
sizeAfterResize=(200,200) #Size of images after resize


# In[193]:


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


def gauss_and_otsu(img):
    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(img,(5,5),0)
    # _,ther = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ther = cv2.bitwise_not(ther)
    t=threshold_otsu(img)
    img[img>t] = 255
    img [img<=t] = 0
    return img


# In[197]:


# Features Functions

    
def imageToVector(image):
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

    
def extractHog(image):
    image = cv2.resize(image, sizeAfterResize)
    ''' for 100x100 images
    winSize = (100,100) # Image Size
    cellSize = (5,5) #Size of one cell    
    blockSizeInCells = (5,5)# will be multiplies by No of cells
    
    comment.
    '''
    winSize = (32,32) # Image Size
    cellSize = (4,4) #Size of one cell    
    blockSizeInCells = (2,2)# will be multiplies by No of cells

    blockSize=(blockSizeInCells[1] * cellSize[1], blockSizeInCells[0] * cellSize[0])
    blockStride=(cellSize[1], cellSize[0])
    nbins = 9 #Number of orientation bins
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins) # 
    h = hog.compute(image)
    h = h.flatten()
    return h.flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    # TODO: 
    # Resize: cv2.resize
    img = cv2.resize(image, sizeAfterResize)
    # Convert from BGR2HSV: cv2.cvtColor
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Get Histogram: use cv2.,cv2.ist (read the help), make the histogram on the 3 channels(HSV)
    #                use the parameter bins passed to the functions.
    #                use the following ranges for HSV [0,180,0,256,0,256] .
    # put the histogram in variable named hist
    ccNum, ccImg, ccStats, ccCentroids = cv2.connectedComponentsWithStats(img)
#     h = max(ccStats[1:,3])
    
    if ccStats[1:,2].shape[0]!=0:
        w = max(ccStats[1:,2])
    else:
        w= 10
    if ccStats[1:,3].shape[0]!=0:
        h = max(ccStats[1:,3])
    else:
        h=10
    return (np.asarray([w/h*100])).flatten()

# help(cv2.calcHist)


# In[198]:


def getSampleFeature(image, mode):
    # TODO: Complete this function to get sample features based on chosen mode.
    # Description of Mode variable is written at the first cell
    if mode == 0:
        sampleFeatures = imageToVector(image)
    if mode == 1:
        sampleFeatures = extract_color_histogram(image)
    if mode ==2:
        sampleFeatures = extractHog(image)
    return sampleFeatures


# In[ ]:



jobs = 1
kNeighbours = 3


imagePaths = list(paths.list_images(path))
print(len(imagePaths))

# initialize matrices and labels list

featuresRaw = []

featuresHist = []

featuresHog = []



labels = []

for (i, imagePath) in enumerate(imagePaths):
    # Load image and labels 
    # path format: path/{class}.{image_num}.jpg
    
    image = Image.open(imagePath, 'r')

    # print(image.shape)
    # show_images([image])
    
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    '''
    We will use raw pixels, color histogram and hog.
    1) Extract the features:
    --------------------
     A)Extract raw pixel intensities
     or 
     B)Extract histogram to characterize the color distribution of the pixels
     or
     C)Extract Hog Features after resizing the original image to 100x100
     
    # Add feature to corresonding matrix using matrix.append(sample feature)
    
    
    '''
    
    
    
    
    # image: contains the image matix
    # label: contains the label of the image (retrieved from the path of the image itself)
    #TODO: 
    # 1) Get Features for the image (sampleFeatures)
          
    rawFeatures = getSampleFeature(image, 0)
   # histFeatures = getSampleFeature(image, 1)
   # hogFeatures = getSampleFeature(image, 2)
  
    # 2) Add sampleFeatures to features List
    featuresRaw.append(rawFeatures)
   # featuresHog.append(hogFeatures)
   # featuresHist.append(histFeatures)
    

    # 3) Add label to labels list.
    labels.append(label)
    
    
    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))


# In[ ]:


# Test on chosen images:
# -----------------------
testpath = './testImages'
testPaths = list(paths.list_images(testpath))
print(len(testPaths))
testImages = []
testImagesLables = []
for (i, testpath) in enumerate(testPaths):
    testImage =Image.open(testpath)
    rawFeatures = getSampleFeature(testImage, 0)
    testImages.append(rawFeatures)
    label = testpath.split(os.path.sep)[-1].split(".")[0]
    testImagesLables.append(label)

    
    
    

#image1Path = 'qdown.51.jpg'
#image2Path = 'test5.jpg'

# TODO: read each path of image1Path and image2Path
# 1) Read the image using cv2.imread
# 2) getSampleFeature

#image1 = cv2.imread(image1Path)
#image2 = cv2.imread(image2Path)

#rawFeatures1 = getSampleFeature(image1, 0)
#histFeatures1 = getSampleFeature(image1, 1)
#hogFeatures1 = getSampleFeature(image1, 2)


#rawFeatures2 = getSampleFeature(image2, 0)
#histFeatures2 = getSampleFeature(image2, 1)
#hogFeatures2 = getSampleFeature(image2, 2)


# TODO: Add the features of the 2 images to an array called testImages
#testImages = [rawFeatures1]

#print (rawFeatures1.shape)

COMMENT='''
For single images:
-------------------
predict_proba and predict expects the parameter to be 2D array, each row presents a sample to predict
To convert the image to 2D array we use  array.reshape(-1, 1)
image= image.reshape(-1, 1)
The first parameter is the probability of Cat the second probability is for dog
'''


# In[ ]:


features = np.array(featuresRaw)
labels = np.array(labels)
# Memory Info
print("[MEM INFO] pixels matrix: {:.2f}MB".format(
    features.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits

(trainFeatures, testFeatures, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=t_size, random_state=r_state)


# In[ ]:



from sklearn import svm
print("[SVM] Mode = ",mode)
lin_model = svm.LinearSVC()
print("[SVM] Training")

lin_model.fit(trainFeatures, trainLabels)
print('get Score')

# acc = lin_model.score(testImages, testImagesLables)
# print("[SVM] Accuracy: {:.2f}%".format(acc * 100))
y_pred = lin_model.predict(testImages)
print(testImagesLables)
print(y_pred)

# The following line get the probability for each test image to 
# be in the learnt classes. It isn't applicable to SVM 
# but can be used in the following requirements

# print (lin_model.predict_proba(testImages))


# In[ ]:


# Train & Evaluate
print("[KNN] Mode = ",mode)
# TODO: Initialize a KNeighborsClassifier using k = kNeighbours and number of jobs= jobs
knnModel = KNeighborsClassifier(n_neighbors=kNeighbours,n_jobs = jobs)

print("[KNN] Training")
# TODO: Fit the model using trainFeatures and trainLabels
knnModel.fit(trainFeatures, trainLabels)
print('get Score')

acc = knnModel.score(testImages, testImagesLables)
print("[KNN] Accuracy: {:.2f}%".format(acc * 100))

y_pred = knnModel.predict(testImages)
print(y_pred)

print (knnModel.predict_proba(testImages))


# In[ ]:



from sklearn.neural_network import MLPClassifier
print("[NN] Mode = ",mode)
# TODO: Initialize a MLPClassigier alpha= 1e-5, 
# solver:'sgd'
# Hidden_layer_sizes: 500. 
# random_state: 100.
# max_iter=2000
# verbose =1 To see the progress
# You can try more number of hidden layers. each layer is written as a value in hidden_layer_sizes, 
# layers are separarted by commas
mlpModel = MLPClassifier(alpha= 1e-5,solver='sgd',hidden_layer_sizes=1000,random_state= 42,max_iter=2000,verbose =1)
# Other Trials: 
# 1) hidden_layers = 15.
# 2) Other random_state (42).
# 3) More max_iter.


# TODO: Fit the model using trainFeatures and trainLabels
print("[NN] Training")
mlpModel.fit(trainFeatures, trainLabels)

# TODO: Get the accuracy of the model using model.score(). Pass testFeatures and testLabes as parameters.
print('get Score')

acc = mlpModel.score(testImages, testImagesLables)

print("[NN] Accuracy: {:.2f}%".format(acc * 100))

y_pred = mlpModel.predict(testImages)
print(y_pred)

print (mlpModel.predict_proba(testImages))


