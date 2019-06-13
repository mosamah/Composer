import glob
import os

import cv2
import skimage.io as io
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
from midiutil import MIDIFile


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


def determine_note(note,standard_notes,standard_note_names):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    max_match_cnt=0.0
    min_match_dis=math.inf
    max_ratio=0
    note_kp, note_des = sift.detectAndCompute(note,None)
    loop_count=0
    # print("note unknown len: ",len(note_des))
    final_note=''
    found_name=''
    for st_note in standard_notes:
        # find the keypoints and descriptors with SIFT
        st_note_kp, st_note_des = sift.detectAndCompute(st_note,None)


        # Apply ratio test
        good = []
        goodDis=[]
        match_cnt=0
        good.clear()
        goodDis.clear()
        # print("############New Note#################")

        if st_note_des is not None and len(st_note_des)>=2:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(note_des,st_note_des, k=2)
            for m,n in matches:
                if m.distance < 0.9*n.distance:
                    good.append([m])
                    goodDis.append(m.distance)
                    match_cnt+=1
            print("############New Note#################")
            print(standard_note_names[loop_count])
            print(goodDis)
            print("upcount: ",match_cnt)



            if match_cnt>max_match_cnt:
                max_match_cnt=match_cnt
                min_match_dis=min(goodDis)
                final_note=st_note
                final_matches_mask=good
                final_note_kp=st_note_kp
                final_note_des=st_note_des
                found_name= standard_note_names[loop_count]
            elif match_cnt==max_match_cnt:
                if  len(goodDis)!=0 and  min(goodDis)<min_match_dis:
                    min_match_dis=min(goodDis)
                    final_note=st_note
                    final_matches_mask=good
                    final_note_kp=st_note_kp
                    final_note_des=st_note_des
                    found_name= standard_note_names[loop_count]
        loop_count+=1
    if final_note !='':
        matching_image = cv2.drawMatchesKnn(note,note_kp,final_note,final_note_kp,final_matches_mask,None,flags=2)
        show_images([matching_image])

    # show_images([note,final_note],["unknown","found"])
    return final_note,found_name

def binarize(img):
    t=threshold_otsu(img)
    img[img>t]=255
    img[img<=t]=0
    return img

def gauss_and_otsu(img):
    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(img,(5,5),0)
    _,ther = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ther = cv2.bitwise_not(ther)
    return ther

def read_images_from_file(img_dir,unKnown):
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)

    imgs=[]
    img_names=[]
    unName=''
    for f1 in files:
        #read img
        if f1 != unKnown:
            img = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)
            #binarize img
            # img=gauss_and_otsu(img)
            splitted=f1.split('.')
            splitted=splitted[0].split("\\")
            imgs.append([img,splitted[1]])
        else:
            splitted=f1.split('.')
            splitted=splitted[0].split("\\")
            unName=splitted[1]
    imgs=np.array(imgs)
    return imgs,unName

##############################Music play################################
note_defs = { ("A4", 69),
             ("B4",71),
             ("C4",60),
             ("D4",62),
             ("E4",64),
             ("F4",65),
             ("G4",67)}

def playMidi(notes):
    track    = 0        #track: Each track is a list of messages
    channel  = 0
    time     = 0    # In beats
    volume   = 100  # 0-127, as per the MIDI standard
    tempo    = 120   # In BPM (beats per minute)
    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)

    MyMIDI.addTempo(track, time, tempo)

    for note_name,note_type in notes:
        print(note_name,note_type)
        for note_midi in note_defs:
            if(note_name == note_midi[0]):
                midi_num = note_midi[1]
                break

        if(note_type == "whole"):
            duration = 4
        elif (note_type == "half"):
            duration = 2
        elif (note_type == "quarter"):
            duration = 1
        elif(note_type == "eigth"):
            duration = 0.5
        elif(note_type == "sixteenth"):
            duration = 0.25

        MyMIDI.addNote(track, channel, midi_num, time, duration, volume)
        time += duration

    with open("JingleBells.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

##############################Main####################################

#show_images(standard_notes,standard_note_names)
data_path = os.path.join("notes/std_y",'*g')
files = glob.glob(data_path)
tcnt=0
fail=0
passed=0
for f1 in files:
    stnotes,unName=read_images_from_file("notes/std_b",f1)
    standard_notes=stnotes[:,0]
    show_images(standard_notes)
    standard_note_names=stnotes[:,1]
    unknown_note=cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    #unknown_note=gauss_and_otsu(unknown_note)
    show_images([unknown_note])
    _,found_name=determine_note(unknown_note,standard_notes,standard_note_names)
    print("found: ",found_name," expected: ",unName)
    tcnt+=1
    if(found_name==unName):
        passed+=1
    else:
        fail+=1
    print("test cnt: ",tcnt)
    print("fail: ",fail)
    print("pass: ",passed)
    print("acc: ",(passed/tcnt) *100)






