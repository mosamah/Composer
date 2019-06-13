
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

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

import pickle

#y_lines are 5 lines of note -- y_note is of center of note
def return_note_letter_clef(y_lines,y_note):
    #initialize lookup lists of notes to be determined
    line_letters=['A','F','D','B','G','E','C']
    space_letters=['B','G','E','C','A','F','D','B']

    #compute average distance if distances between lines arenot equal
    distances=[]
    for i in range(0,(len(y_lines)-1)):
        distances.append(y_lines[i+1]-y_lines[i])
    avg_dist=int(sum(distances)/len(distances))

    #all possible lines to write on or between for notes
    virtual_7_lines=[]
    virtual_7_lines.append(y_lines[0]-avg_dist)
    virtual_7_lines+=y_lines
    virtual_7_lines.append(y_lines[len(y_lines)-1]+avg_dist)
    #print(virtual_7_lines)

    #search for y_note
    note_down=0
    for y_line in virtual_7_lines:
        if y_note<=y_line:
            break
        note_down+=1


    if note_down==0:
        note_down_upper_bound=virtual_7_lines[note_down]-math.ceil(0.2*avg_dist)
        if y_note<note_down_upper_bound:
            #above first virtual line
            result=space_letters[note_down]
        else:
            #on first line
            result=line_letters[note_down]
    elif note_down==7:
        note_down_lower_bound=virtual_7_lines[note_down-1]+math.ceil(0.25*avg_dist)
        if y_note>note_down_lower_bound:
            result=space_letters[note_down]
        else:
            result=line_letters[note_down-1]
    else:
        y_line_up=virtual_7_lines[note_down-1]
        note_up_upper_bound=y_line_up-math.ceil(0.25*avg_dist)
        note_up_lower_bound=y_line_up+math.ceil(0.25*avg_dist)

        y_line_down=virtual_7_lines[note_down]
        note_down_upper_bound=y_line_down-math.ceil(0.25*avg_dist)
        note_down_lower_bound=y_line_down+math.ceil(0.25*avg_dist)

        if y_note>=note_up_upper_bound and y_note<=note_up_lower_bound:
            result=line_letters[note_down-1]
        elif y_note>=note_down_upper_bound and y_note<=note_down_lower_bound:
            result=line_letters[note_down]
        else:
            result=space_letters[note_down]
    return result
ys=[100,200,300,400,500]
note_y=176

r=return_note_letter_clef(ys,note_y)
print(r)