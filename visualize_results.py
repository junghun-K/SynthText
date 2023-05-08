# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import h5py 
from common import *
import math



def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    H,W = text_im.shape[:2]

    # plot the character-BB:
    for i in range(len(charBB_list)):
        bbs = charBB_list[i]
        ni = bbs.shape[-1] # num. of char
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            # print(bb)
            plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # plot the word-BB:
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])        

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)

def exchange_points(bounding_boxes):
    """
    Exchange the x and y-coordinate values of point 2 and point 4 in the bounding box coordinates.

    Parameters:
    bounding_boxes: numpy array, shape (2, 4, n)
        Transformed bounding box coordinates obtained from homographyBB function.

    Returns:
    numpy array, shape (2, 4, n)
        Bounding box coordinates with exchanged x and y-coordinate values of point 2 and point 4.
    """
    exchanged_boxes = bounding_boxes.copy()
    exchanged_boxes[0, 1, :] = bounding_boxes[0, 3, :]  # Exchange x-coordinate of point 2 with point 4
    exchanged_boxes[0, 3, :] = bounding_boxes[0, 1, :]  # Exchange x-coordinate of point 4 with point 2

    exchanged_boxes[1, 1, :] = bounding_boxes[1, 3, :]  # Exchange y-coordinate of point 2 with point 4
    exchanged_boxes[1, 3, :] = bounding_boxes[1, 1, :]  # Exchange y-coordinate of point 4 with point 2

    return exchanged_boxes

def main(db_fname):
    db = h5py.File(db_fname, 'r+')
    dsets = sorted(db['data'].keys())
    print ("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        wordBB = exchange_points(wordBB)
        
        txt = db['data'][k].attrs['txt']
        viz_textbb(rgb, [charBB], wordBB)
        print ("image name        : ", colorize(Color.RED, k, bold=True))
        print ("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        print ("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        print ("  ** text         : ", colorize(Color.GREEN, txt))

        if 'q' in input("next? ('q' to exit) : "):
            break
    db.close()

if __name__=='__main__':
    main('results/SynthText.h5')

