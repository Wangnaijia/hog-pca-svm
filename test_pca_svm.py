from sklearn.externals import joblib
import glob
import os
import time
from config import *
import numpy as np
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage import color
import cv2

m = 125
n = 500

def sliding_window(image, window_size, step_size):
    """
    This function returns a patch of the input 'image' of size
    equal to 'window_size'. The first image returned top-left
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    :param image: Input
    :param window_size: Size of sliding window
    :param step_size: incremented size of window
    :return: a tuple (x, y, im_window)
    """
    for y in xrange(0,image.shape[0], step_size[1]):
        for x in xrange(0,image.shape[1], step_size[0]):
            yield (x, y, image[y: y+window_size[1], x:x+window_size[0]])


def detector(foldername):
    filenames = glob.iglob(os.path.join(foldername, '*'))
    # pca model
    fd_name = "svm_" + str(m) + "_pca_" + str(n) + ".model"
    clf = joblib.load(os.path.join(model_path, fd_name))
    # no pca
    # clf = joblib.load(os.path.join(model_path, 'svm_new.model'))

    num = 0
    total = 0
    for filename in filenames:
        im = cv2.imread(filename)
        #im = imutils.resize(im, width=min(400, im.shape[1]))
        total += 1
        min_wdw_sz = (64, 128)
        step_size = (10, 10)
        downscale = 1.25
        # time cost including hog feature extraction
        t0 = time.time()
        # list to store the detections
        detections = []
        # current scale of image
        scale = 0
        for im_scaled in pyramid_gaussian(im, downscale=downscale):
            # The list contains detections at the current scale
            if im_scaled.shape[0]<min_wdw_sz[1] or im_scaled.shape[1]<min_wdw_sz[0]:
                break
            for (x,y,im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue

                # get feature of test
                im_window = color.rgb2gray(im_window)
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize=visualize,
                         transform_sqrt=normalize)
                fd = fd.reshape(1, -1)
                # -------------------PCA----------------------------------
                fd_name = "meanVal_train_" + str(m) + ".mean"
                fd_path = os.path.join(pca_ph, fd_name)
                meanVal = joblib.load(fd_path)
                fd = fd - meanVal
                fd_name = str(n) + "_eigVects_train_" + str(m) + ".eig"
                fd_path = os.path.join(pca_ph, fd_name)
                n_eigVects = joblib.load(fd_path)
                fd = fd * n_eigVects
                # -------------------PCA----------------------------------
                pred = clf.predict(fd)
                if pred == 1:
                    if clf.decision_function(fd)>0.5:
                        detections.append((int(x*(downscale**scale)),int(y*(downscale**scale)),
                                      clf.decision_function(fd),int(min_wdw_sz[0]*(downscale**scale)),
                                      int(min_wdw_sz[1]*(downscale**scale))))
                        num += 1
            scale += 1
        clone = im.copy()
        for (xd, yd, _, w, h) in detections:
            cv2.rectangle(im, (xd,yd),(xd+w,yd+h),(0,255,0), thickness=2)
        rects = np.array([[x,y,x+w,y+h] for (x,y,_, w,h) in detections])
        sc = [score[0] for (x,y,score,w,h) in detections]
        print "scores:", sc
        sc = np.array(sc)
        pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
        print "shape:", len(pick)

        for(xA,yA,xB,yB) in pick:
            cv2.rectangle(clone,(xA,yA),(xB,yB),(0,0,255),2)
        # t1 = time.time()
        # print "Time cost with feature extraction is :%f seconds" % (t1 - t0)
        # print "Time cost of model prediction is :%f seconds" % (t1 - t2)
        cv2.imshow("Raw Detection before NMS", im)
        cv2.imshow("Final Detections after applying NMS", clone)
        cv2.waitKey(0)
    ############edit####################
    rate = float(num) / total
    print "The classification accuracy is %f" % rate
    t2 = time.time()
    print "Total time cost is %f" % (t2 - t0)


if __name__ == "__main__":
    folder_name = "test_image"
    detector(folder_name)
