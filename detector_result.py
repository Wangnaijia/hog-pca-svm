
from sklearn.externals import joblib
import cv2
from config import *
import time
from skimage import color
import matplotlib.pyplot as plt
import os
import glob

def detector():

    clf = joblib.load(os.path.join(model_path, 'svm_new.model'))
    # only for testing the time difference
    num = 0
    total = 0
    detection = []
    t0 = time.time()
    for feat_path in glob.glob(os.path.join(test_feat_ph, '*.feat')):
        total += 1
        fd = joblib.load(feat_path)
        fd = fd.reshape(1, -1)

        pred = clf.predict(fd)
        if pred == 1:
            if clf.decision_function(fd)>0.5:
                # detections.append((int(x*(downscale**scale)),int(y*(downscale**scale)),
                #               clf.decision_function(fd),int(min_wdw_sz[0]*(downscale**scale)),
                #               int(min_wdw_sz[1]*(downscale**scale))))
                detection.append(clf.decision_function(fd))
                num += 1

    rate = float(num) / total
    t2 = time.time()
    t = t2 - t0
    # record the results
    with open("output.txt", 'w', encoding="utf-8") as f:
        for r in detection:
            f.write(r)
            f.write('\n')
        f.write('\n The accuracy is:\n')
        f.write(str(rate))
        f.write('\n Time cost is:\n')
        f.write(str(t))

if __name__ == '__main__':

    detector()
