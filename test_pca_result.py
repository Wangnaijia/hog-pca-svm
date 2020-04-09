from sklearn.externals import joblib
import glob
import os
import time
from config import *

m = 125
n = 500


def detector():
    # pca model
    fd_name = "svm_" + str(m) + "_pca_" + str(n) + ".model"
    clf = joblib.load(os.path.join(model_path, fd_name))
    # no pca
    # clf = joblib.load(os.path.join(model_path, 'svm_new.model'))
    # only for test the time difference
    num = 0
    total = 0
    detection = []
    t0 = time.time()
    for feat_path in glob.glob(os.path.join(test_feat_ph, '*.feat')):
        total += 1
        fd = joblib.load(feat_path)
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
                # detections.append((int(x*(downscale**scale)),int(y*(downscale**scale)),
                #               clf.decision_function(fd),int(min_wdw_sz[0]*(downscale**scale)),
                #               int(min_wdw_sz[1]*(downscale**scale))))
                detection.append(clf.decision_function(fd))
                num += 1

    rate = float(num) / total
    #print "The classification accuracy is %f" % rate
    t2 = time.time()
    t =t2 - t0
    #print "Total time cost is %f" % (t2 - t0)
    # record the results
    with open('output_pca.txt','w') as f:
        for r in detection:
            f.write(r)
            f.write('\n')
        f.write('The accuracy is:\n')
        f.write(str(rate))
        f.write('Time cost is:\n' )
        f.write(str(t))

if __name__ == "__main__":
    detector()
