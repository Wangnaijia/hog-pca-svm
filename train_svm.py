# -*- coding: utf-8 -*-


from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np
import time

def train_svm():
    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'
    model_path = '../data/models/svm_new.model'

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print np.array(fds).shape, len(labels)
    # fds = np.array(fds).reshape(-1,1)
    # labels = np.array(labels).reshape(1,-1)
    t0 = time.time()
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        t1 = time.time()
        print "Classifier saved to {}".format(model_path)
        print "The cast of time is {} seconds".format(t1 - t0)
        
#训练SVM并保存模型
train_svm()