import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
import time
from config import *

m = 10

def zeroMean(dataMat): #zero normalization
    # If PCA directories don't exist, create them
    if not os.path.isdir(pca_ph):
        os.makedirs(pca_ph)

    meanVal = np.mean(dataMat, axis=0)  # calculate mean value of every column.
    # meanVal = m
    fd_name = "meanVal_train_" + str(m) +".mean"
    fd_path = os.path.join(pca_ph, fd_name)
    joblib.dump(meanVal, fd_path)
    newData = dataMat - meanVal
    return newData, meanVal

def pca(dataMat, n):
    print "start to do PCA"
    t0 = time.time()
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # calculate feature value and feature vector
    fd_name = "eigVals_train_" + str(m) + ".eig"
    fd_path = os.path.join(pca_ph, fd_name)
    joblib.dump(eigVals, fd_path)

    fd_name = "eigVects_train_" + str(m) + ".eig"
    fd_path = os.path.join(pca_ph, fd_name)
    joblib.dump(eigVects, fd_path)

    eigValIndice = np.argsort(eigVals)  # sort feature value
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]  # take n feature value
    n_eigVect = eigVects[:, n_eigValIndice]  # take n feature vector
    fd_name = str(n) + "_eigVects_train_" + str(m) + ".eig"
    fd_path = os.path.join(pca_ph, fd_name)
    joblib.dump(n_eigVect, fd_path)

    lowDataMat = newData * n_eigVect  # calculate low dimention data
    t1 = time.time()
    print "PCA takes %f seconds" % (t1 - t0)
    return lowDataMat

if __name__ == "__main__":
    clf_type = 'LIN_SVM'
    n = 500  # dimentions
    fds = []
    labels = []
    num = 0
    for feat_path in glob.glob(os.path.join(pos_feat_ph, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(np.array([1]))
        num += 1
    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_ph,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(np.array([0]))
        num += 1
    fds = np.array(fds, dtype=int)
    fds.shape = num, -1
    fds = pca(fds, n)

    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        fd_name = "svm_" + str(m) + "_pca_" + str(n) +".model"
        fd_path = os.path.join(model_path, fd_name)
        joblib.dump(clf, fd_path)
        print "Classifier saved to {}".format(model_path)
