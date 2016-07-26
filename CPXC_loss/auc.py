from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

setting = open("temp/setting.txt","r")
labels = loadtxt("temp/label.txt", comments="#", delimiter=",", unpack=False)
n_classes = int(setting.read())
classes = range(0,n_classes)
y_test = label_binarize(labels, classes)
y_score = np.zeros((len(labels),n_classes))
for i in classes:
    scores = loadtxt("temp/p"+str(i)+".txt",comments="#", delimiter = ",", unpack = False)
    for j in range(len(scores)):
        y_score[j,i] = scores[j]

if n_classes > 2:
    print y_test
    print y_score
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print roc_auc[i]

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(roc_auc["macro"])
else:
    fpr, tpr, thresholds = metrics.roc_curve(y_test[:,0], y_score[:,0],pos_label=0)
    auc1 = metrics.auc(fpr, tpr )
    print auc1
    print y_test[:,0]
    print y_score[:,0]
