# model B.I.SCZ, B.II.SCZ and B.III.SCZ
# compare mdoels with only PRS of SCZ (B.I.SCZ), all PRSs (B.II.SCZ), and comorbid PRSs only (B.III.SCZ)
# using logistic, and elastic regression models and LOOCV procedures.
# Performanace matrices are for the left out portion of the samples.

# import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.utils import compute_class_weight
import timeit
import datetime
import math
import gc
import os

# define home directory and parameters
HOME_DIR = "/mnt/c/ResDocs/Manuscripts/SCZ_BD_MDD/Data/"
os.chdir(HOME_DIR)

# read in data
# 42 traits
all_data = pd.read_csv('SCZ_newIID_Diag_SEX_from_42traitsPRSbest_rescaled.csv', header=0)
all_data = np.array(all_data, dtype=float)
all_X = all_data[:, 2:]
all_y = all_data[:, 1]
all_X = np.array(all_X, dtype=float)
all_y = np.array(all_y, dtype=int)

scz = all_X[:, [0, 26]]    # Sex was in column 1, and SCZ PRS was in column 27

# read in data that exclude SCZ related phenotypes
# 35 traits
noSCZ = pd.read_csv('SCZ_newIID_Diag_SEX_from_35traitsPRSbest_noSCZ_rescaled.csv', header=0)
noSCZ = np.array(noSCZ, dtype=float)
noSCZ_X = noSCZ[:, 2:]
noSCZ_y = noSCZ[:, 1]

className = ['CTRL', 'SCZ']

del all_data
gc.collect()


# do leave one out cross validation (LOOCV) procedure
# split the samples into two portions, one for train, one for validation
TRAIN_x, VAL_X, TRAIN_y, VAL_Y = train_test_split(                                                    
                                                    all_X, 
                                                    all_y, 
                                                    test_size=0.2, 
                                                    stratify=all_y,
                                                    random_state=124)

# do the same sample split for scz prs
SCZ_TRAIN_x, SCZ_VAL_X, SCZ_TRAIN_y, SCZ_VAL_Y = train_test_split(
                                                                scz,
                                                                all_y,
                                                                test_size=0.2,
                                                                stratify=all_y,
                                                                random_state=124)


# do the same sample split for noSCZ
noSCZ_TRAIN_x, noSCZ_VAL_X, noSCZ_TRAIN_y, noSCZ_VAL_Y = train_test_split(
                                                                noSCZ_X,
                                                                noSCZ_y,
                                                                test_size=0.2,
                                                                stratify=all_y,
                                                                random_state=124)



# calculate class weights
class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(all_y),
                                        y = all_y                                                    
                                    )
class_weights = dict(zip(np.unique(all_y), class_weights))


# define model
elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.920)
logist_model = LogisticRegression(class_weight=class_weights)

# define k-fold cross validation test harness
cvFold = 5
train_acc_cvscores = []
test_acc_cvscores = []
train_auc_cvscores = []
test_auc_cvscores = []
VAL_ACC_CVSCORES = []
VAL_AUC_CVSCORES = []

scz_train_acc_cvscores = []
scz_test_acc_cvscores = []
scz_train_auc_cvscores = []
scz_test_auc_cvscores = []
SCZ_VAL_ACC_CVSCORES = []
SCZ_VAL_AUC_CVSCORES = []

noscz_train_acc_cvscores = []
noscz_test_acc_cvscores = []
noscz_train_auc_cvscores = []
noscz_test_auc_cvscores = []
noSCZ_VAL_ACC_CVSCORES = []
noSCZ_VAL_AUC_CVSCORES = []


i = 0

for i in range(cvFold):      
    train_x, test_x, train_y, test_y = train_test_split(
                                                    TRAIN_x, 
                                                    TRAIN_y, 
                                                    test_size=0.2, 
                                                    stratify=TRAIN_y)

    scz_train_x, scz_test_x, scz_train_y, scz_test_y = train_test_split(
                                                    SCZ_TRAIN_x, 
                                                    SCZ_TRAIN_y, 
                                                    test_size=0.2, 
                                                    stratify=SCZ_TRAIN_y)

    noscz_train_x, noscz_test_x, noscz_train_y, noscz_test_y = train_test_split(
                                                    noSCZ_TRAIN_x, 
                                                    noSCZ_TRAIN_y, 
                                                    test_size=0.2, 
                                                    stratify=noSCZ_TRAIN_y)

    # fit elastic model for all PRSs
    elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.920)
    elastic_model.fit(train_x, train_y)

    # make a prediction
    train_pred_prob = elastic_model.predict(train_x)
    test_pred_prob = elastic_model.predict(test_x)
    test_y_pred = np.where(test_pred_prob > 0.5, 1, 0)
    print("Test data prediction:\n")
    print(np.c_[test_y, test_pred_prob])

    # validation samples
    VAL_PRED_PROB = elastic_model.predict(VAL_X)
    VAL_Y_PRED = np.where(VAL_PRED_PROB > 0.5, 1, 0)
    print("VALIDATION prediction:\n")
    print(np.c_[VAL_Y, VAL_PRED_PROB])

    #Confution Matrix and Classification Report
    print('Confusion matrix for test samples')
    print(confusion_matrix(test_y, test_y_pred))
    print('Classification Report')
    print(classification_report(test_y, test_y_pred, target_names=className, digits=3))

    # confusion matrix for VALIDATION samples
    print("VALICATION confusion matrix\n", confusion_matrix(VAL_Y, VAL_Y_PRED))
    print("VALICATION report:\n", classification_report(VAL_Y, VAL_Y_PRED, target_names=className, digits=3))
    
    # get ROC data for training and testing
    y_pred_train = elastic_model.predict(train_x)
    y_pred_test = elastic_model.predict(test_x)
    Y_PRED_VAL = elastic_model.predict(VAL_X)
	
    fpr_train, tpr_train, thresholds_train = roc_curve(train_y, y_pred_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(test_y, y_pred_test)
    FPR_VAL, TPR_VAL, THRESHOLDS_VAL = roc_curve(VAL_Y, Y_PRED_VAL)

    auc_train = roc_auc_score(train_y, y_pred_train)
    auc_test = roc_auc_score(test_y, y_pred_test)
    AUC_VAL = roc_auc_score(VAL_Y, Y_PRED_VAL)
	
    y_pred_train = np.where(y_pred_train > 0.5, 1, 0)
    y_pred_test = np.where(y_pred_test > 0.5, 1, 0)
    Y_PRED_VAL = np.where(Y_PRED_VAL > 0.5, 1, 0)
    acc_train = accuracy_score(train_y, y_pred_train)
    acc_test = accuracy_score(test_y, y_pred_test)
    ACC_VAL = accuracy_score(VAL_Y, Y_PRED_VAL)
	
    train_acc_cvscores.append(acc_train)
    test_acc_cvscores.append(acc_test)
    VAL_ACC_CVSCORES.append(ACC_VAL)
    train_auc_cvscores.append(auc_train)
    test_auc_cvscores.append(auc_test)
    VAL_AUC_CVSCORES.append(AUC_VAL)

    # logistic model below
    # fit logistic model for SCZ PRS and Sex
    logist_model.fit(scz_train_x, scz_train_y)

    # make a prediction
    scz_train_pred_prob = logist_model.predict(scz_train_x)
    scz_test_pred_prob = logist_model.predict(scz_test_x)
    scz_test_y_pred = np.where(test_pred_prob > 0.5, 1, 0)
    print("Test data prediction from the linear model:\n")
    print(np.c_[scz_test_y, scz_test_pred_prob])

    # validation samples
    SCZ_VAL_PRED_PROB = logist_model.predict(SCZ_VAL_X)
    SCZ_VAL_Y_PRED = np.where(SCZ_VAL_PRED_PROB > 0.5, 1, 0)
    print("VALIDATION prediction from the linear model:\n")
    print(np.c_[SCZ_VAL_Y, SCZ_VAL_PRED_PROB])

    #Confution Matrix and Classification Report
    print('Confusion matrix for test samples from the linear model')
    print(confusion_matrix(test_y, test_y_pred))
    print('Classification Report for the linear model')
    print(classification_report(scz_test_y, scz_test_y_pred, target_names=className, digits=3))

    # confusion matrix for VALIDATION samples
    print("VALICATION confusion matrix for the linear model\n", confusion_matrix(SCZ_VAL_Y, SCZ_VAL_Y_PRED))
    print("VALICATION report for the linear model:\n", classification_report(SCZ_VAL_Y, SCZ_VAL_Y_PRED, target_names=className, digits=3))

    # get ROC data for training and testing
    scz_y_pred_train = logist_model.predict_proba(scz_train_x)
    scz_y_pred_test = logist_model.predict_proba(scz_test_x)
    SCZ_Y_PRED_VAL = logist_model.predict_proba(SCZ_VAL_X)
	
    scz_fpr_train, scz_tpr_train, thresholds_train = roc_curve(scz_train_y, scz_y_pred_train[:, 1])
    scz_fpr_test, scz_tpr_test, thresholds_test = roc_curve(scz_test_y, scz_y_pred_test[:, 1])
    SCZ_FPR_VAL, SCZ_TPR_VAL, THRESHOLDS_VAL = roc_curve(SCZ_VAL_Y, SCZ_Y_PRED_VAL[:, 1])

    scz_auc_train = roc_auc_score(scz_train_y, scz_y_pred_train[:, 1])
    scz_auc_test = roc_auc_score(scz_test_y, scz_y_pred_test[:, 1])
    SCZ_AUC_VAL = roc_auc_score(SCZ_VAL_Y, SCZ_Y_PRED_VAL[:, 1])	
    scz_y_pred_train = np.where(scz_y_pred_train[:, 1] > 0.5, 1, 0)
    scz_y_pred_test = np.where(scz_y_pred_test[:, 1] > 0.5, 1, 0)
    SCZ_Y_PRED_VAL = np.where(SCZ_Y_PRED_VAL[:, 1] > 0.5, 1, 0)
    scz_acc_train = accuracy_score(scz_train_y, scz_y_pred_train)
    scz_acc_test = accuracy_score(scz_test_y, scz_y_pred_test)
    SCZ_ACC_VAL = accuracy_score(SCZ_VAL_Y, SCZ_Y_PRED_VAL)    

    scz_train_acc_cvscores.append(scz_acc_train)
    scz_test_acc_cvscores.append(scz_acc_test)
    SCZ_VAL_ACC_CVSCORES.append(SCZ_ACC_VAL)
    scz_train_auc_cvscores.append(scz_auc_train)
    scz_test_auc_cvscores.append(scz_auc_test)
    SCZ_VAL_AUC_CVSCORES.append(SCZ_AUC_VAL)
    
    # fit elastic model for noSCZ
    elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.920)
    elastic_model.fit(noscz_train_x, noscz_train_y)

    # make a prediction
    noscz_train_pred_prob = elastic_model.predict(noscz_train_x)
    noscz_test_pred_prob = elastic_model.predict(noscz_test_x)
    noscz_test_y_pred = np.where(noscz_test_pred_prob > 0.5, 1, 0)
    print("Test data prediction for the noSCZ model:\n")
    print(np.c_[noscz_test_y, noscz_test_pred_prob])

    # validation samples
    noSCZ_VAL_PRED_PROB = elastic_model.predict(noSCZ_VAL_X)
    noSCZ_VAL_Y_PRED = np.where(noSCZ_VAL_PRED_PROB > 0.5, 1, 0)
    print("VALIDATION prediction:\n")
    print(np.c_[noSCZ_VAL_Y, noSCZ_VAL_PRED_PROB])

    #Confution Matrix and Classification Report
    print('Confusion matrix for test samples from noSCZ model')
    print(confusion_matrix(noscz_test_y, noscz_test_y_pred))
    print('Classification Report for noSCZ model')
    print(classification_report(noscz_test_y, noscz_test_y_pred, target_names=className, digits=3))

    # confusion matrix for VALIDATION samples
    print("VALICATION confusion matrix for the noSCZ model\n", confusion_matrix(noSCZ_VAL_Y, noSCZ_VAL_Y_PRED))
    print("VALICATION report for the noSCZ model:\n", classification_report(noSCZ_VAL_Y, noSCZ_VAL_Y_PRED, target_names=className, digits=3))
    
    # get ROC data for training and testing
    noscz_y_pred_train = elastic_model.predict(noscz_train_x)
    noscz_y_pred_test = elastic_model.predict(noscz_test_x)
    noSCZ_Y_PRED_VAL = elastic_model.predict(noSCZ_VAL_X)
	
    noscz_fpr_train, noscz_tpr_train, thresholds_train = roc_curve(noscz_train_y, noscz_y_pred_train)
    noscz_fpr_test, noscz_tpr_test, thresholds_test = roc_curve(noscz_test_y, noscz_y_pred_test)
    noSCZ_FPR_VAL, noSCZ_TPR_VAL, THRESHOLDS_VAL = roc_curve(noSCZ_VAL_Y, noSCZ_Y_PRED_VAL)
	
    noscz_auc_train = roc_auc_score(noscz_train_y, noscz_y_pred_train)
    noscz_auc_test = roc_auc_score(noscz_test_y, noscz_y_pred_test)
    noSCZ_AUC_VAL = roc_auc_score(noSCZ_VAL_Y, noSCZ_Y_PRED_VAL)

    noscz_y_pred_train = np.where(noscz_y_pred_train > 0.5, 1, 0)
    noscz_y_pred_test = np.where(noscz_y_pred_test > 0.5, 1, 0)
    noSCZ_Y_PRED_VAL = np.where(noSCZ_Y_PRED_VAL > 0.5, 1, 0)
    noscz_acc_train = accuracy_score(noscz_train_y, noscz_y_pred_train)
    noscz_acc_test = accuracy_score(noscz_test_y, noscz_y_pred_test)
    noSCZ_ACC_VAL = accuracy_score(noSCZ_VAL_Y, noSCZ_Y_PRED_VAL)    
	
    noscz_train_acc_cvscores.append(noscz_acc_train)
    noscz_test_acc_cvscores.append(noscz_acc_test)
    noSCZ_VAL_ACC_CVSCORES.append(noSCZ_ACC_VAL)
    noscz_train_auc_cvscores.append(noscz_auc_train)
    noscz_test_auc_cvscores.append(noscz_auc_test)
    noSCZ_VAL_AUC_CVSCORES.append(noSCZ_AUC_VAL)

       
    # plot ROC 
    pyplot.figure(i)
    pyplot.plot([0, 1], [0, 1], 'k--')

    # Logistic model with only SCZ PRS
    pyplot.plot(
	    SCZ_FPR_VAL, 
	 	SCZ_TPR_VAL, 
	 	label='SCZ PRS only: {:.3f}'.format(SCZ_AUC_VAL),
	 	linestyle='solid',
	 	lw=2)

    # elastic model with all PRSs
    pyplot.plot(
        FPR_VAL, 
		TPR_VAL, 
		label='All PRSs: {:.3f}'.format(AUC_VAL),
		linestyle='dashed',
		lw=2)

    # elastic model with only comorbid PRSs
    pyplot.plot(
	    noSCZ_FPR_VAL, 
		noSCZ_TPR_VAL, 
		label='Comorbid PRSs only: {:.3f}'.format(noSCZ_AUC_VAL),
		linestyle='dotted',
		lw=2)

    pyplot.title('SCZ ROC curve', fontsize=18)
    pyplot.xlabel('False positive rate', fontsize=14)
    pyplot.ylabel('True positive rate', fontsize=14)
    pyplot.legend(loc='best')
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)

    pyplot.savefig('SCZ_base_models_loo_v3.0_cv_' + str(cvFold) + '_ROC_' + str(i) + '.png')
    i = i + 1

# convert list of dict to np array before printing
# SCZ+comorbid model
train_acc_cvscores = np.array(pd.DataFrame(train_acc_cvscores).values, dtype=np.float32)
test_acc_cvscores = np.array(pd.DataFrame(test_acc_cvscores).values, dtype=np.float32)
VAL_ACC_CVSCORES = np.array(pd.DataFrame(VAL_ACC_CVSCORES).values, dtype=np.float32)
train_auc_cvscores = np.array(pd.DataFrame(train_auc_cvscores).values, dtype=np.float32)
test_auc_cvscores = np.array(pd.DataFrame(test_auc_cvscores).values, dtype=np.float32)
VAL_AUC_CVSCORES = np.array(pd.DataFrame(VAL_AUC_CVSCORES).values, dtype=np.float32)

# linear model results:
scz_train_acc_cvscores = np.array(pd.DataFrame(scz_train_acc_cvscores).values, dtype=np.float32)
scz_test_acc_cvscores = np.array(pd.DataFrame(scz_test_acc_cvscores).values, dtype=np.float32)
SCZ_VAL_ACC_CVSCORES = np.array(pd.DataFrame(SCZ_VAL_ACC_CVSCORES).values, dtype=np.float32)
scz_train_auc_cvscores = np.array(pd.DataFrame(scz_train_auc_cvscores).values, dtype=np.float32)
scz_test_auc_cvscores = np.array(pd.DataFrame(scz_test_auc_cvscores).values, dtype=np.float32)
SCZ_VAL_AUC_CVSCORES = np.array(pd.DataFrame(SCZ_VAL_AUC_CVSCORES).values, dtype=np.float32)

# noSCZ model results:
noscz_train_acc_cvscores = np.array(pd.DataFrame(noscz_train_acc_cvscores).values, dtype=np.float32)
noscz_test_acc_cvscores = np.array(pd.DataFrame(noscz_test_acc_cvscores).values, dtype=np.float32)
noSCZ_VAL_ACC_CVSCORES = np.array(pd.DataFrame(noSCZ_VAL_ACC_CVSCORES).values, dtype=np.float32)
noscz_train_auc_cvscores = np.array(pd.DataFrame(noscz_train_auc_cvscores).values, dtype=np.float32)
noscz_test_auc_cvscores = np.array(pd.DataFrame(noscz_test_auc_cvscores).values, dtype=np.float32)
noSCZ_VAL_AUC_CVSCORES = np.array(pd.DataFrame(noSCZ_VAL_AUC_CVSCORES).values, dtype=np.float32)

# print CV results
print("\nElastic model training acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(train_acc_cvscores), np.std(train_acc_cvscores)))
print("\nElastic model testing acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(test_acc_cvscores), np.std(test_acc_cvscores)))
print("\nElastic model VAL ACC: \n")
print("%.3f (+/- %.3f)" % (np.mean(VAL_ACC_CVSCORES), np.std(VAL_ACC_CVSCORES)))
print("\nElastic model training auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(train_auc_cvscores), np.std(train_auc_cvscores)))
print("\nElastic model testing auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(test_auc_cvscores), np.std(test_auc_cvscores)))
print("\nElastic model VAL AUC: \n")
print("%.3f (+/- %.3f)" % (np.mean(VAL_AUC_CVSCORES), np.std(VAL_AUC_CVSCORES)))

print("\nLogistic model training acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(scz_train_acc_cvscores), np.std(scz_train_acc_cvscores)))
print("\nLogistic model testing acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(scz_test_acc_cvscores), np.std(scz_test_acc_cvscores)))
print("\nLogistic model VAL ACC: \n")
print("%.3f (+/- %.3f)" % (np.mean(SCZ_VAL_ACC_CVSCORES), np.std(SCZ_VAL_ACC_CVSCORES)))
print("\nLogistic model training auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(scz_train_auc_cvscores), np.std(scz_train_auc_cvscores)))
print("\nLogistic model testing auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(scz_test_auc_cvscores), np.std(scz_test_auc_cvscores)))
print("\nLogistic model VAL AUC: \n")
print("%.3f (+/- %.3f)" % (np.mean(SCZ_VAL_AUC_CVSCORES), np.std(SCZ_VAL_AUC_CVSCORES)))

print("\nno SCZ PRS model training acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(noscz_train_acc_cvscores), np.std(noscz_train_acc_cvscores)))
print("\nno SCZ PRS model testing acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(noscz_test_acc_cvscores), np.std(noscz_test_acc_cvscores)))
print("\nno SCZ PRS model VAL ACC: \n")
print("%.3f (+/- %.3f)" % (np.mean(noSCZ_VAL_ACC_CVSCORES), np.std(noSCZ_VAL_ACC_CVSCORES)))
print("\nno SCZ PRS model training auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(noscz_train_auc_cvscores), np.std(noscz_train_auc_cvscores)))
print("\nno SCZ PRS model, testing auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(noscz_test_auc_cvscores), np.std(noscz_test_auc_cvscores)))
print("\nno SCZ PRS model, VAL AUC: \n")
print("%.3f (+/- %.3f)" % (np.mean(noSCZ_VAL_AUC_CVSCORES), np.std(noSCZ_VAL_AUC_CVSCORES)))


now = datetime.datetime.now()
print("\nThe run is done by: \n", now.strftime("%Y-%m-%d %H:%M"))
