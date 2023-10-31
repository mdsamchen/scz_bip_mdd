# Model I.SCZ
# This script is used to classify SCZ with all PRSs from 42 traits, 242 attributes
# This script uses a LOOCV procedure. The matrices reported are from the 
# left-out part.

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler, ADASYN, BorderlineSMOTE
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score
import timeit
import datetime
import math
import gc

# define home directory and parameters
HOME_DIR = "D:/SCZ_BP_MD_PRS/SCZ/with_SCZ_PRS/"
os.chdir(HOME_DIR)

# training parameters
numEpoch = 200
batchSize = 128
DROPOUT = 0.50
LR = 0.001
EPSILON = 0.99
DECAY = 0.001
DROP = 0.90
EPOCHS_DROP = 20

# optimizer
adagrad = optimizers.Adagrad(lr=LR, epsilon=EPSILON, decay=DECAY)

# read in prs training data
SCZ_all = pd.read_csv('SCZ_newIID_Diag_SEX_from_42traits_238_rescaled.csv', header=0)
SCZ_all = np.array(SCZ_all, dtype=np.float32)
SCZ_all_prs = SCZ_all[:, 2:]


# get Y, i.e. Diagnosis
SCZ_all_Diagnosis = SCZ_all[:, 1]
SCZ_all_Diagnosis = np.array(SCZ_all_Diagnosis, dtype=np.float32)

# use borderlineSMOTE
oversample = BorderlineSMOTE()
SCZ_all_prs, SCZ_all_Diagnosis = oversample.fit_resample(SCZ_all_prs, SCZ_all_Diagnosis)
SCZ_all_prs = np.reshape(SCZ_all_prs, (-1, 238, 1))

# delete dataframes to save memory
del SCZ_all
gc.collect()

# learning rate scheduler
# define step decay function
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(epoch):
    initial_lrate = LR
    drop = DROP
    epochs_drop = EPOCHS_DROP
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# prs embedding parameters
inputDim = 238
prsInput = Input(shape=(238, 1, ))

# do leave one out validation procedure
# split the samples into two portions, one for train, one for validation
TRAIN_x, VAL_X, TRAIN_y, VAL_Y = train_test_split(
                                                SCZ_all_prs, 
                                                SCZ_all_Diagnosis, 
                                                test_size=0.2, 
                                                stratify=SCZ_all_Diagnosis)


# use class weights
class_weights = compute_class_weight(
                                    class_weight = "balanced",
                                    classes = np.unique(SCZ_all_Diagnosis),
                                    y = SCZ_all_Diagnosis                                                    
                                    )
class_weights = dict(zip(np.unique(SCZ_all_Diagnosis), class_weights))

print("Class Weights: \n", class_weights)

# define k-fold cross validation test harness
cvFold = 5
train_acc_cvscores = []
test_acc_cvscores = []
train_auc_cvscores = []
test_auc_cvscores = []
VAL_ACC_CVSCORES = []
VAL_AUC_CVSCORES = []
i = 0

skf = StratifiedKFold(n_splits=cvFold, shuffle=True, random_state=1234)

for train_idx, test_idx in skf.split(TRAIN_x, TRAIN_y):
    train_x, test_x = TRAIN_x[train_idx], TRAIN_x[test_idx]
    train_y, test_y = TRAIN_y[train_idx], TRAIN_y[test_idx]

    checkpointer = ModelCheckpoint(
        filepath='./best_weights_' + str(i) + '.hdf5',
        monitor='val_acc',
        save_best_only=True,
        save_weights_only=False,
        verbose=2)

    # learning schedule callback
    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate, checkpointer]

    # build the models: first prs layers
    prs_dnn = Dense(
        units=256, 
        input_dim = inputDim,
        activation='relu')(prsInput)
    prs_dnn = Flatten()(prs_dnn)
    prs_dnn = Dense(units=1024, activation='relu')(prs_dnn)
    prs_dnn = Dense(units=1024, activation='relu')(prs_dnn)
    prs_dnn = Dense(units=1024, activation='relu')(prs_dnn)
    prs_dnn = Dense(units=1024, activation='relu')(prs_dnn)
    prs_dnn = Dense(units=1024, activation='relu')(prs_dnn)
    prs_dnn = Dense(units=1024, activation='relu')(prs_dnn)
    prs_dnn = Dense(units=64, activation='relu')(prs_dnn)
    output = Dense(units=1, activation='sigmoid')(prs_dnn)
    classifier = Model(inputs=prsInput, outputs=output)
    print("Model summary: \n", classifier.summary())

    # compile the model
    classifier.compile(
        optimizer=adagrad, 
        loss='binary_crossentropy', 
        metrics=['acc', tf.keras.metrics.AUC()])
   
    # fit the model with training data
    training_start_time = timeit.default_timer()
    history = classifier.fit(
        x = train_x,
        y = train_y,
        batch_size = batchSize,
        epochs = numEpoch,
        validation_data = (test_x, test_y),
        class_weight=class_weights,
        callbacks=callbacks_list,
        shuffle=True,
        verbose=2)

    training_end_time = timeit.default_timer()
    print("Training time: {:10.2f} min. \n" .format((training_end_time - training_start_time) / 60))    

    # evaluate the model with testing data
    train_scores = classifier.evaluate(train_x, train_y, verbose = 0)
    test_scores = classifier.evaluate(test_x, test_y, verbose = 0)
    VAL_SCORES = classifier.evaluate(VAL_X, VAL_Y, verbose = 0)
    print("%s: %.2f%%" % (classifier.metrics_names[1], train_scores[1]*100))
    print("%s: %.2f%%" % (classifier.metrics_names[1], test_scores[1]*100))    
    print("%s: %.2f%%" % ("VALIDATION ACC", VAL_SCORES[1]*100))    
    train_acc_cvscores.append(train_scores[1])
    test_acc_cvscores.append(test_scores[1])
    VAL_ACC_CVSCORES.append(VAL_SCORES[1])
	
    # prediction 
    train_pred_prob = classifier.predict(train_x)
    print("Train data prediction:\n")
    print(np.c_[train_y, train_pred_prob])
    test_pred_prob = classifier.predict(test_x)
    test_y_pred = np.where(test_pred_prob > 0.5, 1, 0)
    print("Test data prediction:\n")
    print(np.c_[test_y, test_pred_prob])
    VAL_PRED_PROB = classifier.predict(VAL_X)
    VAL_Y_PRED = np.where(VAL_PRED_PROB > 0.5, 1, 0)
    print("VALIDATION prediction:\n")
    print(np.c_[VAL_Y, VAL_PRED_PROB])

    #Confution Matrix and Classification Report
    print('Confusion matrix for test samples')
    print(confusion_matrix(test_y, test_y_pred))
    print('Classification Report')
    target_names = ['CTRL', 'SCZ']
    print(classification_report(test_y, test_y_pred, target_names=target_names, digits=3))

	# confusion matrix for VALIDATION samples
    print("VALICATION confusion matrix\n", confusion_matrix(VAL_Y, VAL_Y_PRED))
    print("VALICATION report:\n", classification_report(VAL_Y, VAL_Y_PRED, target_names=target_names, digits=3))	

    # make cm plot
    val_cm = confusion_matrix(VAL_Y, VAL_Y_PRED)
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=target_names)
    cm_plot = disp.plot()
    for labels in disp.text_.ravel():
        labels.set_fontsize(18)
    pyplot.ylabel('True Label', fontsize=14)
    pyplot.xlabel('Predicted Label', fontsize=14)
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)
    pyplot.savefig('MDD_newIID_Diag_SEX_42traitsPRS238_rescaled_loo_CV5_v5_cv_' + str(cvFold) + '_cm_' + str(i) +'.png')
    pyplot.close()
    
    # plot training and validation history
    pyplot.figure(i)
    pyplot.plot(
        history.history['acc'], 
        label='train (acc={:.3f})'.format(train_scores[1]),
        linestyle='-',
        lw=2)
    pyplot.plot(
        history.history['val_acc'], 
        label='test (acc={:.3f})'.format(test_scores[1]),
        linestyle='-.', 
        lw=2)
    pyplot.title('model accuracy', fontsize=18)
    pyplot.ylabel('accuracy',fontsize=14)
    pyplot.xlabel('epoch',fontsize=14)
    pyplot.legend(loc='best',fontsize=14)
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)
    pyplot.savefig('SCZ_newIID_Diag_SEX_42traitsPRS238_rescaled_loo_CV5_cv_' + str(cvFold) + '_acc_' + str(i) +'.png')

    # get ROC data for training and testing
    y_pred_train = classifier.predict(train_x).ravel()
    y_pred_test = classifier.predict(test_x).ravel()
    Y_PRED_VAL = classifier.predict(VAL_X).ravel()
	
    fpr_train, tpr_train, thresholds_train = roc_curve(train_y, y_pred_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(test_y, y_pred_test)
    FPR_VAL, TPR_VAL, THRESHOLDS_VAL = roc_curve(VAL_Y, Y_PRED_VAL)
	
    auc_train = roc_auc_score(train_y, y_pred_train)
    auc_test = roc_auc_score(test_y, y_pred_test)
    AUC_VAL = roc_auc_score(VAL_Y, Y_PRED_VAL)
	
    train_auc_cvscores.append(auc_train)
    test_auc_cvscores.append(auc_test)
    VAL_AUC_CVSCORES.append(AUC_VAL)
	
    # plot ROC 
    pyplot.figure(cvFold + i)
    pyplot.plot([0, 1], [0, 1], 'k--')
    pyplot.plot(
	    fpr_train, 
	    tpr_train, 
		label='train auc: {:.3f}'.format(auc_train),
		linestyle='solid',
		lw=2)
    pyplot.plot(
	    fpr_test, 
		tpr_test, 
		label='test auc: {:.3f}'.format(auc_test),
		linestyle='dashed',
		lw=2)
    pyplot.plot(
	    FPR_VAL, 
		TPR_VAL, 
		label='VAL AUC: {:.3f}'.format(AUC_VAL),
		linestyle='dotted',
		lw=2)
	
    pyplot.xlabel('False positive rate', fontsize=14)
    pyplot.ylabel('True positive rate',fontsize=14)
    pyplot.title('ROC curve',fontsize=18)
    pyplot.legend(loc='best',fontsize=14)
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)
    pyplot.savefig('SCZ_newIID_Diag_SEX_42traitsPRS238_rescaled_loo_CV5_cv_' + str(cvFold) + '_ROC_' + str(cvFold + i) + '.png')
    i = i + 1

# convert list of dict to np array before print
train_acc_cvscores = np.array(pd.DataFrame(train_acc_cvscores).values, dtype=np.float32)
test_acc_cvscores = np.array(pd.DataFrame(test_acc_cvscores).values, dtype=np.float32)
VAL_ACC_CVSCORES = np.array(pd.DataFrame(VAL_ACC_CVSCORES).values, dtype=np.float32)
train_auc_cvscores = np.array(pd.DataFrame(train_auc_cvscores).values, dtype=np.float32)
test_auc_cvscores = np.array(pd.DataFrame(test_auc_cvscores).values, dtype=np.float32)
VAL_AUC_CVSCORES = np.array(pd.DataFrame(VAL_AUC_CVSCORES).values, dtype=np.float32)

# print CV results
print("\nCV results for the training acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(train_acc_cvscores), np.std(train_acc_cvscores)))
print("\nCV results for the testing acc: \n")
print("%.3f (+/- %.3f)" % (np.mean(test_acc_cvscores), np.std(test_acc_cvscores)))
print("\nCV results for the VAL ACC: \n")
print("%.3f (+/- %.3f)" % (np.mean(VAL_ACC_CVSCORES), np.std(VAL_ACC_CVSCORES)))
print("\nCV results for the training auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(train_auc_cvscores), np.std(train_auc_cvscores)))
print("\nCV results for the testing auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(test_auc_cvscores), np.std(test_auc_cvscores)))
print("\nCV results for the VAL AUC: \n")
print("%.3f (+/- %.3f)" % (np.mean(VAL_AUC_CVSCORES), np.std(VAL_AUC_CVSCORES)))

now = datetime.datetime.now()
print("\nThe run is done by: \n", now.strftime("%Y-%m-%d %H:%M"))

