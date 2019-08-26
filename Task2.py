# Task - 2 Sumeet Kumar - 5873137 - Sk521
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from keras import optimizers
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegressionCV

K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train_CNN = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_CNN = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train_CNN = X_train_CNN / 255
X_test_CNN = X_test_CNN / 255

# one hot encode outputs
y_train_CNN = np_utils.to_categorical(y_train)
y_test_CNN = np_utils.to_categorical(y_test)
num_classes = y_test_CNN.shape[1]

#Reshaping to a vector to process for SVM and LRC
trainSamples, trainWidth, trainHeight = X_train.data.shape
testSamples, testWidth, testHeight = X_test.data.shape

X_train = X_train.reshape((trainSamples,trainWidth*trainHeight))
X_test = X_test.reshape((testSamples,testWidth*testHeight))


# Create a classifier: a support vector classifier and LRC

svmClassifier = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
            intercept_scaling=1, loss='squared_hinge', max_iter=1000,
           multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)

# We learn the digits on the now
svmClassifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
expected = y_test
predicted = svmClassifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (svmClassifier, metrics.classification_report(expected, predicted)))
      
# compute the confusion matrix
cmSVM = confusion_matrix(expected, predicted)

df = pd.DataFrame(cmSVM, index=range(num_classes) )
fig = plt.figure(figsize=(20,7))
heatmap = sns.heatmap(df, annot=True, fmt="d")

#print Accuracy
print("Accuracy for SVM ={}".format(metrics.accuracy_score(expected, predicted)))


#LRC without Cross Validation
lrcClassifier = LogisticRegression(solver = 'lbfgs',multi_class='ovr',random_state = 1)
lrcClassifier.fit(X_train, y_train)

lrcPredicted = lrcClassifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (lrcClassifier, metrics.classification_report(expected, lrcPredicted)))
      
#Compute Confusion Matrix
cmLRC = confusion_matrix(expected, lrcPredicted)

dfLRC = pd.DataFrame(cmLRC, index=range(num_classes) )
fig = plt.figure(figsize=(20,7))
heatmap = sns.heatmap(dfLRC, annot=True, fmt="d")

#Print accuracy
print("Accuracy for LRC ={}".format(metrics.accuracy_score(expected, lrcPredicted)))

#LRC with Cross Validation
lrcClassifierCV = LogisticRegressionCV(solver = 'lbfgs',multi_class='multinomial',random_state = 1, cv=5)
lrcClassifierCV.fit(X_train, y_train)

lrcPredictedCV = lrcClassifierCV.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (lrcClassifierCV, metrics.classification_report(expected, lrcPredicted)))
      
#Compute Confusion Matrix
cmLRCCV = confusion_matrix(expected, lrcPredictedCV)

dfLRCCV = pd.DataFrame(cmLRCCV, index=range(num_classes) )
fig = plt.figure(figsize=(20,7))
heatmap = sns.heatmap(dfLRCCV, annot=True, fmt="d")

#Print accuracy
print("Accuracy for LRCCV ={}".format(metrics.accuracy_score(expected, lrcPredictedCV)))


# CNN model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model  
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()
# Fit the model
k=model.fit(X_train_CNN, y_train_CNN, validation_data=(X_test_CNN, y_test_CNN), epochs=10, batch_size=128)
# Final evaluation of the model
scores = model.evaluate(X_test_CNN, y_test_CNN, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

print('Test loss:', scores[0])  
print('Test accuracy:', scores[1]) 

# Predict the values from the validation dataset
ypred_onehot = model.predict(X_test_CNN)
# Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2
ypred = np.argmax(ypred_onehot,axis=1)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(y_test_CNN,axis=1)


#Classification Report
print("Classification report for classifier %s:\n%s\n"
     % (model, metrics.classification_report(ytrue, ypred)))
      
#Compute Confusion Matrix
cmCNN = confusion_matrix(ytrue, ypred)

dfCNN = pd.DataFrame(cmCNN, index=range(num_classes) )
fig = plt.figure(figsize=(20,7))
heatmap = sns.heatmap(dfCNN, annot=True, fmt="d")

#Plot for Accuracy and Epochs
plt.plot(k.history['acc'])
plt.plot(k.history['val_acc'])
plt.legend(['Training','Test'])
plt.title('Accuracy')
plt.xlabel('Epochs')