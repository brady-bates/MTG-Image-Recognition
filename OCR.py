import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Activation
from keras.models import Sequential


# ------------------ Preprocessing Data ------------------

# create empty lists for training images and labels
images = []
labels = []

# category encoder
cat_encoder = OneHotEncoder(sparse=False)

# path to training data
path = "MTG_ML/Data/Training"

# iterate over the training data, convert images into arrays, and append to their respective arrays
dir_list = os.listdir(path)                                                     # get a list of directories from given path
for i in dir_list:                                                              # iterate over all directories in this directory
    dir = os.path.join(path, i)                                                 # grab the directory path we want to extract files from
    file_list = os.listdir(dir)                                                 # create a list of files we would like to extract
    for j in file_list:                                                         # iterate over all files in this directory
        files = os.path.join(dir, j)                                            # store path to file
        img = cv2.imread(files)                                                 # open the image file
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                            # convert image to grayscale image
        T, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)                # threshold the grayscale image
        thresh_resized = cv2.resize(thresh, (64, 64))                           # resize the thresholded image
        bin = np.array((thresh_resized > T).astype(int))                        # binarize the image. this normalizes our images pixels to either 1 or 0
        images.append(bin)                                                      # append the img array to the images list
    labels.append(i)                                                            # append the label to the labels list

# shuffle data
images, labels = shuffle(images, labels)

# print shape of train data
X_train = np.array(images)
y_train = np.array(labels)
y_train_1hot = cat_encoder.fit_transform(y_train.reshape(-1, 1))
print("X_train Shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("y_train_1hot shape: ", y_train_1hot.shape)

# Create a dictionary of the onehot encoded features so we can decode our model's output
oneHot_dict = {}
y_train_unique = np.unique(y_train, return_index=True)
for i in y_train_unique[1]:
  oneHot_dict["{}".format(y_train[i])] = y_train_1hot[i]

# empty the lists for testing images and labels
images = []
labels = []

# new path to training data
path = "MTG_ML/Data/Testing"

# iterate over the testing data, convert images into arrays, and append to their respective arrays
dir_list = os.listdir(path)                                                     # get a list of directories from given path
for i in dir_list:                                                              # iterate over all directories in this directory
    dir = os.path.join(path, i)                                                 # grab the directory path we want to extract files from
    file_list = os.listdir(dir)                                                 # create a list of files we would like to extract
    for j in file_list:                                                         # iterate over all files in this directory
        files = os.path.join(dir, j)                                            # store path to file
        img = cv2.imread(files)                                                 # open the image file
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                            # convert image to grayscale image
        T, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)                # threshold the grayscale image
        thresh_resized = cv2.resize(thresh, (64, 64))                           # resize the thresholded image
        bin = np.array((thresh_resized > T).astype(int))                        # binarize the image. this normalizes our images pixels to either 1 or 0
        images.append(bin)                                                      # append the img array to the images list
    labels.append(i)                                                            # append the label to the labels list

# shuffle data
images, labels = shuffle(images, labels)

# print shape of test data
X_test = np.array(images)
y_test = np.array(labels)
y_test_1hot = cat_encoder.fit_transform(y_test.reshape(-1, 1))
print("\nX_test Shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
print("y_test_1hot shape: ", y_test_1hot.shape)

# ------------------ Creating the Model ------------------

# parameters for the model
num_classes = 36
dropout_rate = 0.3
num_epochs = 5

kernel_size_conv = (3, 3)
strides_conv = 1
padding_conv = 'same'
input_layer_shape = (64, 64, 1)

kernel_size_max_pool = (2, 2)
stride_max_pool = 2

n_filters_conv_1 = 64
n_filters_conv_2 = 32

# we will use a sequential model. They are more simple than fucntional models
model = Sequential()

model.add(Conv2D(filters = n_filters_conv_1,
                 kernel_size = kernel_size_conv,
                 strides = strides_conv,
                 padding = padding_conv,
                 input_shape = input_layer_shape))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = kernel_size_max_pool,
                       strides = stride_max_pool))

model.add(Conv2D(filters = n_filters_conv_1,
                 kernel_size = kernel_size_conv,
                 strides = strides_conv,
                 padding = padding_conv,
                 input_shape = input_layer_shape))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=kernel_size_max_pool,
                       strides = stride_max_pool))

model.add(Flatten())
model.add(Dense(36))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
model.add(Activation('softmax'))


# ------------------ Training the Model ------------------

# compile the model
model.compile(loss='categorical_crossentropy', metrics = ['accuracy'])
# fit the model
history = model.fit(X_train,
                    y_train_1hot,
                    epochs = num_epochs,
                    validation_split = 0.1,
                    verbose = 1)
# list data in history
print(history.history.keys())
# summarize hisory for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd

# create predictions using test set
yhat = model.predict(X_test, verbose=1)

# If probability is greater than 50%, set to 1. If less than 50%, set to 0
for prediction in yhat:
  for i in range(0, len(prediction)):
    if prediction[i] > 0.5:
      prediction[i] = 1
    else:
        prediction[i] = 0

# print the metrics per feature
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
metrics = pd.DataFrame()
for i in range(len(yhat)):
  # print("<<<<< Prediction Metrics for Test Data {} ({}) >>>>>".format(i, y_test[i]))
  accuracy = accuracy_score(y_test_1hot[i], yhat[i])
  con_matrix = confusion_matrix(y_test_1hot[i], yhat[i])
  precision = precision_score(y_test_1hot[i], yhat[i])
  recall = recall_score(y_test_1hot[i], yhat[i])
  f1 = f1_score(y_test_1hot[i], yhat[i])
  # print('Accuracy: {0:0.2%}'.format(accuracy))
  # print('Confusion Matrix:\n{}'.format(con_matrix))
  # print('Precision: {0:0.2%}'.format(precision))
  # print('Recall: {0:0.2%}'.format(recall))
  # print('F1 Score: {0:0.2%}'.format(f1))
  # print()
  accuracy_list.append(accuracy)
  precision_list.append(precision)
  recall_list.append(recall)
  f1_list.append(f1)

metrics.insert(0, "y_test", y_test)
metrics.insert(1, "accuracy", accuracy_list)
metrics.insert(2, "precision", precision_list)
metrics.insert(3, "recall", recall_list)
metrics.insert(4, "f1 score", f1_list)
print("<<<<< Metrics Dataframe >>>>>")
print(metrics)

# print metrics for the overall model by averaging the metrics of each feature
print("\n<<<<< Overall Prediction Metrics >>>>>")
print('Accuracy: {0:0.2%}'.format(np.mean(accuracy_list)))
print('Precision: {0:0.2%}'.format(np.mean(precision_list)))
print('Recall: {0:0.2%}'.format(np.mean(recall_list)))
print('F1 Score: {0:0.2%}'.format(np.mean(f1_list)))
print()

rows_in_question = []
for i in range(len(metrics["y_test"])):
  if metrics["accuracy"][i] != 1 or metrics["precision"][i] != 1 or metrics["recall"][i] != 1 or metrics["f1 score"][i] != 1:
    rows_in_question.append(i)
poor_metrics = metrics.iloc[rows_in_question]

print("<<<< Worst Metrics >>>>>")
print(poor_metrics)
