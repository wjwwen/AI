# Business Case
# Assuming you are working in a hospital as a data sceintist
# You need to predict if a person has alzhemier or not (there are 4 categories).
# You are to create the predictive model using machine learning model (CNN) and Orange and compare the results between them with explanation.
# Please use the data from kaggle: 
# https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

# measurement is accuracy
# random state is 1
# Use seaborn heatmap and matplotlib plot.
# you are required to plot loss and accuracy curve (with tail to ensure no underfitting) and not too long (to ensure no overfitting).
# you should start with the default parameter CNN layer (32 32 64 64 3,3 filter with activation relu, maxpool 2,2 with stride 2, MLP of 128)
# how to improve the results by change the parameter setting
# Qualitatively, explain the pros and cons of your model
# How to overcome the weakness of your models (future study)
# the important of your model to the hostpital

# Good luck, Enjoy doing your homework

# %%
batch_size = 5

from keras.preprocessing.image import ImageDataGenerator

train_directory = "/Users/jingwen.wang/Downloads/Alzheimers Dataset/train/"
train_data = ImageDataGenerator().flow_from_directory(train_directory,
                                (150, 150), batch_size = batch_size, shuffle=False)
# Found 5121 images belonging to 4 classes.

train_directory = "/Users/jingwen.wang/Downloads/Alzheimers Dataset/test/"
test_data = ImageDataGenerator().flow_from_directory(train_directory,
                                (150, 150), batch_size = batch_size, shuffle=False)
# Found 1279 images belonging to 4 classes.

# %%
from matplotlib.pyplot import imshow
from keras.preprocessing.image import array_to_img

img = array_to_img(train_data[0][0][0])

imshow(img)

import os
num_class = 4

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

model = Sequential()

# %%
model.add(Conv2D(32, (3,3), input_shape=(150,150,3), activation="relu"))

# relu goes to positive infinity 
# relu has no upper limit; thus with many layers, not suitable for relu...
# when there is many layers, don't use relu, use sigmoid (as sigmoid's upper limit=1)

model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.2))

# %%
model.add(Conv2D(32, (3,3), activation="relu"))

model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.2))

# %%
model.add(Conv2D(64, (3,3), activation="relu"))

model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.2))

# %%
model.add(Conv2D(64, (3,3), activation="relu"))

model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.2))

# %%
model.summary()

# %% 
model.add(Flatten())

model.summary()
# flatten = 3136 (no. of image)) - 3136 to represent a photo

# %%
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(4, activation="softmax"))

# %% 
model.summary()

# %%
# 2 neurons - use cross entropy
import keras
import tensorflow as tf
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
history = model.fit(train_data, epochs=10) # epoch=10

# during epoch, look at cross entropy; advising weight change with entropy
# only when looking at accuracy, we consider false positive/negative
# thus we need to use confusion matrix to see the false 

model.evaluate(test_data)
# lower loss is better

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.title("Loss")
plt.plot(history.history["accuracy"])
plt.title("Accuracy")

import numpy as np
predict_x=model.predict(test_data) 
classes_x=np.argmax(predict_x,axis=1)

pred = model.predict_classes(test_data)
cm = confusion_matrix(test_data.classes,classes_x)
cm 

# %%
# Answer: Array([[106, 128], [6, 384]])
# Recommended to use seaborn to plot heatmap

import seaborn as sns
sns.heatmap(cm, annot=True)
print("accuracy is", (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/(sum(sum(cm))))
# accuracy: 0.5003909304143862

# %% 
# Qualitatively, explain the pros and cons of your model
# How to overcome the weakness of your models (future study)
# the important of your model to the hospital
'''
More often than not, Orange has a higher accuracy (i.e. Recall) 
than running the code on python.

Pros:
1. Speed advantage over RNN (CNN can be parallelized, RNN cannot)
2. Automatically detects important features without human supervision
3. High accuracy in image recognition problems

Cons:
1. CNN is a highly specific method, not suitable for many problems; perhaps look into visual recognition
2. Does not have benefits that some of the larger transformer models have
3. Computational power/complex architecture

The model is important to correctly classify the stage of Alzheimers. 
The speed and accuracy a machine learning algorithm provides for image classification is unparalleled.
'''