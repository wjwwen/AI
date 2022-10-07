## Notes
### **Top 10 Must-Know Machine Learning Algorithms:**
1. Naive Bayes Classifier
2. K-Means Clustering
3. Support Vector Machine (SVM)
4. Apriori Algorithm
5. Linear Regression
6. Logistic Regression
7. Decision Tree
8. Random Forest
9. Artificial Neural Networks
10. Nearest Neighbours

### **AI Model and Popular Application**
Classifier (Supervised) ML
- Regression (Financial/Economic Prediction)
- Decision Tree (Operation Research)
- Random Forest (Operation Research)
- XG Boost (Operation Research)
- SVM (Text, Handwriting, Biol vogy Classification)
- Naive Bayesian (Document Classification, Word Frequency)
- Genetic Algorithm (Solar Collector, Antenna Location)
- Multi-Layer Perceptron (General Classification)
- Hidden Markov (Speech & Handwriting Recognition)
- Recurrent Neural Network (Speech & Handwriting + Time Series Analysis)
- Long Short Term Memory (LSTM) (Same as RNN)
- Generative Adversarial Network (Fashion, Art, Game)
- Convolution Neural Network (Picture Recognition)

### **AI Models** 
Clustering (Unsupervised)
1. K-Means (Market Segmentation) - KNN Classifier
2. Gaussian Mixture Model (Market Segmentation)

Others
1. Reinforcement Learning (Robot, Autonomous Vehicle)
2. Auto-Encoder (Reduce dimensions for images)
3. Kernel Density Estimator (Advanced data wrangling technique to reduce outlier/noise)
4. Manifold Learning (Feature selection for lage dimensions)
5. Blockchain
6. Robotic Process Automation (RPA)
7. Chatbot (Automation)

### Feature Selection
- Kernel Density Estimator
- Manifold Learning (PCA)
- Correlation Coefficient
- Random Forest

### Accuracy v.s. Interpretability
![Image](https://miro.medium.com/max/1400/0*fvrBaILFAaCCsYHT)

### Categorical/Classification: Accuracy
![Image](https://shuzhanfan.github.io/assets/images/confusion_matrix.png)

### Continuous/Regression
- MSE
- RMSE

### Generalization & Regularization
- Training loss: how well the model fits the training data
- Validation loss: how well the model fits new data

Methods to avoid overfitting:
- Lasso/Ridge (Regression)
- Pruning (Decision Tree)
- Dropout (Neural Network)

Method to check outlier:
- Interquartile Range (IQR)

Imbalance Data (Oversampling/Undersampling) - SMOTE
- Have to oversample the minority on the train set to ensure that the test set is correct.

## Week 2 & 3: Neural Network
- Two hidden layers as a means of prudence for the model to work harder
- Weights (aka coefficients): connect neurons in one layer to neurons in the next layer

### NN Network Architecture
1. Select no. of hidden layers
2. Select no. of nodes in hidden layers
3. Set Learning Rate
4. Choose combination & activation functions at each layers

# Keras
## 1. Make sequential model/layer
- LSTM/RNN

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

model=Sequential()

# Network Design
# Layer --> Drop Out --> Activation Function
# Sigmoid/ReLU

model.add(Dense(10, input_dim=16, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))
```

## 2. Compile function (containing loss function)
```python
# Loss function, optimizer,m etrics
model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy’])
# all parameters are needed
```

**Losses**
- Mean Squared Error (Continuous)
- Binary Cross Entropy (Discrete for 2 classes)
- Categorical Cross Entropy (Discrete for >2 classes)

**Softmax/Cross Entropy**
- Softmax Loss: Softmax Activation + Cross Entropy Loss
- Softmax: activation function that outputs probability for each class (sum up to 1)
- Cross Entropy loss (logarithm loss/negative logarithm): measures perforamnce of classification model whose output is a 0 < probability < 1 

**Optimizer/Loss/Metrics**
- Adam()
- MeanSquaredError()
- True Positive/False Positive/False Negative/True Negative

## 3. Fit - Epoch/Batch Size
```python
model.fit(X_train, Y_train, batch_size = 10, epochs=15, verbose=1)
Batch size is optional

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```

## 4. Evaluate after training
```python
model.evaluate(X_train, Y_train)
model.evaluate(X_test, Y_test)

import seaborn as sns

pred = model.predict_classes(X_test)
cm = confusion_matrix(Y_test, pred)
sns.heatmap(cm, annot=True)

#depends on number of classes
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))
```

## 5. Prediction on test set
```python
import numpy as np
from sklearn.metrics import confusion_matrix

pred=model.predict(X_train)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Y_train, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)

pred=model.predict(X_test)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Y_test, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)
```

## 6. Summary/Load + Test
```python
model.summary()

model.weights

from keras.models import load_model

model = load_model("C:/Users/User/Dropbox/TT Library/AI Model/Neural Network/NN")
model.summary()
pred = model.predict([[0.5, 0.1, 0.01]])
```
# Machine Learning Programming in Python (20 Steps) for Structured Data
## 1. Import data
## 2. Decide on Target/Y/Dependent Variable
## 3. Check nulls
```python
df.isnull().any()
pd.set_option("display.max_rows", 10) # if needed
```
## 4. Select features
```python
# Clean data: Kbest
# Dirty data: Correlation

# Kbest – Need to split X,Y
X = df.drop(columns="Creditability")
Y = df["Creditability"]
from sklearn.feature_selection import SelectKBest, f_regression
f = SelectKBest(score_func=f_regression, k=10).fit(X,Y).get_support()
X.columns[f)
df = df.loc[:, ["Creditability", 'Account Balance', 'Duration of Credit Mths',
       'Payment Status of Previous Credit', 'Credit Amount’]]

# Correlation
pd.set_option("max_rows", None)
c = df.corr()
c = abs(c["fund_treynor_ratio_5years"])
c[c>0.2].sort_values()
df["sector_technology"].isnull().sum() #check if selected columns many nulls
pd.set_option("max_rows", 10)
```

## 5. Clean - Drop NA()
## 6. Data Visualization
```python
df.describe() #check mean : too big then normalization, max or min too much difference, then remove (outlier)
df.boxplot(figsize=(40,10)) #check for outlier
df.hist(figsize=(20,20))
import seaborn as sns
sns.heatmap(df.corr())
sns.catplot(data=df,x="Class", y="EBIT“, kind="violin") #kind=violin, bar, nothing means scatter
```

## 7. Categorical Input - Use Dummy variables
```python
dummy = pd.get_dummies(df["Sex and Marital Status"])
df = df.merge(dummy, right_index=True, left_index=True)
df = df.drop(columns="Sex and Marital Status")
```

## 8. Remove outliers
```python
from scipy import stats
z = abs(stats.zscore(df))
df = df[(z<5).all(axis=1)]

import numpy as np
z = abs(stats.zscore(df.astype(np.float)))
```

## 9. Normalization
```python
df["Credit Amount"] = stats.zscore(df["Credit Amount"])
```

![Image](https://www.w3resource.com/w3r_images/pandas-dataframe-describe-2.svg)

## 10. Check no. of category of Y and imbalance (use SMOTE if imbalanced)

```python
df["Creditability"].value_counts()
X = df.drop(columns=["Creditability"])
Y = df["Creditability"]
from sklearn.model_selection import train_test_splitX_train, X_test, Y_train, Y_test = train_test_split(X,Y)
Y_train.value_counts()
from imblearn.over_sampling import SMOTE
X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)
```

## 11. Split data in X, Y (if imbalance done, skip step)

## 12. Split into train/test (if imbalance done, skip step)

## 13. Import model

## 14. Train on train set

## 15. Train on test set
- RMSE if continuous Y
- Accuracy if categorical Y, use confusion matrix to ensure no bias in prediction on 1 category

## 16. Save model using joblib

## 17. Optimization
- A must for Decision Tree, else overfitting

## 18. Frontend (HTML)

## 19. Backend (Flask)

## 20. Cloud (Heroku/Github)

# Week 8 - CNN
Measurement of loss: Gini Index v.s. Entropy
- The smaller, the better
- Natural log (2.718) v.s. log 

- During epoch, look at cross entropy; advising weight change with entropy
- Only when looking at accuracy, we consider false positive/negative

# Week 10 - Recap
- Feature Extraction
- Classification - Python/Orange 
- No Machine Learning: CNN (32, 32, 64, 64)
- Machine Learning: 128
- ResNet: 2048
- VGG16: 16 layers that have weight
- Image Embedding
- Clustering: K-Means Clustering

# Week 10 - Speech
- Wavelength
- Transform to a frequency domain then decode it
- MaxPool: Blur the image
- Transformer is suitable for fast but less accurate
