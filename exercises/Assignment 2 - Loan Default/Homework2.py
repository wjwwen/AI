# Business Case
# Assuming you are working in a bank and a customer is requesting for loan
# You need to predict if the customer will default the loan
# You are to create the predictive model using machine learning model
# Please use all the attributes except the index

# Remember to split train test using default rates
# use dummy variables for Employed
# split before normalize and oversample the minority

# Normalize annual salary and Bank Balance using zscore

# All machine learning use default setting
# measurement is accuracy
# random state is 1 (need to use in train test split and model creation)

# data visualization is needed. At least use one of pandas, seaborn and matplotlib.

# the accuracy on test set should be regression 0.6024, Decision Tree 0.7148,Random Forest 0.6692, GBoost 0.606, NN 0.5992

# you are required to plot ROC, precision_recall_curve
# how to improve the results by change the features selected, data wrangling or parameter setting
# Qualitatively, explain the pros and cons about your model
# How to overcome the weakness of your model (future study)
# the important of your model to the bank
# Good luck, Enjoy doing your homework

# %%
import pandas as pd

# %%
# 1. Import Data
# Employed, Bank Balance, Annual Salary, Defaulted?
df = pd.read_csv('Loan Default 2.csv')
df = df.loc[:,['Employed', 'Bank Balance', 'Annual Salary', 'Defaulted?']]

# %%
# 3. Check nulls
df.isnull().any()

# %%
# 5. Drop NA - None
# 6. Data Visualizations
df.describe()
df.boxplot(figsize=(40,10)) # check for outlier
df.hist(figsize=(20,20))

import seaborn as sns
sns.boxplot(data=df, x="Defaulted?", y="Bank Balance")
sns.boxplot(data=df, x="Defaulted?", y="Employed")

# %%
# 7. Employed - Use Dummy Variables
dummy = pd.get_dummies(df["Employed"])
df = df.merge(dummy,left_index=True,right_index=True).drop(columns='Employed')

# %%
# Check no. of category of Y and imbalance (SMOTE)
df["Defaulted?"].value_counts()
X = df.drop(columns="Defaulted?")
Y = df["Defaulted?"]

# Split before normalize
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
Y_train.value_counts()

from imblearn.over_sampling import SMOTE
X_train, Y_train = SMOTE(random_state=1).fit_resample(X_train, Y_train)

# %%
# Normalize annual salary and Bank Balance using zscore
from scipy import stats
X_train["Annual Salary"] = stats.zscore(X_train["Annual Salary"])
X_train["Bank Balance"] = stats.zscore(X_train["Bank Balance"])

X_test["Annual Salary"] = stats.zscore(X_test["Annual Salary"])
X_test["Bank Balance"] = stats.zscore(X_test["Bank Balance"])

# %%
# Regression 0.6024
from sklearn.metrics import confusion_matrix
from sklearn import linear_model

model = linear_model.LogisticRegression(random_state=1)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test,pred)
print("accuracy is ",(cm[0,0]+cm[1,1])/(sum(sum(cm))))

'''
Logistic Model
Pros:
1. Can easily be extended to multi-class classification i.e. multinomial
2. Shorter training time compared with neural network
3. Less prone to overfitting on low-dimensional dataset

Cons:
1. More prone to overfitting on high-dimensional dataset
2. Requires that independent variables are linearly related to the log odds (log(p/(1-p)).
3. Sensitive to outliers
'''

# %%
# Decision Tree 0.7148
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state=1)

model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test,pred)
print("accuracy is ",(cm[0,0]+cm[1,1])/(sum(sum(cm))))

'''
Decision Tree Classifier
Pros:
1. Compared to other algorithms, decision trees require less effort for data preparation during pre-processing
2. Decision tree does not require normalisation of data
3. Decision tree does not require scaling of data

Cons:
1. A small change in data can cause a large change in the structure of the decision tree, causing instability
2. Often involves higher time to train the model
3. Inadequate for applying regression and predicting continuous values
'''

# %%
# Random Forest 0.6692
from sklearn import ensemble

model = ensemble.RandomForestClassifier(random_state=1)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test,pred)
print("accuracy is ",(cm[0,0]+cm[1,1])/(sum(sum(cm))))


'''
Random Forest Classifier
Pros:
1. Ability to handle large datasets
2. Better accuracy in prediction compared with decision tree
3. Robust to outliers by binning variables

Cons:
1. "Black box" algorithm, very little control over what the model does
2. Slow training
3. Not suitable for linear methods with sparse features
'''

# %%
# GBoost 0.606
model = ensemble.GradientBoostingClassifier(random_state=1)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test,pred)
print("accuracy is ",(cm[0,0]+cm[1,1])/(sum(sum(cm))))

'''
Gradient Boosting Classifier
Pros:
1. Higher accuracy 
2. Flexibility - can optimize on different loss functions
3. Curbs overfitting easily

Cons:
1. May overemphasise outliers and cause overfitting; 
- use L1/L2 regularisation penalties or low learning rate
2. Sensitive to outliers
3. Less interpretative method
'''

# %%
# NN 0.5992
from sklearn import neural_network
model = neural_network.MLPClassifier(random_state=1)

model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test,pred)
print("accuracy is ",(cm[0,0]+cm[1,1])/(sum(sum(cm))))

'''
Neural Network
Pros:
1. Good to model nonlinear data with large no. of inputs e.g. images
2. User flexibility in terms of inputs/layers
3. Works well with more data points

Cons:
1. Black box - we do not know how much each independent variable is influencing the dependent variables
2. Computationally expensive
3. Depend on more training data
''' 

# %%
# ROC curve: show the performance of a classification model at all classification thresholds
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
plot_precision_recall_curve(model, X_test, Y_test)
plot_roc_curve(model, X_test, Y_test)

# %%
# CSV file for Orange
df.to_csv("LoanDefault_Updated.csv")

# %%
# Use Keras to make better model
# Accuracy: 0.7956
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

# 4, 4, 4, 1
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
model.add(Dropout(0.2))

# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=2, epochs=40)

model.evaluate(X_train, Y_train)
model.evaluate(X_test,Y_test)

'''
Keras
Pros:
1. Supports almost all neural network models
2. Keras integrated with Tensorflow, easing the customization of workflows
3. Highly developed documentation

Cons:
1. Slower speed using with Tensorflow
2. For low-performance models; not ideal for deep learning research/complex networks
3. Requires more epochs for higher performane, which means more time to train
''' 

# %% 
import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"])

plt.plot(history.history["loss"])
plt.title("Loss Function")
plt.xlabel("epochs")

model.evaluate(X_test,Y_test)

'''
Ways to improve the model:
1. Increase no. of neurons in the layer
2. Input more attributes 
3. Increase dataset size

Importance of the model to the bank:
1. Ability to process multiple attributes with speed and accuracy, better than traditional methods/the human brain
2. Backed by statistics to determine accuracy of prediction on default
'''