# Business Case
# Assuming you are working in a bank and a customer is requesting for house loan
# You need to estimate the amount of loan you could grant to the customer
# Hence you need to estimate the value of the house
# You are to create model to predict the value of the house using machine learning model
# There are 10 attributes and you have finally decided on 'town', 'flat_type','floor_area_sqm','flat_model', 'lease_commence_date'.
# Remember to use dummy variables on categorical input => you should have 55 columns include Y
# Normalize floor_area_sqm using zscore before split (**)
# Remember to split train test using default rates
# All machine learning use default setting
# measurement is rmse
# random state is 1 (need to use in train test split and model creation)
# data visualization is needed. At least use one of pandas, seaborn and matplotlib.
# the results rmse on the test set should be (around) regression 62455, Decision Tree 43711,Random Forest 40131, GBoost 62325, NN 151107
# how to improve the results by change the features selected, data wrangling or parameter setting including using keras and tensorflow.
# Qualitatively, explain the pros and cons about all your models (5 models and Keras)
# How to overcome the weakness of your models (future study)
# the important of your model to the bank
# apply orange on all the models (5 models)
# Good luck, Enjoy doing your homework

# %%
# Purpose: ML Model to predict the value of the house
import pandas as pd

# %%
# Import Data
df = pd.read_csv('ResaleFlatPrice.csv')
df = df[['town', 'flat_type','floor_area_sqm','flat_model', 'lease_commence_date', 'resale_price']]

# %%
# Check nulls
df.isnull().any()

# %%
# Data Visualizations
import seaborn as sns
sns.heatmap(df.corr())
sns.catplot(data=df, x="resale_price", y="town", kind="violin")
sns.catplot(data=df, x="resale_price", y="flat_model", kind="violin")

'''
Data visualization comments:
1. A stronger correlation between resale_price and floor_area_sqm.
2. Resale price varies most significantly in the Central Area
3. Resale price varies most significantly for DBSS
'''

# %%
# Categorical Input - Use Dummy Variables
dummy = pd.get_dummies(df[['town', 'flat_type','flat_model']])
df = df.merge(dummy, right_index=True, left_index=True)
df = df.drop(columns=['town', 'flat_type','flat_model'])

# %%
# Normalize floor_area_sqm using zscore
from scipy import stats
df["floor_area_sqm"] = stats.zscore(df["floor_area_sqm"])

# %%
# Check no. of category of Y
df["resale_price"].value_counts()
X = df.drop(columns=["resale_price"])
Y = df["resale_price"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
Y_train.value_counts()

# %%
# Linear Model
# Ans: 62455
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
print("rmse is: ", (mean_squared_error(Y_test, pred)**0.5))

'''
Linear Model
Pros:
1. Simple to implement and understand
2. Overfitting can be reduced by regularisation
3. Quick computation

Cons:
1. Prone to noise; sensitive to outliers
2. Assumption of linearity (in the real world, usually untrue)
3. Assumption that data is independent (must check for multicollinearity)
'''

# %%
# Decision Tree Regressor
# Ans: 43711
from sklearn import tree 
model = tree.DecisionTreeRegressor(random_state=1)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
print("rmse is :", (mean_squared_error(Y_test,pred)**0.5))

'''
Decision Tree Regressor
Pros:
1. Good for Y/N, True/False questions i.e. decision-related
2. Non-parametric i.e. no assumption about the shape of data
3. Can capture nonlinear relationships

Cons:
1. Unstable with continuous numerical variables
2. Cannot be used in big data
3. Prone to overfitting
'''

# %%
# Random Forest Regressor
# Ans: 40131
from sklearn import ensemble
model = ensemble.RandomForestRegressor(random_state=1)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
print("rmse is :", (mean_squared_error(Y_test,pred)**0.5))

'''
Random Forest Regressor
Pros:
1. Ability to handle large datasets
2. Better accuracy in prediction compared with decision tree
3. Robust to outliers

Cons:
1. Random Forest are found to be biased while dealing with categorical variables
2. Slow training
3. Not suitable for linear methods with sparse features
'''

# %% 
# Gradient Boosting Regressor
# Ans: 62325
from sklearn import ensemble
model = ensemble.GradientBoostingRegressor(random_state=1)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
print("rmse is :", (mean_squared_error(Y_test,pred)**0.5))

'''
Gradient Boosting Regressor
Pros:
1. Higher accuracy than linear regression
2. Flexibility - can optimize on different loss functions
3. One of the most effective and powerful method

Cons:
1. May overemphasise outliers and cause overfitting; 
- use L1/L2 regularisation penalties or low learning rate
2. Computationally expensive
3. Less interpretative method
'''

# %% 
# Neural Network
# Ans: 151107
from sklearn import neural_network
model = neural_network.MLPRegressor(random_state=1)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
print("rmse is :", (mean_squared_error(Y_test,pred)**0.5))

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
# CSV Output for Orange
df.to_csv("ResaleFlat_ForOrange.csv")

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

# 54, 54, 54, 1, 
model.add(Dense(54, input_dim=54, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(54, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))
model.add(Dropout(0.2))

# model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(X_train, Y_train, batch_size=2, epochs=30)

model.evaluate(X_train, Y_train)
model.evaluate(X_test,Y_test)

history = model.fit(X_train,Y_train,batch_size=2,epochs=200)

'''
Keras
Pros:
1. Supports almost all neural network models
2. Keras integrated with Tensorflow, easing the customization of workflows
3. Highly developed documentation

Cons:
1. Slower speed using with Tensorflow
2. For low-performance models; not ideal for deep learning research/complex networks
3. Less support compared with Tensorflow
''' 

# %% 
import matplotlib.pyplot as plt
plt.plot(history.history["mse"])

plt.plot(history.history["loss"])
plt.title("Loss Function")
plt.xlabel("epochs")

r = model.evaluate(X_test,Y_test)
r[1] ** 0.5 # rmse

# Ans: 127001.56597459734

'''
Ways to improve the model:
1. Increase no. of neurons in the layer
2. Input more attributes such as GDP forecast (i.e. economic outlook)
3. Increase dataset size

Importance of the model to the bank:
1. Ability to process multiple attributes with speed and accuracy, better than traditional methods/the human brain
2. Backed by statistics to ensure reliability of future housing price and subsequently, loan amount 
'''
