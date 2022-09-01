import pandas as pd

df = pd.read_csv('german_credit.csv')
# Y is the creditability
# 20 X (i.e. columns)

df.isnull().any()
# No null values

from sklearn.feature_selection import f_regression, SelectKBest

# %%
# F-REGRESSION: feature selection criterion to identify potentially predictive feature
# for a downstream classifier, irrespective of the sign of the association with the target variable
# f_regression 
k = SelectKBest(score_func=f_regression,k=10)
# Selecting 10 columns

X = df.drop(columns="Creditability")
Y = df["Creditability"]

k.fit(X,Y).get_support()

# X.columns[k]
# X.columns

df = df.loc[:,['Creditability','Account Balance', 'Duration of Credit Mths',
       'Payment Status of Previous Credit', 'Credit Amount',
       'Value of Savings and Stocks', 'Length of current employment',
       'Sex and Marital Status', 'Most valuable available asset', 'Age',
       'Concurrent Credits']]

'''
df.boxplot(figsize=(20,20))
df.hist(figsize=(20,20))

# %%
# DATA VISUALISATIONS
import seaborn as sns

sns.heatmap(df.corr())
# looking at creditability, the lighter the better

sns.catplot(data=df, x="Creditability", y="Age")

sns.catplot(data=df, x="Creditability", y="Age", kind="violin")
'''
import matplotlib.pyplot as plt
'''
plt.scatter(df["Creditability"], df["Credit Amount"])
plt.title("Creditability")
plt.xlabel("Creditability")
'''

# %%
# 1. Create dummy variables
# 2. Normalize

# Decision-tree do not need normalisation
# Unstructured data relies on decision tree
# But if you see large numbers, MUST normalize
# Don't normalize categorical data... Must create dummy data

# Normalization: change numeric columns in dataset to use a common scale,
# without distorting differences in the ranges of values or losing information

# Normalize credit amount and age
from scipy import stats
df["Credit Amount"] = stats.zscore(df["Credit Amount"])
df["Age"] = stats.zscore(df["Age"])

dummy = pd.get_dummies(df["Sex and Marital Status"])#.value_counts()
dummy

df = df.merge(dummy, left_index=True, right_index=True).drop(columns='Sex and Marital Status')

# Check if data is balanced
df["Creditability"].value_counts()

# %% 
X = df.drop(columns="Creditability")
Y = df["Creditability"]

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=123)
Y_train.value_counts()

from imblearn.over_sampling import SMOTE
X_train, Y_train = SMOTE(random_state=123).fit_resample(X_train, Y_train)

# %%
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
model = linear_model.LogisticRegression(random_state=123)

model.fit(X_train, Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print("accuracy is", (cm[0,0]+cm[1,1]/sum(sum(cm))))

# %%
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state=123)

model.fit(X_train, Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print("accuracy is", (cm[0,0]+cm[1,1]/sum(sum(cm))))

# %%
from sklearn import ensemble
model = ensemble.RandomForestClassifier(random_state=123)

model.fit(X_train, Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print("accuracy is", (cm[0,0]+cm[1,1]/sum(sum(cm))))

# %%
model = ensemble.GradientBoostingClassifier(random_state=123)

model.fit(X_train, Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print("accuracy is", (cm[0,0]+cm[1,1]/sum(sum(cm))))

# %%
from sklearn import neural_network
model = neural_network.MLPClassifier(random_state=123)

model.fit(X_train, Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print("accuracy is", (cm[0,0]+cm[1,1]/sum(sum(cm))))

# %%
# ROC curve: show the performance of a classification model at all classification thresholds
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
plot_precision_recall_curve(model, X_test, Y_test)
plot_roc_curve(model, X_test, Y_test)

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

# RELU - Rectified Linear Unit (ReLU)
# RELU - piecewise linear function that will output the input directly if it is positive, otherwise, output zero
model.add(Dense(20, input_dim=13, activation="relu"))
model.add(Dropout(0.2))
# input layer = 13, hidden layer = 15
# usually, use 2/3 of the input layer i.e. 10 for hidden layer

model.add(Dense(15, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))
model.add(Dropout(0.2))

# loss - binary cross entropy

# Dropout necessary to perform well in both training and testing
# Dropout - should not learn every single neuron. 
# During backprop, e.g. only 80% of the neuron will learn i.e. 80% of neuron change weight, 20% don't change weight 
# Overfitting - train set super performance, test set underperform

model.summary()

# %%
model.compile(loss="binary_crossentropy", optimizer="adamax", metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size = 2, epochs=100)

plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("Loss Function")
plt.xlabel("epochs")

# %%
model.evaluate(X_test, Y_test)