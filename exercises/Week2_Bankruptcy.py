import pandas as pd

df = pd.read_csv('bankruptcy.csv')
# y class

df.isnull().any()

pd.set_option("display.max_rows", None)

from sklearn.feature_selection import SelectKBest, f_regression
X = df.drop(columns = "class")
Y = df["class"]

# f-statistics is a statistic used to test the significance 
# of regression coefficients in linear regression models.

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html

# Select features according to the k highest scores.

f = SelectKBest(score_func=f_regression, k = 10).fit(X,Y)
f # object
f = f.get_support()

X.columns[f]

f = df.loc[:, ['class', 'Attr3', 'Attr8', 'Attr11', 'Attr16', 'Attr17', 'Attr22', 'Attr26',
       'Attr35', 'Attr50', 'Attr51']]
# class = bankrupt or not; attribute that matters accordingly
# df = df.iloc[:, 0:11]

df["class"].value_counts()
df.describe()

df.boxplot(figsize=(20,20))

df.hist(figsize=(20,20))

import seaborn as sns
sns.heatmap(df.corr())

sns.catplot(data=df, x="class", y="Attr3")

import matplotlib.pyplot as plt
plt.scatter(df["class"], df["Attr3"])
plt.scatter(df["class"], df["Attr8"])
plt.title("Bankruptcy")
plt.xlabel("Factors")
plt.legend(["Working Capital", "Book Value"])

from scipy import stats
df["Attr8"] = stats.zscore(df["Attr8"])

X = df.drop(columns="class")
Y = df["class"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=158)
Y_train.value_counts()

from imblearn.over_sampling import SMOTE
# Synthetic Minority Oversampling Technique
X_train, Y_train, = SMOTE().fit_resample(X_train, Y_train)

from sklearn.metrics import confusion_matrix
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
CM = confusion_matrix(Y_test, pred)
print("accuracy", (CM[0,0]+CM[1,1])/(sum(sum(CM))))

from sklearn import ensemble
model = ensemble.GradientBoostingClassifier()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
CM = confusion_matrix(Y_test, pred)
print("accuracy", (CM[0,0]+CM[1,1])/(sum(sum(CM))))

model = ensemble.RandomForestClassifier()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
CM = confusion_matrix(Y_test, pred)
print("accuracy", (CM[0,0]+CM[1,1])/(sum(sum(CM))))