import pandas as pd 

df = pd.read_csv("/Users/jingwen/Desktop/AN6001 AI and Big Data/Week 1/DBS_SingDollar.csv")

dir(pd)
# modules for pandas

dir(df)

X = df.loc[:, ["SGD"]]
# row, column

X = df.loc[:,["SGD"]]
Y = df.loc[:,['DBS']]

# using x to predict y

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X,Y)
dir(model) # check 
# coef_, intercept_
pred = model.predict(X)
model.coef_
model.intercept_
#  DBS = -50.6 * SGD + 90.2

# --------------------- SSE, MSE, RMSE
# Error = Actual - Predict
# Error^2 
# Sum of Squared Error (SSE) = sum(Error^2)
# Mean of Squared Error = SSE/122 (i.e. total no. of data points)
# Root Mean Squared Error (RMSE) = MSE^0.5 (i.e. sqrt MSE)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(Y,pred)**0.5
print(rmse)
# print(rmse/Y.mean(axis=0)*100)

model.coef_
model.intercept_

import joblib
joblib.dump(model, "regression")
