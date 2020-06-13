# Polynomial Regressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data=pd.read_csv('Dataset/Data 2.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)
regressor=LinearRegression()
regressor.fit(x_poly,y_train)

y_pred=regressor.predict(poly_reg.fit_transform(x_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),(y_test.reshape(len(y_test),1))),1))

print(r2_score(y_test,y_pred))
# *************** R2 score 0.945819