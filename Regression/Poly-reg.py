import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data=pd.read_csv('Dataset/Position_Salaries.csv')
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

lin_reg=LinearRegression()
lin_reg.fit(x,y)



poly_reg=PolynomialFeatures(degree=4)#degree 2 or 4
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)


plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear results')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(x_poly),color='blue')
plt.title('Polynomial')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()

#predict values 
print('Linear Model Prediction:')
print(lin_reg.predict([[6.5]]))

print('Polynomial Model prediction')
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))