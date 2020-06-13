import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
data=pd.read_csv('Dataset/Position_Salaries.csv')

x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
print(regressor.predict([[6.5]]))

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Reg-Tree')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.show()

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Reg-Tree-Smooth')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.show()