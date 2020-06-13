import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


data=pd.read_csv('Dataset/Position_Salaries.csv')

x=data.iloc[:,1:-1]
y=data.iloc[:,-1]

regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

print(regressor.predict([[6.5]]))

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Ran-Tree')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.show()

x_grid=np.arange(min(x),max(x),0.01)
x_grid=np.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Ran-Tree')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.show()