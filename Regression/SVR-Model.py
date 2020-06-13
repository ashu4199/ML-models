import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
data=pd.read_csv('Dataset/Position_Salaries.csv')

x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

y=y.reshape((10,1))

sc_x=StandardScaler()
x=sc_x.fit_transform(x)

sc_y=StandardScaler()
y=sc_y.fit_transform(y)

regressor= SVR(kernel='rbf')
regressor.fit(x,y)

y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print(y_pred)

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color='blue')
plt.title('SVR Model')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.show()

#improving resolution 

x_grid=np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color='blue')
plt.title('SVR Model High resolution')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.show()