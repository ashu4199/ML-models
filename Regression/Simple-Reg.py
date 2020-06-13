import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('Dataset/Salary_Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
test=regressor.predict(x_test)

plt.scatter(x_train,y_train,color ='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# test set result
plt.scatter(x_test,y_test,color ='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# To print single prediction: print(regressor.predict([[12]]))
# to print parameters print(regressor.coef_) and print(regressor.intercept_)