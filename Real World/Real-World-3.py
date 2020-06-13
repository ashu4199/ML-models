# SVR model with kernel rbf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

data=pd.read_csv('Dataset/Data 2.csv')

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

y=y.reshape((len(y),1))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

sc_x=StandardScaler()
sc_y=StandardScaler()

x_train=sc_x.fit_transform(x_train)
y_train=sc_y.fit_transform(y_train)

regressor=SVR(kernel='rbf')
regressor.fit(x_train,y_train)

y_pred=sc_y.inverse_transform(regressor.predict(sc_x.fit_transform(x_test)))
np.set_printoptions(precision=2)
print(np.concatenate(((y_pred.reshape(len(y_pred),1)),(y_test.reshape(len(y_test),1))),1))

print(r2_score(y_test,y_pred))

# ************** R2 score =0.948369