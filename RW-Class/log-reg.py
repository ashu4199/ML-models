import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression

data=pd.read_csv('../Dataset/Data 3.csv')

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

# ******* 0.94736