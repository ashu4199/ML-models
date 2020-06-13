import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('../Dataset/Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

# cleaning and Stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]', ' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    ps=SnowballStemmer('english')
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)
    
# creating a bag of model
# print(corpus)
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
print(len(x[0]))
y=data.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# ********* Naive - Bayes Classification
classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print('Naive-bayes\n',confusion_matrix(y_test,y_pred))
print('Naive-bayes\n',accuracy_score(y_test,y_pred))

# ************* Random Forest Classifier
classi=RandomForestClassifier()

classi.fit(x_train,y_train)
y_pred_r=classi.predict(x_test)
print('Random Forest\n',confusion_matrix(y_test,y_pred_r))
print('Random Forest\n',accuracy_score(y_test,y_pred_r))

# ************** K Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred_k=knn.predict(x_test)

print('K-Neighbors\n',confusion_matrix(y_test,y_pred_k))
print('K-Neighbors\n',accuracy_score(y_test,y_pred_k))


# *************** SVM
from sklearn.svm import SVC
svm=SVC(kernel='rbf',random_state=0,degree=4)
svm.fit(x_train,y_train)

y_pred_sv=svm.predict(x_test)

print('SVM\n',confusion_matrix(y_test,y_pred_sv))
print('SVM\n',accuracy_score(y_test,y_pred_sv))


