
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/TRIPT/Downloads/data.csv")
X = df[['x']].values
Y = df[['y']].values
data  = df[['x', 'y']]

from sklearn.preprocessing import StandardScaler, QuantileTransformer
scaler = StandardScaler()
scaler1 = QuantileTransformer(100)
X_new = scaler1.fit_transform(X)
Y_new = scaler1.fit_transform(Y)
data_new = scaler1.fit_transform(data)
plot1 = plt.scatter(X_new,Y_new, s = 5)
plt.show(plot1)
km = KMeans(n_clusters=2)
y_predicted  = km.fit_predict(data_new)
y_predicted
plot2 = plt.scatter(X_new,Y_new, c = y_predicted)
plt.show(plot2)
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.109,min_samples=145, p = 2)

model0 = dbscan.fit(data_new)

labels = model0.labels_

print(labels)

plot3 = plt.scatter(X_new,Y_new, c = labels, s = 10)
plt.show(plot3)
i = 0
A = []
for label in labels:
    if label == -1:
        print(i)
        A.append(i)
        
    i = i+1
    

labels_1 = np.delete(labels, A)    
X_new_1 = np.delete(X_new, A)
Y_new_1 = np.delete(Y_new, A)
class_data_X = []
class_data_Y = []
for i in A:
    class_data_X = np.append (class_data_X, X_new[i])
    class_data_Y = np.append (class_data_Y, Y_new[i])    
class_data = np.stack((class_data_X, class_data_Y), axis = 1)
training_data = np.stack((X_new_1, Y_new_1), axis=1)
training_cl = labels_1


import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


trn_data, tst_data, trn_cat, tst_cat = train_test_split(training_data, training_cl, test_size=0.30) 
#Gaussian Naive Bayes Classifier
clf_GNB = GaussianNB()
clf_GNB.fit(trn_data, trn_cat)
testGNB_predict = clf_GNB.predict(tst_data)
print("Naive Bayes Classifier, Classification Report: \n")
print(classification_report(tst_cat, testGNB_predict))


#K Neighbors Classifier
clf_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)  
clf_KNN.fit(trn_data, trn_cat)
testKNN_predict = clf_KNN.predict(tst_data)
print("KNeighborsClassifier, Classification Report: \n")
print(classification_report(tst_cat, testKNN_predict))


#Logistic Regression
clf_LR = LogisticRegression()
clf_LR.fit(trn_data, trn_cat)
testLR_predict = clf_LR.predict(tst_data)
print("Logistic Regression, Classification Report: \n")
print(classification_report(tst_cat, testLR_predict))

        
#Support Vector Machine
clf_SVM = svm.SVC()
clf_SVM.fit(trn_data, trn_cat)
testSVM_predict = clf_SVM.predict(tst_data)
print("Support Vector Machine, Classification Report: \n")
print(classification_report(tst_cat, testSVM_predict))


#Picking the best classifier 
f1_GNB = f1_score(tst_cat, testGNB_predict, average='macro')
f1_KNN = f1_score(tst_cat, testKNN_predict, average='macro')
f1_LR  = f1_score(tst_cat, testLR_predict, average='macro')
f1_SVM = f1_score(tst_cat, testSVM_predict, average="macro")
        
#Camparing the f1 scores to determine the best classifier
dict1 = {'NaiveBayes':f1_GNB, 'KNN':f1_KNN, 'LogisticRegression':f1_LR, 'SupportVectorMachine':f1_SVM}

print(max(dict1, key=dict1.get))

#printing the type and value of the best classifier
print(f"The best classifier is given by {max(dict1, key=dict1.get)} and it's value is {dict1.get(max(dict1, key=dict1.get))}. So, we will use {max(dict1, key=dict1.get)} for classification of the class")

if f1_KNN < f1_SVM:
            cl_SVM_predict = clf_SVM.predict(tst_data)
            Result = np.array(cl_SVM_predict)
            print("Support Vector Machine Plot:  ")
else:
            cl_KNN_predict = clf_KNN.predict(tst_data)
            Result = np.array(cl_KNN_predict)
            print("K Neighbours Plot:  ")
        #plotting the result of the best classifier for the given test data
test_data_x = []
test_data_y = []
        
for i in tst_data:
            test_data_x.append(i[0])
            test_data_y.append(i[1])
        
plt.scatter(test_data_x, test_data_y, c = Result, cmap=plt.cm.Accent, s=10)
plt.show()
        
#exporting the results into a csv file
np.savetxt("classlabels.csv", Result, delimiter =" ",  fmt ='% s') 
Result_1 = clf_SVM.predict(class_data)

print(Result_1)
j = 0
k = 0
for i in labels:
    if i<0:
        labels[j] = Result_1[k]
        k = k + 1
    j = j+1

plot3 = plt.scatter(X_new,Y_new, c = labels, s = 10)
plt.show(plot3)

np.savetxt("classlabelsmlassign2.csv", labels, delimiter =" ",  fmt ='% s') 