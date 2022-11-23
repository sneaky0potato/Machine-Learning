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


class classifiers:
    
    def GNB_Classifier(self, training_data, training_cl):
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(training_data, training_cl, test_size=0.30) 
        clf_GNB = GaussianNB() 
        clf_GNB.fit(trn_data, trn_cat)
        testGNB_predict = clf_GNB.predict(tst_data)
        print("KNeighborsClassifier, Classification Report: \n")
        print(classification_report(tst_cat, testGNB_predict))
        
    def KNN_Classifier(self, training_data, training_cl):
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(training_data, training_cl, test_size=0.30) 
        clf_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)  
        clf_KNN.fit(trn_data, trn_cat)
        testKNN_predict = clf_KNN.predict(tst_data)
        print("KNeighborsClassifier, Classification Report: \n")
        print(classification_report(tst_cat, testKNN_predict))
        
    def LR(self, training_data, training_cl):
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(training_data, training_cl, test_size=0.30) 
        clf_LR = LogisticRegression()
        clf_LR.fit(trn_data, trn_cat)
        testLR_predict = clf_LR.predict(tst_data)
        print("Logistic Regression, Classification Report: \n")
        print(classification_report(tst_cat, testLR_predict))
            
    def SVM(self, training_data, training_cl):
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(training_data, training_cl, test_size=0.30) 
        clf_SVM = svm.SVC()
        clf_SVM.fit(trn_data, trn_cat)
        testSVM_predict = clf_SVM.predict(tst_data)
        print("Support Vector Machine, Classification Report: \n")
        print(classification_report(tst_cat, testSVM_predict))
        
    def choose_classifier(self, training_data, training_cl, test_data):
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
            cl_SVM_predict = clf_SVM.predict(test_data)
            Result = np.array(cl_SVM_predict)
            print("Support Vector Machine Plot:  ")
        else:
            cl_KNN_predict = clf_KNN.predict(test_data)
            Result = np.array(cl_KNN_predict)
            print("K Neighbours Plot:  ")
        #plotting the result of the best classifier for the given test data
        test_data_x = []
        test_data_y = []
        
        for i in test_data:
            test_data_x.append(i[0])
            test_data_y.append(i[1])
        
        plt.scatter(test_data_x, test_data_y, c = Result, cmap=plt.cm.Accent, s=10)
        plt.show()
        
        #exporting the results into a csv file
        np.savetxt("classlabels.csv", Result, delimiter =" ",  fmt ='% s') 
        
    

        
    