from numpy import genfromtxt
from codes import classifiers

training_data = genfromtxt('C:/Users/TRIPT/Downloads/training_data.csv', delimiter=',')
training_cl = genfromtxt('C:/Users/TRIPT/Downloads/training_data_class_labels.csv', delimiter=',')
test_data = genfromtxt('C:/Users/TRIPT/Downloads/test_data.csv', delimiter=',')


inp = input('''
          Choose a given option by typing the number:
              1,2,3,4 will give you the classification report of the chosen classifier classifier
              1  : Gaussian Naive Bayes Classifier
              2  : K Neighbors Classifier
              3  : Logistic Regression
              4  : Support Vector Machine
              5  : Get the best classifier for the given data along with a csv file named classlabels.csv & the scatterplot
              ''')
clk = classifiers()

if inp == '1':
    clk.GNB_Classifier(training_data, training_cl)
elif inp == '2':
    clk.KNN_Classifier(training_data, training_cl)
elif inp == '3':
    clk.LR(training_data, training_cl)
elif inp == '4':
    clk.SVM(training_data, training_cl)
elif inp == '5':
    clk.choose_classifier(training_data, training_cl, test_data)
