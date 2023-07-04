import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

class validation_set: 
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class data_set:
    def __init__(self, validation_set):
        self.validation_set = validation_set
        #self.test_set = test_set

def dataset_iris():
    #Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
    df = pd.read_csv('iris.csv', sep=',', engine='python')
    X = df.drop(['species'],axis=1).values #estructura X del mismo tamaño que y
    y = df['species'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 0 )

    # ##K FOLDS##
    # promedio_accuracy = []
    # validation_sets = []
    # promedio_accuracy_mult = []
    # kf = KFold(n_splits= 3) #cambiando a 3 pliegues
	# #print("kf:")
    # for train_index, test_index in kf.split(X_train): #repite el mismo proceso de acuerdo a los folds que tengas
    #     X_train_, X_test_ = X_train[train_index], X_train[test_index]
    #     y_train_, y_test_ = y_train[train_index], y_train[test_index]
    #     validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
	
	# #Guarda el dataset con los pliegues del conjunto de validación
    # my_data_set_validation = data_set(validation_sets) 

    # i = 1

    # for val_set in my_data_set_validation.validation_set:
    #     clf = GaussianNB()
    #     clf_2 = MultinomialNB()
    #     clf.fit(val_set.X_train, val_set.y_train)
    #     clf_2.fit(val_set.X_train, val_set.y_train)

    #     y_predict = clf.predict(val_set.X_test)
    #     y_predict_mult = clf_2.predict(val_set.X_test)
    #     print('------------Gaussian NB fold ' + str(i)+' +------------')
    #     print (y_predict)
    #     accuracy_per_fold = accuracy_score(val_set.y_test, y_predict)
    #     print (accuracy_per_fold)
    #     print (accuracy_score(val_set.y_test, y_predict, normalize=False))
    #     promedio_accuracy.append(accuracy_per_fold)

    #     print('------------Multinomial NB fold ' + str(i)+' +------------')
    #     print(y_predict_mult)
    #     accuracy_per_fold_mult = accuracy_score(val_set.y_test, y_predict_mult)
    #     print (accuracy_per_fold_mult)
    #     print (accuracy_score(val_set.y_test, y_predict_mult, normalize=False))
    #     promedio_accuracy_mult.append(accuracy_per_fold_mult)
        

    #     i = i+1
    
    # print("Promedio accuracy de los 3 folds Gaussiano= ", sum(promedio_accuracy)/3)
    # print("Promedio accuracy de los 3 folds Multinomial = ", sum(promedio_accuracy_mult)/3)

    ##PREDICCION FINAL USANDO GAUSSIANO##
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)

    print (accuracy_score(y_test, y_predict))
    print (accuracy_score(y_test, y_predict, normalize=False))

    target_names =clf.classes_
    print(classification_report(y_test, y_predict, target_names=target_names))
    cm = confusion_matrix(y_test, y_predict, labels=target_names)
    print (cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()




def dataset_email():
    #Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
    df = pd.read_csv('emails.csv', sep=',', engine='python')
    X = df.drop(['Prediction','Email No.'],axis=1).values #estructura X del mismo tamaño que y
    y = df['Prediction'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 0 )

    ## K FOLDS##
    # promedio_accuracy = []
    # validation_sets = []
    # promedio_accuracy_mult = []
    # kf = KFold(n_splits= 3) #cambiando a 3 pliegues
	# #print("kf:")
    # for train_index, test_index in kf.split(X_train): #repite el mismo proceso de acuerdo a los folds que tengas
    #     X_train_, X_test_ = X_train[train_index], X_train[test_index]
    #     y_train_, y_test_ = y_train[train_index], y_train[test_index]
    #     validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
	
	# #Guarda el dataset con los pliegues del conjunto de validación
    # my_data_set_validation = data_set(validation_sets) 

    # i = 1

    # for val_set in my_data_set_validation.validation_set:
    #     clf = GaussianNB()
    #     clf_2 = MultinomialNB()
    #     clf.fit(val_set.X_train, val_set.y_train)
    #     clf_2.fit(val_set.X_train, val_set.y_train)

    #     y_predict = clf.predict(val_set.X_test)
    #     y_predict_mult = clf_2.predict(val_set.X_test)
    #     print('------------Gaussian NB fold ' + str(i)+' +------------')
    #     print (y_predict)
    #     accuracy_per_fold = accuracy_score(val_set.y_test, y_predict)
    #     print (accuracy_per_fold)
    #     print (accuracy_score(val_set.y_test, y_predict, normalize=False))
    #     promedio_accuracy.append(accuracy_per_fold)

    #     print('------------Multinomial NB fold ' + str(i)+' +------------')
    #     print(y_predict_mult)
    #     accuracy_per_fold_mult = accuracy_score(val_set.y_test, y_predict_mult)
    #     print (accuracy_per_fold_mult)
    #     print (accuracy_score(val_set.y_test, y_predict_mult, normalize=False))
    #     promedio_accuracy_mult.append(accuracy_per_fold_mult)
        

    #     i = i+1
    
    # print("Promedio accuracy de los 3 folds Gaussiano= ", sum(promedio_accuracy)/3)
    # print("Promedio accuracy de los 3 folds Multinomial = ", sum(promedio_accuracy_mult)/3)

    ##PREDICCION FINAL USANDO GAUSSIANO##
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)

    print (accuracy_score(y_test, y_predict))
    print (accuracy_score(y_test, y_predict, normalize=False))

    target_name =clf.classes_
    print("target names: ", target_name)
    print(classification_report(y_test, y_predict, target_names=['0','1']))
    cm = confusion_matrix(y_test, y_predict, labels=[0,1])
    print (cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_name)
    disp.plot()
    plt.show()

#dataset_iris()
dataset_email()

