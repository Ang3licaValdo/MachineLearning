from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('mnist_train.csv', sep=',', engine='python')
X_train = df.drop(['label'],axis=1).values   
y_train = df['label'].values


###TRAINING THE MODEL###
clf = MLPClassifier(random_state = 0, learning_rate = 'constant')

parameters = {'hidden_layer_sizes':[100,350], 'learning_rate_init':[0.001, 0.0001], 'activation': ('identity','logistic'), 'max_iter':[500, 1500]}

mlp = GridSearchCV(clf, parameters, scoring =['accuracy','recall_macro', 'precision_macro'], refit='recall_macro',return_train_score=True,cv =3)

mlp.fit(X_train, y_train)

best_parameters_without_scalation = mlp.best_params_
best_score_without_scalation = mlp.best_score_

print (best_parameters_without_scalation)
print (best_score_without_scalation)

###PREDICTION###
df = pd.read_csv('mnist_test.csv', sep=',', engine='python')
X_test = df.drop(['label'],axis=1).values   
y_test = df['label'].values

clf_pred = MLPClassifier(random_state = 0, learning_rate = 'constant', hidden_layer_sizes = 350, learning_rate_init = 0.0001, activation = 'logistic', max_iter=500)
clf_pred.fit(X_train, y_train)

y_pred = clf_pred.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("La accuracy es: ",accuracy)

#target_names = str(clf_pred.classes_)
print(classification_report(y_test, y_pred, target_names=['0', '1', '2', '3','4', '5', '6', '7', '8', '9']))
#print(classification_report(y_test, y_pred, target_names=target_names))
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])
print (cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3','4', '5', '6', '7', '8', '9'])
disp.plot()
plt.show()

##PARA LO DE OBTENER LAS IMAGENES DE LOS EQUIVOCADOS##
tupla2 = X_test.shape
m2 = tupla2[0]
for i in range(m2):
    if(y_test[i] != y_pred[i]):
        img = X_test[i]
        image = np.reshape(img, (28, 28))
        plt.title("Clase correcta: "+ str(y_test[i]) + " Clase predicha: "+ str(y_pred[i]))
        plt.imshow(image, cmap="Greys")
        plt.show()

