from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv', sep=',', engine='python')
X = df.drop(['output'],axis=1).values   
y = df['output'].values
#Separa el corpus cargado en el DataFrame en entrenamiento y el pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 0 )

parameters = {'n_neighbors':[1,3,5,10], 'weights': ('uniform', 'distance')}

# ###Training data without scalation###
clf = KNeighborsClassifier()

cv = GridSearchCV(clf, parameters, verbose=3)
cv.fit(X_train, y_train)

best_parameters_without_scalation = cv.best_params_
best_score_without_scalation = cv.best_score_

print (best_parameters_without_scalation)
print (best_score_without_scalation)

###TRAINING DATA WITH ESCALATION###
scaler_names = ['Escalado estándar', 'Escalado robusto']
scalers = [preprocessing.StandardScaler(), preprocessing.RobustScaler()]
clf = KNeighborsClassifier()
parameters_pipeline = [{'clf__n_neighbors':[1,3,5,10], 'clf__weights': ('uniform', 'distance')}]


for scaler_name, scaler in zip(scaler_names, scalers):
	print ('Método de escalado: ', scaler_name)
	
	pipeline = Pipeline([
					('scalers', scaler),
					('clf', clf)
					])
					
	cv = GridSearchCV(pipeline, param_grid=parameters_pipeline, verbose=3)
	cv.fit(X_train, y_train)
	print (cv.best_score_)
	print (cv.best_params_)


###PREDICCION OFICIAL###
X_train_escalado = preprocessing.StandardScaler().fit_transform(X_train)
X_test_escalado = preprocessing.StandardScaler().fit_transform(X_test)
cl_pred = KNeighborsClassifier(n_neighbors=10, weights = 'uniform')

cl_pred.fit(X_train_escalado, y_train)

y_pred = cl_pred.predict(X_test_escalado)

accuracy = accuracy_score(y_test, y_pred)
print("La accuracy es: ",accuracy)

print(classification_report(y_test, y_pred, target_names=['0','1']))

cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print (cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0','1'])
disp.plot()
plt.show()


