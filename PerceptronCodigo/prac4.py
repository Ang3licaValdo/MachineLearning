import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
#import csv
from  sklearn import preprocessing
from statistics import mean
class validation_set:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		
class train_set:
	def __init__(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
		

class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test
class data_set:
	def __init__(self, validation_set, test_set):
		self.validation_set = validation_set
		self.test_set = test_set

class data_set_final:
	def __init__(self, train_set, test_set):
		self.train_set = train_set
		self.test_set = test_set
		
def activation_function (predicted_values):
	threshold_values = []
	
	for value in predicted_values:
		if value <0:
			threshold_values.append(0)
		else:
			threshold_values.append(1)
	
	return (threshold_values)
	
def weight_adjustment(y_predicted, y_train, weights, x_train):
	for i in range(len(y_train)):
		#print ('y_train: {} - y_predicted: {}'.format(y_train[i], y_predicted[i]))
		error = y_train[i] - y_predicted[i]
		#print ('error: ', error)
		if error != 0:
			#print ('weights: {}'.format(weights)) 
			#print('x_train[i]: ',x_train[i])
			#print('error: ',error)
			weights += np.sum([weights,np.multiply(x_train[i], error)], axis=0)
			#print ('weights: {}'.format(weights)) 
	return (weights)

def generate_train_test(file_name):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['target'],axis=1).values   
	y = df['target'].values
	
	#Separa el corpus cargado en el DataFrame en el 70% para entrenamiento y el 30% para pruebas
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	
	
	#~ #Crea pliegues para la validación cruzada
	validation_sets = []
	kf = KFold(n_splits=3)
	for train_index, test_index in kf.split(X_train):
	
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		#~ #Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
	
	#Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)	
	
	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set) 
	
	return (my_data_set)
	
def generate_train_test_final(file_name):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['target'],axis=1).values   
	y = df['target'].values
	
	#Separa el corpus cargado en el DataFrame en el 50% para entrenamiento y el 50% para pruebas
	#~ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state=0)
	
	#~ print (X_train.shape)
	#~ print (X_train)
	#~ print (y_train.shape)
	#~ print (y_train)
	
	
	
	#Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)	
	my_train_set = train_set(X_train, y_train)
	
	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set_final = data_set_final(my_train_set, my_test_set) 
	
	return (my_data_set_final)
	
	
def perceptron_fun(my_data_set,epochs):

	
	accuracy_list=[]
	cont=0
	
	for val_set in my_data_set.validation_set:
		x_scaler_train = preprocessing.StandardScaler().fit_transform(val_set.X_train)
		x_scaler_test = preprocessing.StandardScaler().fit_transform(val_set.X_test)
		cont+=1	
		#weights = np.zeros((13,),dtype=int)
		weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		#print(cont)
		
		for i in range (epochs):
			#print ('-----valset:',cont,'-----------Iteración ', i, ' -------------------\n')
			#print(weights)
			vector = np.vectorize(np.int_)
			x_scaler_train = preprocessing.StandardScaler().fit_transform(val_set.X_train)
			weight_sums = np.dot(x_scaler_train,weights.T)
			#weight_sums = vector(weight_sums)
			#print ('weight_sums:\n', weight_sums)
			y_predicted = activation_function(weight_sums)
			#print ('y_predicted:', y_predicted)
			#print ('y_true: ', val_set.y_train)
			
			#print ('accuracy: ', accuracy_score(val_set.y_train, y_predicted))
			weights= vector(weights)
			x_scaler_train= vector(x_scaler_train)
			
			#y_scaler_train = preprocessing.StandardScaler().fit_transform(val_set.y_train)
			#x_scaler_train = preprocessing.StandardScaler().fit_transform(val_set.X_train)
			
			weights = weight_adjustment(y_predicted,val_set.y_train, weights, x_scaler_train)
	
		#print ('final weights :', weights)
		print ('final accuracy: ', accuracy_score(val_set.y_train, y_predicted))		
		accuracy_list.append(accuracy_score(val_set.y_train, y_predicted))
		
		#for x in val_set.X_test:
		result = np.dot(x_scaler_test,weights.T)
		#print("{}:{}->{}".format(x_scaler_test[:2],result,activation_function(result)))
		#print ('final accuracy: ', accuracy_score(val_set.y_test, activation_function(result)))
    		#print("{}:{}->{}".format(x[:2],result,activation_function(result)))
	
	print(mean(accuracy_list))
	
	return None
	
	
def perceptron_fun_final(my_data_set_final,epochs):

	
	accuracy_list=[]
	cont=0
	
			
	#weights = np.zeros((13,),dtype=int)
	weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	#print(cont)
	x_scaler_train = preprocessing.StandardScaler().fit_transform(my_data_set_final.train_set.X_train)
	x_scaler_test = preprocessing.StandardScaler().fit_transform(my_data_set_final.test_set.X_test)
		
	for i in range (epochs):
		#print ('-----valset:',cont,'-----------Iteración ', i, ' -------------------\n')
		#print(weights)
		vector = np.vectorize(np.int_)
		
		
		weight_sums = np.dot(x_scaler_train,weights.T)
		#weight_sums = vector(weight_sums)
		#print ('weight_sums:\n', weight_sums)
		y_predicted = activation_function(weight_sums)
		#print ('y_predicted:', y_predicted)
		#print ('y_true: ', val_set.y_train)
			
		#print ('accuracy: ', accuracy_score(val_set.y_train, y_predicted))
		#weights= vector(weights)
		x_scaler_train= vector(x_scaler_train)
			
		#y_scaler_train = preprocessing.StandardScaler().fit_transform(val_set.y_train)
		#x_scaler_train = preprocessing.StandardScaler().fit_transform(val_set.X_train)
			
		weights = weight_adjustment(y_predicted,my_data_set_final.train_set.y_train, weights, x_scaler_train)
	
	
	print ('final weights :', weights)
	print ('final accuracy: ', accuracy_score(my_data_set_final.train_set.y_train, y_predicted))		
	#accuracy_list.append(accuracy_score(my_data_set_final.train_set.y_train, y_predicted))
	
	result = np.dot(x_scaler_test,weights.T)
	y_predicted_test = activation_function(result)
	print("Imprimiendo el arreglo predicho del test: ", y_predicted_test)
	print("Imprimiendo el arreglo y actaul: ",my_data_set_final.test_set.y_test)
	print ('final accuracy- test: ', accuracy_score(my_data_set_final.test_set.y_test, y_predicted_test))	
	
	#print(mean(accuracy_list))
	
	return None
	
if __name__ == "__main__":
	#my_data_set=generate_train_test('heart.csv')
	my_data_set_final=generate_train_test_final('heart.csv')
	#print("1 epoca")
	#perceptron_fun(my_data_set,1)
	#print("2 epoca")
	#perceptron_fun(my_data_set,2)
	#print("5 epoca")
	#perceptron_fun(my_data_set,5)
	perceptron_fun_final(my_data_set_final,2)
	
	
	
	
