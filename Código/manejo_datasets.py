import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle


class validation_set: 
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test

class data_set:
	def __init__(self, validation_set, test_set):
		self.validation_set = validation_set
		self.test_set = test_set

def generate_train_test(file_name, folds_number):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['RainTomorrow'],axis=1).values #estructura X del mismo tamaño que y
	y = df['RainTomorrow'].values  
	#y = df.loc[:,['RainToday', 'RainTomorrow']].values

	print("target: ", y)
	
	#Separa el corpus cargado en el DataFrame en el 50% para entrenamiento y el 50% para pruebas
	#~ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle = True)

	#print("EL TIPO DE X_TRAIN ES: ", type(X_train))
	
	#print (X_train.shape)
	#print ("x train: > \n", X_train)
	#print (y_train.shape)
	#print ("y test: ",y_test)
	
	#~ #Crea pliegues para la validación cruzada
	validation_sets = []
	kf = KFold(n_splits=folds_number) #cambiando a 3 pliegues
	#print("kf:")
	for train_index, test_index in kf.split(X_train): #repite el mismo proceso de acuerdo a los folds que tengas
		print("TRAIN:", train_index, "\n",  "TEST:", test_index)
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		print("X_train_ es de la forma: ", X_train_)
		#print("El x train con train_index", X_train[train_index])
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		#~ #Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
	
	#Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)	
	
	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set) 
	
	
	return (my_data_set)
	
if __name__=='__main__':
	my_data_set = generate_train_test('weatherAUS2.csv',3) #devuelve un my_data_set wue es un data_set, por ello se puede acceder a sus dos argumentos como my_data_set.validation_set y el otro
	
	print ("el X_test despues de todo: ",my_data_set.test_set.X_test)
	# print(type(my_data_set.test_set.X_test))
	print ('\n----------------------------------------------------------------------------------\n')
	
	pliegues = 10
	#~ print (my_data_set.validation_set[0].y_train)
	
	#Guarda el dataset en formato csv
	# np.savetxt("data_test_X.csv", my_data_set.test_set.X_test, delimiter=",", fmt='%s',
    #        header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm")
	
	# np.savetxt("target_test_y.csv", my_data_set.test_set.y_test, delimiter=",", fmt='%s',
    #        header="RainToday,RainTomorrow", comments="")
    
	# #Para este proximo for, estoy accediendo a los validations sets por medio del constructor de data_set, el primer argumento son los validation_set
	# i = 1
	# for val_set in my_data_set.validation_set:
	# 	np.savetxt("data_validation_train_" +str(pliegues)+"pliegues_" + str(i) + ".csv", val_set.X_train, delimiter=",", fmt='%s',
    #        header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm", comments="")
	# 	np.savetxt("data_validation_test_"+str(pliegues)+"pliegues_" + str(i) + ".csv", val_set.X_test, delimiter=",", fmt='%s',
    #        header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm", comments="")
	# 	np.savetxt("target_validation_train"+str(pliegues)+"pliegues_" + str(i) + ".csv", val_set.y_train, delimiter=",", fmt='%s',
    #        header="RainTomorrow", comments="")
	# 	np.savetxt("target_validation_test_"+str(pliegues)+"pliegues_" + str(i) + ".csv", val_set.y_test, delimiter=",", fmt='%s',
    #        header="RainTomorrow", comments="")
	# 	i = i + 1
	
	# # #Guarda el dataset en pickle
	# dataset_file = open ('dataset.pkl','wb')
	# pickle.dump(my_data_set, dataset_file)
	# dataset_file.close()
	
	# dataset_file = open ('dataset.pkl','rb')
	# my_data_set_pickle = pickle.load(dataset_file)
	# print ("-----------------------------------------------")
	# print (my_data_set_pickle.test_set.X_test)












