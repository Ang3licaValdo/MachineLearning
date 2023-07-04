import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

class validation_set: #se cambio a mandarle mas argumentos
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test


def generate_train_test(file_name):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['price'],axis=1).values 
	y = df['price'].values  
	#y = df.loc[:,['RainToday', 'RainTomorrow']].values

	print("target: ", y)
	
	#Separa el corpus cargado en el DataFrame en el 50% para entrenamiento y el 50% para pruebas
	#~ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = True, random_state = 0)
	
	#print (X_train.shape)
	print ("x train: > \n", X_train)
	#print (y_train.shape)
	print ("y train: ",y_train)
	#print (X_train.shape)
	print ("x test: > \n", X_test)
	#print (y_train.shape)
	print ("y test: ",y_test)

	return validation_set(X_train, y_train, X_test, y_test)
	
	
def F(w, X, y):
    	return sum((w * x - y)**2 for x, y in zip(X, y))/len(y)


def dF(w, X, y):
	return sum(2*(w * x - y) * x for x, y in zip(X, y))/len(y)


def print_line(points, w, iteration, line_color = None, line_style = 'dotted'):
	list_x = []
	list_y = []
	for index, tuple in enumerate(points):
		x = tuple[0] #La posicion 0 de la tupla, que serían las x's
		y = x * w
		list_x.append(x)
		list_y.append(y)
	plt.text(x,y, iteration, horizontalalignment='right')
	plt.plot(list_x, list_y, color = line_color, linestyle= line_style)
	
if __name__=='__main__':
	my_data_set = generate_train_test('dataset_ejercicio_I_regresion_lineal.csv') #devuelve un my_data_set wue es un data_set, por ello se puede acceder a sus dos argumentos como my_data_set.validation_set y el otro

	#Train
	X_train = my_data_set.X_train
	y_train = my_data_set.y_train
	
	iterations = int(sys.argv[1])
	
	plt.scatter(X_train, y_train)
	
	w= 0
	alpha = float(sys.argv[2])
	#alpha = 0.01
	# ~ alpha = 0.05 #Efecto similar al de no sacar el promedio
	#Entrenamiento 
	#w = 11.62
	# print("SUMA loss function: ",sum((w * x - y)**2 for x, y in zip(X_train, y_train)))
	# print("tamaño de len loss function: ", len(y_train))
	# print("RESULTADO loss fucntion: ",sum((w * x - y)**2 for x, y in zip(X_train, y_train))/len(y_train))

	# print("SUMA gradiente: ",sum(2*(w * x - y) * x for x, y in zip(X_train, y_train)))
	# print("tamaño de len gradiente: ", len(y_train))
	# print("RESULTADO loss gradiente: ",sum(2*(w * x - y) * x for x, y in zip(X_train, y_train))/len(y_train))


	for t in range(iterations):
		loss_function = F(w, X_train, y_train)
		gradient = dF(w, X_train, y_train)
		w = w - alpha * gradient
		print ('iteration {}: w = {}, F(w) = {}'.format(t, w, loss_function))
		#print_line(zip(X_train, y_train), w, t)
	print ('mse training set: {}'.format(loss_function))
	print_line(zip(X_train, y_train), w, t, 'red', 'solid')
	plt.show()


	#Test
	X_test = my_data_set.X_test
	print("X test: ", X_test)
	y_test = my_data_set.y_test
	print("Y test: ", y_test)
	
	print ('Calculated weight: {}'.format(w))
	print ('Predictions')
	for x, y in zip(X_test, y_test):
		print ('true value: {}, predicted value {}'.format(y, x*w))

	loss_function = F(w, X_test, y_test)
	print ('mse test set: {}'.format(loss_function))
	plt.scatter(X_test, y_test)
	print_line(zip(X_test, y_test), w, 'prediction', 'red', 'solid')
	plt.show()	

	
	












