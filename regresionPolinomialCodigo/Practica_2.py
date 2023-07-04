import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator
import pandas as pd
from  sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

mse_lista = []
r2_lista = []

class mse_r2:
    def __init__(self,mse,r2):
        self.mse = mse
        self.r2 = r2

def regresion_lineal_SGD(iteraciones, eta_cero,):
    i = 1


    ##PARA REGRESION LINEAL SGD CON DATOS SIN ESCALAR####
    
    # for n in range(10):

    #         nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
    #         nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"
    #         #Obteniendo valores de los archivos generados de los k-folds para X
    #         df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
    #         X = df.values 
    #         #Obteniendo valores de y
    #         df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
    #         y = df2.values.ravel()

    #         #DATOS PARA HACER LA PREDICCION#
    #         x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
    #         y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

    #         #x_test
    #         df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
    #         x_test = df4.values
    #         #y test
    #         df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
    #         y_test = df3.values.ravel()
    #         regr = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
    #         regr.fit(X, y)
    #         y_poly_pred = regr.predict(x_test)

    #         mse = mean_squared_error(y_test, y_poly_pred)
    #         r2 = r2_score(y_test, y_poly_pred)

    #         i = i+1

    #         mse_lista.append(mse)
    #         r2_lista.append(r2)


    ###PARA REGRESION LINEAL SGD STANDARD SCALER###
 

    # for n in range(10):

    #     nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
    #     nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"
    #         #Obteniendo valores de los archivos generados de los k-folds para X
    #     df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
    #     X = df.values 
    #         #Obteniendo valores de y
    #     df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
    #     y = df2.values.ravel()

    #         #DATOS PARA HACER LA PREDICCION#
    #     x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
    #     y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

    #         #x_test
    #     df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
    #     x_test = df4.values
    #         #y test
    #     df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
    #     y_test = df3.values.ravel()       
        



    #     x_SGD_standard_scaler = preprocessing.StandardScaler().fit_transform(X)
    #     x_SGD_standard_scaler_test = preprocessing.StandardScaler().fit_transform(x_test)
    #     regr = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
    #     regr.fit(x_SGD_standard_scaler,y)
    #     #PREDICCION#
    #     y_SGD_standard_scaler_prediction = regr.predict(x_SGD_standard_scaler_test)

    #     mse = mean_squared_error(y_test, y_SGD_standard_scaler_prediction)
    #     r2 = r2_score(y_test, y_SGD_standard_scaler_prediction)

    #     i = i+1

    #     mse_lista.append(mse)
    #     r2_lista.append(r2)

    # ###PARA REGRESION LINEAL SGD CON ROBUST SCALER###


    for n in range(10):


        nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
        nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"
            #Obteniendo valores de los archivos generados de los k-folds para X
        df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
        X = df.values 
            #Obteniendo valores de y
        df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
        y = df2.values.ravel()

            #DATOS PARA HACER LA PREDICCION#
        x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
        y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

            #x_test
        df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
        x_test = df4.values
            #y test
        df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
        y_test = df3.values.ravel()

        x_SGD_robust_scaler = preprocessing.RobustScaler().fit_transform(X)
        x_SGD_robust_scaler_test = preprocessing.RobustScaler().fit_transform(x_test)
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
        regr.fit(x_SGD_robust_scaler,y)
        y_SGD_robust_scaler_prediction = regr.predict(x_SGD_robust_scaler_test)
        mse = mean_squared_error(y_test, y_SGD_robust_scaler_prediction)
        r2 = r2_score(y_test, y_SGD_robust_scaler_prediction)

        i = i+1

        mse_lista.append(mse)
        r2_lista.append(r2)

    promedio_mse = sum(mse_lista)/10
    promedio_r2 = sum(r2_lista)/10

    print("MSE")
    print(mse_lista)

    print("R2")
    print(r2_lista)

    return mse_r2(promedio_mse,promedio_r2)

def polynomial_regresion_grado2(iteraciones, eta_cero):
    i = 1
    ###Modelo de regresión polinomial grado 2 con SGD###
    # for n in range(10):
        
    #     nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
    #     nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"

    #     #Obteniendo valores de los archivos generados de los k-folds para X
    #     df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
    #     X = df.values 
    #     #Obteniendo valores de y
    #     df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
    #     y = df2.values.ravel()


    #     #DATOS PARA HACER LA PREDICCION#
    #     x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
    #     y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

    #     #x_test
    #     df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
    #     x_test = df4.values
    #     #y test
    #     df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
    #     y_test = df3.values.ravel()

    #     #Conversión de las variables de la ecuación original a polinomio de grado 2
    #     polynomial_features= PolynomialFeatures(degree=2)
    #     x_poly = polynomial_features.fit_transform(X)

    #     ###HACIENDO POLINOMICO DE GRADO TRES EL CONJUNTO DE PRUEBA###
    #     x_poly_test = polynomial_features.fit_transform(x_test)
    #     model_poly = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
    #     model_poly.fit(x_poly, y)

    #     y_poly_pred = model_poly.predict(x_poly_test)

    #     mse = mean_squared_error(y_test, y_poly_pred)
    #     r2 = r2_score(y_test, y_poly_pred)

    #     i = i+1

    #     mse_lista.append(mse)
    #     r2_lista.append(r2)

    ###MODELO DE REGRESION POLINOMIAL CON SGD STANDARD SCALING###
    # for n in range(10):
    #     nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
    #     nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"

    #     #Obteniendo valores de los archivos generados de los k-folds para X
    #     df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
    #     X = df.values 
    #     #Obteniendo valores de y
    #     df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
    #     y = df2.values.ravel()


    #     #DATOS PARA HACER LA PREDICCION#
    #     x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
    #     y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

        #x_test
        # df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
        # x_test = df4.values
        # #y test
        # df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
        # y_test = df3.values.ravel()

        # #Conversión de las variables de la ecuación original a polinomio de grado 2
        # polynomial_features= PolynomialFeatures(degree=2)
        # x_poly = polynomial_features.fit_transform(X)

        # ###HACIENDO POLINOMICO DE GRADO TRES EL CONJUNTO DE PRUEBA###
        # x_poly_test = polynomial_features.fit_transform(x_test)
        # x_polynomialSGD_standard_scaler = preprocessing.StandardScaler().fit_transform(x_poly)
        # x_polynomialSGD_standard_scaler_test = preprocessing.StandardScaler().fit_transform(x_poly_test)

        # regr = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
        # regr.fit(x_polynomialSGD_standard_scaler,y)

        # y_polynomialSGD_standard_scaler_prediction = regr.predict(x_polynomialSGD_standard_scaler_test)

        # mse = mean_squared_error(y_test, y_polynomialSGD_standard_scaler_prediction)
        # r2 = r2_score(y_test, y_polynomialSGD_standard_scaler_prediction)

        # i = i+1

        # mse_lista.append(mse)
        # r2_lista.append(r2)

    #     ###MODELO DE REGRESION POLINOMIAL CON SGD ROBUST SCALING###
    for n in range(10):

        nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
        nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"

        #Obteniendo valores de los archivos generados de los k-folds para X
        df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
        X = df.values 
        #Obteniendo valores de y
        df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
        y = df2.values.ravel()


        #DATOS PARA HACER LA PREDICCION#
        x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
        y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

        #x_test
        df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
        x_test = df4.values
        #y test
        df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
        y_test = df3.values.ravel()

        #Conversión de las variables de la ecuación original a polinomio de grado 2
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform(X)

        ###HACIENDO POLINOMICO DE GRADO TRES EL CONJUNTO DE PRUEBA###
        x_poly_test = polynomial_features.fit_transform(x_test)

        x_polynomialSGD_robust_scaler = preprocessing.RobustScaler().fit_transform(x_poly)
        x_polynomialSGD_robust_scaler_test = preprocessing.RobustScaler().fit_transform(x_poly_test)

        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
        regr.fit(x_polynomialSGD_robust_scaler,y)#ESTE ES MI MODELO ENTRENADO, LUEGO A ESTE LE DOY PREDICT CON EL Y_TEST

        y_polynomialSGD_robust_scaler_prediction = regr.predict(x_polynomialSGD_robust_scaler_test)
        mse = mean_squared_error(y_test, y_polynomialSGD_robust_scaler_prediction)#se usa la y_test
        r2 = r2_score(y_test, y_polynomialSGD_robust_scaler_prediction)#se usa la y_test para el mse y r2 test 

        i = i+1

        mse_lista.append(mse)
        r2_lista.append(r2)
        
    promedio_mse = sum(mse_lista)/10
    promedio_r2 = sum(r2_lista)/10

    print("MSE")
    print(mse_lista)

    print("R2")
    print(r2_lista)


    return mse_r2(promedio_mse,promedio_r2)

def polynomial_regresion_grado3(iteraciones, eta_cero):
    i = 1
     ### MODELO DE REGRESION POLINOMIAL GRADO 3 SGD SIN ESCALAR###
    # for n in range(10):

    #     #Datos para entrenar el modelo#
    #     nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
    #     nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"

    #     #Obteniendo valores de los archivos generados de los k-folds para X
    #     df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
    #     X = df.values 
    #     #Obteniendo valores de y
    #     df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
    #     y = df2.values.ravel()

    #     #DATOS PARA HACER LA PREDICCION#
    #     x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
    #     y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

    #     #x_test
    #     df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
    #     x_test = df4.values
    #     #y test
    #     df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
    #     y_test = df3.values.ravel()

    #     #Conversión de las variables de la ecuación original a polinomio de grado 3
    #     polynomial_features= PolynomialFeatures(degree=3)
    #     x_poly = polynomial_features.fit_transform(X)

    #     ##HACIENDO POLINOMICO DE GRADO TRES EL CONJUNTO DE PRUEBA###
    #     x_poly_test = polynomial_features.fit_transform(x_test)

    #     model_poly = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
    #     model_poly.fit(x_poly, y) #ESTE DEVUELVE UN MODELO ENTRENADO

    #     ###haciendo la predicción##
    #     y_poly_pred = model_poly.predict(x_poly_test)

    #     i = i+1

    #     mse = mean_squared_error(y_test, y_poly_pred)
    #     r2 = r2_score(y_test, y_poly_pred)

    #     mse_lista.append(mse)
    #     r2_lista.append(r2)



    ###MODELO DE REGRESION POLINOMIAL CON SGD STANDARD SCALING###
    # for n in range(10):
    #     #Datos para entrenar el modelo#
    #     nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
    #     nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"

    #     #Obteniendo valores de los archivos generados de los k-folds para X
    #     df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
    #     X = df.values 
    #     #Obteniendo valores de y
    #     df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
    #     y = df2.values.ravel()

    #     #DATOS PARA HACER LA PREDICCION#
    #     x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
    #     y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

    #     #x_test
    #     df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
    #     x_test = df4.values
    #     #y test
    #     df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
    #     y_test = df3.values.ravel()

    #     #Conversión de las variables de la ecuación original a polinomio de grado 3
    #     polynomial_features= PolynomialFeatures(degree=3)
    #     x_poly = polynomial_features.fit_transform(X)

    #     ##HACIENDO POLINOMICO DE GRADO TRES EL CONJUNTO DE PRUEBA###
    #     x_poly_test = polynomial_features.fit_transform(x_test)
    #     x_polynomialSGD_standard_scaler = preprocessing.StandardScaler().fit_transform(x_poly)
    #     x_polynomialSGD_standard_scaler_test = preprocessing.StandardScaler().fit_transform(x_poly_test)

    #     regr = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
    #     regr.fit(x_polynomialSGD_standard_scaler,y)
    #     y_polynomialSGD_standard_scaler_prediction = regr.predict(x_polynomialSGD_standard_scaler_test)
    #     mse = mean_squared_error(y_test, y_polynomialSGD_standard_scaler_prediction)
    #     r2 = r2_score(y_test, y_polynomialSGD_standard_scaler_prediction)

    #     i = i+1

    #     mse_lista.append(mse)
    #     r2_lista.append(r2)

    ###MODELO DE REGRESION POLINOMIAL CON SGD ROBUST SCALING###

    for n in range(10):


        #Datos para entrenar el modelo#
        nombre_archivo_X = "data_validation_train_10pliegues_" + str(i) + ".csv"
        nombre_archivo_y = "target_validation_train10pliegues_" + str(i) + ".csv"

        #Obteniendo valores de los archivos generados de los k-folds para X
        df = pd.read_csv(nombre_archivo_X, sep=',', engine='python')
        X = df.values 
        #Obteniendo valores de y
        df2 = pd.read_csv(nombre_archivo_y,sep=',', engine='python')
        y = df2.values.ravel()

        #DATOS PARA HACER LA PREDICCION#
        x_archivo_test = "data_validation_test_10pliegues_" + str(i) + ".csv"
        y_archivo_test = "target_validation_test_10pliegues_" + str(i) + ".csv"

        #x_test
        df4 = pd.read_csv(x_archivo_test, sep = ',', engine = 'python')
        x_test = df4.values
        #y test
        df3 = pd.read_csv(y_archivo_test,sep=',', engine='python')
        y_test = df3.values.ravel()

        #Conversión de las variables de la ecuación original a polinomio de grado 3
        polynomial_features= PolynomialFeatures(degree=3)
        x_poly = polynomial_features.fit_transform(X)

        ##HACIENDO POLINOMICO DE GRADO TRES EL CONJUNTO DE PRUEBA###
        x_poly_test = polynomial_features.fit_transform(x_test)   
        x_polynomialSGD_robust_scaler = preprocessing.RobustScaler().fit_transform(x_poly)
        x_polynomialSGD_robust_scaler_test = preprocessing.RobustScaler().fit_transform(x_poly_test)

        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta_cero, max_iter= iteraciones)
        regr.fit(x_polynomialSGD_robust_scaler,y)
        y_polynomialSGD_robust_scaler_prediction = regr.predict(x_polynomialSGD_robust_scaler_test)
        mse = mean_squared_error(y_test, y_polynomialSGD_robust_scaler_prediction)
        r2 = r2_score(y_test, y_polynomialSGD_robust_scaler_prediction)

        i = i+1

        mse_lista.append(mse)
        r2_lista.append(r2)

    
    promedio_mse = sum(mse_lista)/10
    promedio_r2 = sum(r2_lista)/10

    print("MSE")
    print(mse_lista)

    print("R2")
    print(r2_lista)


    return mse_r2(promedio_mse,promedio_r2)

def prediccion_final():
    	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
    df = pd.read_csv('cal_housing.csv', sep=',', engine='python')
    X = df.drop(['medianHouseValue'],axis=1).values #estructura X del mismo tamaño que y
    y = df['medianHouseValue'].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 0)

    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(X_train)

    x_polynomialSGD_standard_scaler = preprocessing.StandardScaler().fit_transform(x_poly)

    #ENTRENANDO EL MODELO CON X_TRAIN y y_train#
    regr = SGDRegressor(learning_rate = 'constant', eta0 = 0.000001, max_iter= 200000)
    regr.fit(x_polynomialSGD_standard_scaler,y_train)

    #AJUSTANDO LO DATOS TEST PARA HACER LA PREDICCION#
    x_poly_test = polynomial_features.fit_transform(X_test)
    x_polynomialSGD_standard_scaler_test = preprocessing.StandardScaler().fit_transform(x_poly_test)

    #HACIENDO LA PREDICCION
    y_polynomialSGD_standard_scaler_prediction = regr.predict(x_polynomialSGD_standard_scaler_test)
    mse = mean_squared_error(y_test, y_polynomialSGD_standard_scaler_prediction)
    r2 = r2_score(y_test, y_polynomialSGD_standard_scaler_prediction)

    return mse_r2(mse,r2)



suma_para_promedio_mse = 0
suma_para_promedio_r2 = 0

##PARA LOS K-FOLDS##

# iteracion = int(input("Introduce numero de iteraciones: "))
# beta = float(input("Introduce el eta0: "))

###REGRESION LINEAL SGD###
#regresion_SGD_MSE = regresion_lineal_SGD(iteracion, beta)
#print("promedio mse: ",regresion_SGD_MSE.mse)
#print("promedio r2: ",regresion_SGD_MSE.r2)

    
#### REGRESION POLINOMIAL DE GRAD0 2 ###
# regresion_polynomial2_MSE = polynomial_regresion_grado2(iteracion,beta)
# print("promedio mse: ",regresion_polynomial2_MSE.mse)
# print("promedio r2: ",regresion_polynomial2_MSE.r2)


# ### REGRESION POLINOMIAL DE GRAD0 3 ###
# regresion_polynomial3_MSE = polynomial_regresion_grado3(iteracion,beta)
# print("promedio mse: ",regresion_polynomial3_MSE.mse)
# print("promedio r2: ",regresion_polynomial3_MSE.r2)


###HACIENDO LA PREDICCION FINAL###

mse_r2_prediccion = prediccion_final()

print("El mse de la prediccion es: ", mse_r2_prediccion.mse)
print("El r2 de la prediccion es: ", mse_r2_prediccion.r2)
