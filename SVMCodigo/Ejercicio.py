import numpy as np


i = 0
def prediccion(calculo_obtenido, c_magnitud):
    #arr_pred = np.array([])
    
    if(calculo_obtenido < c_magnitud):
        i = 0
        #np.append(arr_pred, 0)
    elif(calculo_obtenido > c_magnitud):
        #np.append(1)
        i = 1
    return i #arr_pred

def prediccion_previa(vector_test,vector_c,magnitud_c):
    p = np.array([])
    p = np.dot(vector_test,vector_c)/magnitud_c
    return p


pred_vec = np.array([])
array_test = np.array([[3,3],[2,4],[4,4]])
positivos = np.array([[5,7],[6,6],[5,5],[4,5],[4,6]])
negativos = np.array([[1,1],[2,1],[3,1],[1,2],[2,2]])

print("El array test es: ", array_test)

#Para obtener el tamaño m y poder hacer las operaciones de promedios
tupla = positivos.shape
m = tupla[0]
#Promedios positivos y negativos c
c_promedio_positivo = positivos.sum(axis=0)/m
c_promedio_negativo = negativos.sum(axis=0)/m

#print("El size del positivo: ", positivos.shape(1))
print(c_promedio_positivo)
print(c_promedio_negativo)

#Obteniendo w
w = c_promedio_positivo - c_promedio_negativo
print("El w vale: ",w)

#Calculando el c general
c = (c_promedio_positivo + c_promedio_negativo)/2
print("El c vale: ", c)

#Magnitud del c
magnitud_c = np.linalg.norm(c)
print("La magnitud de c es",magnitud_c)

for elemen_prueba in array_test:
    p_1 = prediccion_previa(elemen_prueba,c,magnitud_c)
    print("La primera prediccion previa es: ", p_1)

    print("El valor pertenece a: ",prediccion(p_1,magnitud_c))
    pred_vec = np.append(pred_vec,prediccion(p_1,magnitud_c),axis = None)
    print(pred_vec)

#primera prediccion
# p_1 = prediccion_previa(np.array([3,3]),c,magnitud_c)
# print("La primera prediccion previa es: ", p_1)

# print("El valor pertenece a: ",prediccion(p_1,magnitud_c))
# pred_vec = np.append(pred_vec,prediccion(p_1,magnitud_c),axis = None)
# print(pred_vec)

# #prediccion 2
# p_2 = prediccion_previa(np.array([2,4]),c,magnitud_c)
# print("La primera prediccion previa es: ", p_1)

# print("El valor pertenece a: ",prediccion(p_2,magnitud_c))
# pred_vec=np.append(pred_vec,prediccion(p_2,magnitud_c),axis = None)
# print(pred_vec)
