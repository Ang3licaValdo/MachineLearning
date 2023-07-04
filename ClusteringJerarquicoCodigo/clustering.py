import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

# Carga del conjunto de datos
dataset = pd.read_csv('Country-data.csv')
X = dataset.iloc[:, [3, 5]].values
countries = dataset['country'].values

# print(X)
# print(sch.linkage(X, method = 'ward'))
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendograma')
plt.xlabel('Países')
plt.ylabel('Distancias Euclidianas')
plt.show()


hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

print(y_hc)

# # Visualising the clusters
# #es una condicion, al final, se rellena de True y False por cada uno de los componentes del vector y_hc
# #y si es True entonces, pesca el valor de la columna columna 0 con el indice del vector y_hc en el que 
# #nos encontremos evaluando, lo mismo hace con el siguiente X, al final si los dos son TRUE, pues se tiene
# #un valor de X y uno de y y así es como se grafican las instancias 
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1') #de la X es una condicion booleana y luego la columna, ya que ese X contiene los valores de dos columnas
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')


# for i in range(len(y_hc)):
#     plt.annotate(str(countries[i]), (X[i,0],X[i,1]))

plt.title('Países que necesitan ayuda económica')
plt.xlabel('Salud')
plt.ylabel('Ingresos')
plt.legend()
plt.show()

n = 1
paises_ricos = []
paises_pobres = []

for i in range(len(y_hc)):
    if(y_hc[i] == 1):
        print(str(n)+". " + str(countries[i]))
        paises_pobres.append(str(countries[i]))
        n = n+1
    else:
        paises_ricos.append(str(countries[i]))

        

print("Paises que no necesitan el dinero: ")

z = 1
for i in range(len(paises_ricos)):
    print(str(z) + ". "+paises_ricos[i])
    z = z+1

# for i in range(len(paises_pobres)):
#     if(i == 67 or i > 67 and i < 100):
#         paises_ricos.append(' ')

# columnas_csv = {'Países más necesitados': paises_pobres, 'Países más estables': paises_ricos}
# df2 = pd.DataFrame(columnas_csv)
# df2.to_csv('paises.csv')
