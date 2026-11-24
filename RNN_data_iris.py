#Réseaux de neurones,Neural network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
#import des données
from sklearn.datasets import load_iris
iris=load_iris()

#Visualisation
X=iris.data
y=iris.target
colormap=np.array(["red",'green','blue'])
plt.scatter(X[:,3],X[:,1],c=colormap[y])

#modèle avec  trois layers de 10 neurones
modele_neural_network=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
modele_neural_network.fit(X,y)
print("le score du modèle R²=",modele_neural_network.score(X,y))