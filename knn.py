"""k-NN"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)

training_accuracy=[]
test_accuracy=[]
data_neighbors=range(1,11)
for n_neighbor in data_neighbors:
    clf=KNeighborsClassifier(n_neighbors=n_neighbor)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(data_neighbors,training_accuracy,label="score  du modèle du les valeurs entrainées en fonction de ")
plt.plot(data_neighbors,test_accuracy,label="le score du modèle en fonction de k")
plt.legend()
plt.show()