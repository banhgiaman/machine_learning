import numpy as np
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

randIdx = np.array(range(iris_X.shape[0]))
np.random.shuffle(randIdx)

iris_X = iris_X[randIdx]
iris_y = iris_y[randIdx]

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

knn = neighbors.KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

y_predicts = knn.predict(X_test)

print('Iris accuracy with KNN: {}%'.format(int(accuracy_score(y_predicts, y_test)*100)))
