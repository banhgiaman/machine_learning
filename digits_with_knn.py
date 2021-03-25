from sklearn import datasets, neighbors
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

digits = datasets.load_digits()

X_digits = digits.data
y_digits = digits.target

''' show some samples '''
# fig = plt.figure()
# plt.gray()

# for i in range(1, 17):
#     fig.add_subplot(4, 4, i)
#     plt.imshow(X_digits[i].reshape(8,8))

# plt.show()

randIdx = np.array(range(X_digits.shape[0]))
np.random.shuffle(randIdx)

X_digits = X_digits[randIdx]
y_digits = y_digits[randIdx]

knn = neighbors.KNeighborsClassifier(n_neighbors=9)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=365)

knn.fit(X_train, y_train)
y_predicts = knn.predict(X_test)
print('MNIST accuracy with KNN: {}% '.format(int(accuracy_score(y_predicts, y_test)*100)))
