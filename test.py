import numpy as np

from naive_bayes import Naive_Bayes_Classifier

np.random.seed(100)
n = 100
cat_1 = np.array([np.random.normal(0, 1, n), np.random.normal(3, 1, n)]).T
cat_2 = np.array([np.random.normal(3, 1, n), np.random.normal(0, 1, n)]).T

X = np.vstack((cat_1, cat_2)).copy(order="c")
y = np.array([0 for _ in cat_1] + [1 for _ in cat_2]).copy(order="c")

clf = Naive_Bayes_Classifier(2, 2)
clf.train(X, y)
print clf.classify(np.array([3.0, 3.0]))
