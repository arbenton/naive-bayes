import numpy as np
from naive_bayes import GaussianNaiveBayesClassifier
from sklearn.naive_bayes import GaussianNB
import timeit

np.random.seed(100)
n = 100
cat_1 = np.array([np.random.normal(0, 1, n), np.random.normal(3, 1, n)]).T
cat_2 = np.array([np.random.normal(3, 1, n), np.random.normal(0, 1, n)]).T
X = np.vstack((cat_1, cat_2)).copy(order="c")
y = np.array([0 for _ in cat_1] + [1 for _ in cat_2]).copy(order="c")

print "Scikit Learn Benchmark"
start_time = timeit.default_timer()
skclf = GaussianNB()
skclf.fit(X, y)
skclf.predict(X)
del skclf
print "Time Elapsed: ", timeit.default_timer() - start_time
print "Custom Benchmark"
start_time = timeit.default_timer()
cuclf = GaussianNaiveBayesClassifier(2, 2)
cuclf.train(X, y)
cuclf.classify(X)
del cuclf
print "Time Elapsed: ", timeit.default_timer() - start_time
