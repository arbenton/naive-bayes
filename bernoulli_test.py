import numpy as np
from naive_bayes import BernoulliNaiveBayesClassifier
from sklearn.naive_bayes import BernoulliNB
import timeit

np.random.seed(100)
n = 100
cat_1 = np.array([np.random.binomial(1, .1, n), np.random.binomial(1, .9, n)]).T
cat_2 = np.array([np.random.binomial(1, .9, n), np.random.binomial(1, .1, n)]).T
X = np.vstack((cat_1, cat_2)).copy(order="c")
y = np.array([0 for _ in cat_1] + [1 for _ in cat_2]).copy(order="c")

print "Scikit Learn Benchmark"
start_time = timeit.default_timer()
skclf = BernoulliNB()
skclf.fit(X, y)
skclf.predict(X)
del skclf
print "Time Elapsed: ", timeit.default_timer() - start_time
print "Custom Benchmark"
start_time = timeit.default_timer()
cuclf = BernoulliNaiveBayesClassifier(2, 2)
cuclf.train(X, y)
cuclf.classify(X)
del cuclf
print "Time Elapsed: ", timeit.default_timer() - start_time
