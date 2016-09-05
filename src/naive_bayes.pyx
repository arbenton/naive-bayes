cimport naive_bayes
import numpy as np
cimport numpy as np

cdef class GaussianNaiveBayesClassifier:

    cdef naive_bayes.gnb_classifier* clf

    def __cinit__ (self, n_factors, n_dimensions):

        if isinstance(n_factors, int):
            pass
        else:
            raise TypeError("n_factors must be of type int.")
        if isinstance(n_dimensions, int):
            pass
        else:
            raise TypeError("n_dimensions must be of type int.")

        self.clf = naive_bayes.new_gnb_classifier(n_factors, n_dimensions)

        if self.clf == NULL:
            raise MemoryError("Naive_Bayes_Classifier could not be allocated.")

    def _t(self,
            np.ndarray[double, ndim=2, mode="c"] X,
            np.ndarray[long, ndim=1, mode="c"] y,
            int pop):

        return naive_bayes.gnb_train(self.clf, &X[0, 0], &y[0], pop)

    def _c (self,
            np.ndarray[double, ndim=2, mode="c"] X,
            np.ndarray[long, ndim=1, mode="c"] y,
            N):

        return naive_bayes.gnb_classify(self.clf, &X[0, 0], &y[0], N)

    def train (self, X, y):

        if isinstance(X, np.ndarray) and X.ndim==2:
            pass
        else:
            raise TypeError()
        if isinstance(y, np.ndarray) and y.ndim==1:
            pass
        else:
            raise TypeError()

        cdef int pop = len(y)

        status = self._t(X, y, pop)

        if status:
            raise RuntimeError("GNB Training Failed")

    def classify (self, X):

        cdef int N

        if isinstance(X, np.ndarray) and X.ndim==2:
            N = len(X)
        elif isinstance(X, np.ndarray) and X.ndim==1:
            X = X[:, ]
            N = 1
        else:
            raise TypeError()

        y = np.zeros(N, dtype=int)
        status = self._c(X, y, N)

        return y

    def __del__ (self):

        status = naive_bayes.free_gnb_classifier(self.clf)

        if status:
            raise MemoryError("Deallocation failed")


cdef class BernoulliNaiveBayesClassifier:

    cdef naive_bayes.bnb_classifier* clf

    def __cinit__ (self, n_factors, n_dimensions):

        if isinstance(n_factors, int):
            pass
        else:
            raise TypeError("n_factors must be of type int.")
        if isinstance(n_dimensions, int):
            pass
        else:
            raise TypeError("n_dimensions must be of type int.")

        self.clf = naive_bayes.new_bnb_classifier(n_factors, n_dimensions)

        if self.clf == NULL:
            raise MemoryError("Naive_Bayes_Classifier could not be allocated.")

    def _t(self,
            np.ndarray[long, ndim=2, mode="c"] X,
            np.ndarray[long, ndim=1, mode="c"] y,
            int pop):

        return naive_bayes.bnb_train(self.clf, &X[0, 0], &y[0], pop)

    def _c (self,
            np.ndarray[long, ndim=2, mode="c"] X,
            np.ndarray[long, ndim=1, mode="c"] y,
            N):

        return naive_bayes.bnb_classify(self.clf, &X[0, 0], &y[0], N)

    def train (self, X, y):

        if isinstance(X, np.ndarray) and X.ndim==2:
            pass
        else:
            raise TypeError()
        if isinstance(y, np.ndarray) and y.ndim==1:
            pass
        else:
            raise TypeError()

        cdef int pop = len(y)

        status = self._t(X, y, pop)

        if status:
            raise RuntimeError("bnb Training Failed")

    def classify (self, X):

        cdef int N

        if isinstance(X, np.ndarray) and X.ndim==2:
            N = len(X)
        elif isinstance(X, np.ndarray) and X.ndim==1:
            X = X[:, ]
            N = 1
        else:
            raise TypeError()

        y = np.zeros(N, dtype=long)
        status = self._c(X, y, N)

        return y

    def __del__ (self):

        status = naive_bayes.free_bnb_classifier(self.clf)

        if status:
            raise MemoryError("Deallocation failed")
