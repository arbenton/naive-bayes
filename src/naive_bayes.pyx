cimport naive_bayes
import numpy as np
cimport numpy as np


cdef class Naive_Bayes_Classifier:

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

    cdef int _t(self,
            np.ndarray[double, ndim=2, mode="c"] X,
            np.ndarray[long, ndim=1, mode="c"] y,
            int pop):

        return naive_bayes.gnb_train(self.clf, &X[0, 0], &y[0], pop)

    cdef int _c (self,
            np.ndarray[double, ndim=1, mode="c"] X):

        return naive_bayes.gnb_classify(self.clf, &X[0])

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

        if isinstance(X, np.ndarray) and X.ndim==1:
            pass
        else:
            raise TypeError()

        result = self._c(X)

        return result
