cimport naive_bayes
import numpy as np
cimport numpy as np

cdef class Naive_Bayes_Classifier:

    cdef naive_bayes.gnb_classifier* clf

    def __cinit__ (self, int fac, int dim):

        self.clf = naive_bayes.new_gnb_classifier(fac, dim)

    def train (self,
        np.ndarray[double, ndim=2, mode="c"] X,
        np.ndarray[long, ndim=1, mode="c"] y):

        cdef int pop = len(y)

        status = naive_bayes.gnb_train(self.clf, &X[0, 0], &y[0], pop)

        if status:
            raise RuntimeError("GNB Training Failed")

    def classify (self,
        np.ndarray[double, ndim=1, mode="c"] X):

        result = naive_bayes.gnb_classify(self.clf, &X[0])

        return result
