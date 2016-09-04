cdef extern from "../src/c_naive_bayes.h":

    ctypedef struct gnb_classifier
    gnb_classifier *new_gnb_classifier(long int fac, long int dim)
    long int gnb_train(gnb_classifier *clf, double *X, long int *y, long int pop)
    long int gnb_classify(gnb_classifier *clf, double *X)
