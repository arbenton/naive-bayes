cdef extern from "c_gaussian_naive_bayes.h":

    ctypedef struct gnb_classifier
    gnb_classifier *new_gnb_classifier(long int fac, long int dim)
    int free_gnb_classifier(gnb_classifier *clf)
    int gnb_train(gnb_classifier *clf, double *X, long int *y, long int pop)
    int gnb_classify(gnb_classifier *clf, double *X, long int *y, int N)

cdef extern from "c_bernoulli_naive_bayes.h":

    ctypedef struct bnb_classifier
    bnb_classifier *new_bnb_classifier(long int fac, long int dim)
    int free_bnb_classifier(bnb_classifier *clf)
    int bnb_train(bnb_classifier *clf, long int *X, long int *y, long int pop)
    int bnb_classify(bnb_classifier *clf, long int *X, long int *y, int N)
