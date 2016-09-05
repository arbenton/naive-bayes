#ifndef C_BERNOULLI_NAIVE_BAYES_
#define C_BERNOULLI_NAIVE_BAYES_

#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>

typedef struct _bnb_classifier bnb_classifier;

bnb_classifier * new_bnb_classifier(long int fac, long int dim);

int free_bnb_classifier(bnb_classifier *clf);

int bnb_train(bnb_classifier *clf, long int *X, long int *y, long int pop);

int *bnb_classify(bnb_classifier *clf, long int *X, long int *y, long int N);

#endif // C_BERNOULLI_NAIVE_BAYES
