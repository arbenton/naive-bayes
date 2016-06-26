#ifndef NAIVE_BAYES_
#define NAIVE_BAYES_

typedef struct {
	double *mean;
	double *std;
	double *class_prior;
	double *class_mean;
	double *class_std;
	int fac;
	int dim;
} gnb_classifier;

gnb_classifier * new_gnb_classifier(double *X, int *y, int fac, int pop, int dim);

int gnb_classify(gnb_classifier *clf, double *X);

#endif // NAIVE_BAYES_
