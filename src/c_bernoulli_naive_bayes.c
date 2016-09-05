#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>

#include "c_bernoulli_naive_bayes.h"

struct _bnb_classifier {
    double *mean;
    double *class_prior;
    double *class_mean;
    long int fac;
    long int dim;
};

bnb_classifier *new_bnb_classifier(long int fac, long int dim)
{
	/* Initialize classifier and all arrays */

	bnb_classifier *clf = malloc(sizeof(bnb_classifier));
	clf->mean = malloc(dim * sizeof(double));
	clf->class_prior = malloc(fac * sizeof(double));
	clf->class_mean = malloc(fac * dim * sizeof(double*));
	clf->fac = fac;
	clf->dim = dim;

	return clf;
}

int free_bnb_classifier(bnb_classifier *clf)
{
    free(clf->mean);
    free(clf->class_prior);
    free(clf->class_mean);
    free(clf);

    return 0;
}

int bnb_train(bnb_classifier *clf, long int *X, long int *y, long int pop)
{
    long int dim = clf->dim;
    long int fac = clf->fac;

    double work[pop*dim]; // workspace array
	int len;
	int f, p, d;

    for (d=0; d<dim*pop; d++) {
        work[d] = (double) X[d];
    }

    /* mean and standard deviation */

	for (d=0; d<dim; d++) {
		clf->mean[d] = gsl_stats_mean(work + d, dim, pop); //!
	}

	/* class-wise mean and standard deviation */

	for (f=0; f<fac; f++) {

		/* Find observations in each class */

		len = 0;
		for (p=0; p<pop; p++) {
			if (y[p] == f) {
				for (d=p*dim; d<(p+1)*dim; d++, len++) {
					work[len] = X[d];
				}
			}
		}
		len /= dim;

		/* statistics for the found observations */

		for (d=0; d<dim; d++) {
			clf->class_prior[f] = len / ((double) pop);
			clf->class_mean[f*dim + d] = gsl_stats_mean(work + d, dim, len); //!
		}
	}
    return 0;
}

inline int bnb_classify_one(bnb_classifier *clf, long int *X)
{
	long int dim, fac;

	dim = clf->dim;
	fac = clf->fac;

	long int class;
	double best;
	double evid;
	double like;
	double post[fac];
	int d, f;

	/* Evidence */

	evid = 0.;
	for (d=0; d<dim; d++) {
		evid += log(gsl_ran_bernoulli_pdf(X[d], clf->mean[d]));
	}

	/* Select best class */

	best = 0.0;
	class = 0;

	for (f=0; f<fac; f++) {

		/* Calculate Likelhood */

		like = 0.;
		for (d=f*dim; d<(f+1)*dim; d++) {
			like += log(gsl_ran_bernoulli_pdf(X[d%dim], clf->class_mean[d]));
		}
		/* Posteriors and selection */

		post[f] = exp(log(clf->class_prior[f]) + like - evid);
		if (best < post[f]) {
			best = post[f];
			class = f;
		}
	}

	return class;
}

int *bnb_classify(bnb_classifier *clf, long int *X, long int *y, long int N)
{
    int n;
    long int dim = clf->dim;

    for (n=0; n<N; n++) {
        y[n] = bnb_classify_one(clf, X+n*dim);
    }

    return 0;
}
