#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>

#include "c_naive_bayes.h"

struct _gnb_classifier {
	double *mean;
	double *std;
	double *class_prior;
	double *class_mean;
	double *class_std;
	long int fac;
	long int dim;
};

gnb_classifier *new_gnb_classifier(long int fac, long int dim)
{
	/* Initialize classifier and all arrays */

	gnb_classifier *clf = malloc(sizeof(gnb_classifier));
	clf->mean = malloc(dim * sizeof(double));
	clf->std = malloc(dim * sizeof(double));
	clf->class_prior = malloc(fac * sizeof(double));
	clf->class_mean = malloc(fac * dim * sizeof(double*));
	clf->class_std = malloc(fac * dim * sizeof(double*));
	clf->fac = fac;
	clf->dim = dim;

	return clf;
}

int free_gnb_classifier(gnb_classifier *clf)
{
    free(clf->mean);
    free(clf->std);
    free(clf->class_prior);
    free(clf->class_mean);
    free(clf->class_std);
    free(clf);

    return 0;
}

long int gnb_train(gnb_classifier *clf, double *X, long int *y, long int pop)
{
    long int dim = clf->dim;
    long int fac = clf->fac;

    double work[pop*dim]; // workspace array
	long int len;
	long int f, p, d;

    /* mean and standard deviation */

	for (d=0; d<dim; d++) {
		clf->mean[d] = gsl_stats_mean(X + d, dim, pop); //!
		clf->std[d] = gsl_stats_sd_m(X + d, dim, pop, clf->mean[d]);
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
			clf->class_std[f*dim + d] = gsl_stats_sd_m(work + d, dim, len, clf->class_mean[f*dim + d]);
		}
	}
    return 0;
}

long int gnb_classify(gnb_classifier *clf, double *X)
{
	long int dim, fac;

	dim = clf->dim;
	fac = clf->fac;

	long int class;
	double best;
	double evid;
	double like;
	double post[fac];
	double temp;
	long int d, f;

	/* Evidence */

	evid = 1.;
	for (d=0; d<dim; d++) {
		temp = (X[d] - clf->mean[d]) / clf->std[d];
		evid *= gsl_ran_gaussian_pdf(temp, 1.0);
	}

	/* Select best class */

	best = 0.0;
	class = 0;

	for (f=0; f<fac; f++) {

		/* Calculate Likelhood */

		like = 1.;
		for (d=f*dim; d<(f+1)*dim; d++) {
			temp = (X[d%dim] - clf->class_mean[d]) / clf->class_std[d];
			like *=  gsl_ran_gaussian_pdf(temp, 1.0);
		}

		/* Posteriors and selection */

		post[f] = clf->class_prior[f] * like / evid;
		if (best < post[f]) {
			best = post[f];
			class = f;
		}
	}

	return class;
}
