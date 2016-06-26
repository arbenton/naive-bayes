#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>

#include "naive_bayes.h"

gnb_classifier *new_gnb_classifier(double *X, int *y, int fac, int pop, int dim)
{
	double work[pop*dim]; // workspace array
	int len;
	int f, p, d;

	/* Initialize classifier and all arrays */

	gnb_classifier *clf = malloc(sizeof(gnb_classifier));
	clf->mean = malloc(dim * sizeof(double));
	clf->std = malloc(dim * sizeof(double));
	clf->class_prior = malloc(fac * sizeof(double));
	clf->class_mean = malloc(fac * dim * sizeof(double*));
	clf->class_std = malloc(fac * dim * sizeof(double*));
	clf->fac = fac;
	clf->dim = dim;

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

	return clf;
}

int gnb_classify(gnb_classifier *clf, double *X)
{
	int dim, fac;

	dim = clf->dim;
	fac = clf->fac;

	int class;
	double best;
	double evid;
	double like;
	double post[fac];
	double temp;
	int d, f;

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