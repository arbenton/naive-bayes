#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "naive_bayes.h"

int main ()
{
    const gsl_rng_type *T;
    gsl_rng *r;
    int dim, pop, fac;
    int d, p;
	fac = 2;
    dim = 2;
    pop = 100;
    double X[pop*dim];
    int y[pop];
	gnb_classifier *clf;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    for (p=0; p<pop; p++) {
		for (d=p*dim; d<(p+1)*dim; d++) {
        	X[d] = gsl_ran_gaussian(r, 1) + 3*((p+d)%2);
		}
        y[p] = p%2;
    }

	clf = new_gnb_classifier(X, y, fac, pop, dim);

	double obs[] = {3.0, 0.0};

	printf("%d \n", gnb_classify(clf, obs));

    gsl_rng_free (r);

    return 0;
}