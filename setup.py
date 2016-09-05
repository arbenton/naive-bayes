from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_type = Extension("naive_bayes",
                    sources=["src/naive_bayes.pyx", 
                            "src/c_gaussian_naive_bayes.c", 
                            "src/c_bernoulli_naive_bayes.c"],
                    libraries=["gsl", "gslcblas", "m"],
                    library_dirs=["/usr/lib/x86_64-linux-gnu"])

setup(name="naive_bayes",
      ext_modules = cythonize([ext_type]))
