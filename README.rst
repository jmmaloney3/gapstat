.. -*- mode: rst -*-

gap-stat
========
:code:`gap-stat` is a Python module that provides a
`scikit-learn <http://scikit-learn.org>`_ compatible implementation of the
the gap statistic [1]_ to estimate the optimal number of clusters
contained in a data set.  The module provides the following:

:code:`GapStatClustering`
    a `sklearn.cluster <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster>`_
    compatible wrapper that uses a base clustering estimator to both estimate
    the optimal number of clusters and generate a clustering with that optimal
    number of clusters
:code:`gapstat`
    a function that implements the gap statistic algorithm and, if desired,
    returns the various statistics that were used to estimate the optimal
    number of clusters
:code:`gapstat_score`
    a `sklearn.metrics.cluster <https://scikit-learn.org/stable/modules/classes.html#clustering-metrics>`_
    compatible function that uses the gap statistic algorithm to provide an
    evaluation metric for cluster analysis

Installation
------------
TBD

Dependencies
~~~~~~~~~~~~
:code:`gap-stat` requires:

- Python (tested with version 3.7.2)
- scikit-learn (tested with version 0.20.1)
- numpy (tested with version 1.15.4)

Examples
--------

References
----------

.. [1] Tibshirani, R. , Walther, G. and Hastie, T. (2001), Estimating the
        number of clusters in a data set via the gap statistic. Journal of
        the Royal Statistical Society: Series B (Statistical Methodology),
        63: 411-423. doi:10.1111/1467-9868.00293
