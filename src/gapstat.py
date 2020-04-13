"""A scikit learn compatible wrapper for clustering a data set without
specifying the number of clusters to generate.  Instead, the gap
statistic method is used to estimate the optimal number of clusters for
the data set."""

# Author: John Maloney
# License: BSD-3-Clause

import math
import numpy as np
from random import randint
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import column_or_1d
from sklearn.utils.validation import check_is_fitted


class GapStatClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    """A clusterer that uses the gap statistic to estimate the optimal
    number of clusters.

    For details on the gap statistic method for estimating the optimal
    number of clusters see [1]_.

    Parameters
    ----------

    base_clusterer : object or None, optional (default=None)
        The base clusterer to use to cluster the data.
        If None, then the base clusterer is K-Means.

    max_k : int, optional, default: 10
        The maximum number of clusters to consider when estimating the
        optimal number of clusters for the data set.

    B1 : int, optional, default: 10
        The number of null reference data sets that are generated and
        clustered in order to estimate the optimal number of clusters
        for the data set.

    B2 : int, optional, default: 1
        The number of times the input data set is clustered in order to
        estimate the average pooled with-in cluster sum of squares.  This
        can be used to improve the stability of the results.

    Attributes
    ----------

    n_clusters_ : int
        The estimate of the optimal number of clusters identified using
        the gap statistic method.

    labels_ :
        Labels of each point

    Examples
    --------
    >>> from gapstat import GapStatClustering
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> from sklearn.datasets import make_blobs

    >>> X,_ = make_blobs(n_samples=16, centers=[[4,4],[-4,4],[-4,-4],[4,-4]],
    ...       n_features=2, random_state=2)
    >>>
    >>> gstat_km = GapStatClustering(max_k=5).fit(X)
    >>> gstat_km.n_clusters_
    4
    >>> gstat_km.labels_
    array([0, 0, 3, 1, 2, 0, 3, 2, 2, 1, 3, 0, 1, 2, 1, 3])
    >>> gstat_km.predict([[-3, -3], [3, 3]])
    array([4, 3], dtype=int32)
    >>>
    >>> gstat_ac = GapStatClustering(base_clusterer=AgglomerativeClustering(),
    ...                              max_k=5).fit(X)
    >>> gstat_ac.n_clusters_
    4
    >>> gstat_ac.labels_
    array([3, 3, 2, 0, 1, 3, 2, 1, 1, 0, 2, 3, 0, 1, 0, 2])

    References
    ----------

    .. [1] Tibshirani, R. , Walther, G. and Hastie, T. (2001), Estimating the
           number of clusters in a data set via the gap statistic. Journal of
           the Royal Statistical Society: Series B (Statistical Methodology),
           63: 411-423. doi:10.1111/1467-9868.00293
    """
    def __init__(self,
                 base_clusterer=None,
                 max_k=10,
                 B1=10,
                 B2=1):
        # create default base clusterer if necessary
        self.base_clusterer = _check_clusterer(base_clusterer)
        self.max_k = max_k
        self.B1 = B1
        self.B2 = B2

    def fit(self, X, y=None):
        """Compute the clustering.  The gap statistic method is used to estimate
        the optimal number of clusters.

        TO DO: allow optional fit parameters to be passed to the base clusterer

        Parameters
        ----------
        X : array-like, sparse matrix or dataframe,shape=[n_samples,n_features]
            The observations to cluster.

        y : Ignored
            not used, present here for API consistency by convention.

        Raises
        ------
        NotFittedError
            If the data set contains more clusters than k_max.
        """

        n_clusters, labels = \
            gapstat(X, clusterer=self.base_clusterer,
                    max_k=self.max_k, B1=self.B1, B2=self.B2)

        if ((n_clusters is None) | (labels is None)):
            msg = "The estimated optimal number of clusters is greater than " \
                    "max_k=%d"
            raise NotFittedError(msg % self.max_k)
        else:
            self.n_clusters_, self.labels_ = (n_clusters, labels)

        return self

    def fit_predict(self, X, y=None):
        """Compute the clustering and return the cluster label for each
        observation.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        TO DO: allow optional fit parameters to be passed to the base clusterer

        Parameters
        ----------
        X : array-like, sparse matrix or dataframe,shape=[n_samples,n_features]
            The observations to cluster.

        y : Ignored
            not used, present here for API consistency by convention.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.

        Raises
        ------
        NotFittedError
            If the data set contains more clusters than k_max.
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute the clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        If the base clusterer does not implement the trnsform() method then
        X is return untransformed.

        TO DO: allow optional fit parameters to be passed to the base clusterer

        Parameters
        ----------
        X : array-like, sparse matrix or dataframe,shape=[n_samples,n_features]
            New data to transform.

        y : Ignored
            not used, present here for API consistency by convention.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.

        Raises
        ------
        NotFittedError
            If the data set contains more clusters than k_max.

        AttributeError
            If the base_clusterer does not implement transform().
        """

        # make sure the base cluster implements transform()
        # -- raises AttributeError if it doesn't
        getattr(self.base_clusterer, 'transform')

        # fit the data and then call transform
        return self.fit(X).transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.
        If the base clusterer does not implement the trnsform() method then
        X is return untransformed.

        Parameters
        ----------
        X : array-like, sparse matrix or dataframe,shape=[n_samples,n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted to a data set.

        AttributeError
            If the base_clusterer does not implement transform().
        """
        check_is_fitted(self)

        # call transform on the base clusterer
        return self.base_clusterer.transform(X)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, sparse matrix or dataframe,shape=[n_samples,n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted to a data set.

        AttributeError
            If the base_clusterer does not implement predict().
        """
        check_is_fitted(self)

        # call predict on the base clusterer
        return self.base_clusterer.predict(X)


def gapstat(X, clusterer=None, max_k=10, B1=10, B2=1, calcStats=False):
    """Gap statistic clustering algorithm.  Uses the gap statistic method
    to estimate the optimal number of clusters and uses that estimate
    to generate a clustering.

    TO DO: Provide a way to pass additionl parameters to the base clusterer.

    Parameters
    ----------

    X : array-like, sparse matrix or dataframe, shape (n_samples, n_features)
        The observations to cluster.

    clusterer : object or None, optional (default=None)
        The base clusterer to use to cluster the data.
        If None, then the base clusterer is K-Means.

    max_k : int, optional, default: 10
        The maximum number of clusters to consider when estimating the
        optimal number of clusters for the data set.

    B1 : int, optional, default: 10
        The number of null reference data sets that are generated and
        clustered in order to estimate the optimal number of clusters
        for the data set.

    B2 : int, optional, default: 1
        The numbe of times the input data set is clustered in order to
        estimate the average pooled with-in cluster sum of squares.  This
        can be used to improve the stability of the results.

    calcStats : boolean, optional, default: False
        Calculate and return the statistics for all values of k from
        1 through max_k.  The statistics include W, log(W), log(W*),
        gap and standard error.  Otherwise, stop when the estimated optimal
        k is determined and only return n_clusters and labels.

    Returns
    -------

    n_clusters : int
        The estimate of the optimal number of clusters identified using the
        gap statistic method.

    labels : int array, shape = [n_samples]
        The labels identifying the cluster that each sample belongs to.
        label[i] is the index of the cluster for the i-th observation.  The
        clustering includes n_clusters clusters.

    stats : dict, optional
        When calcStats is true, the statistics are returned in a dictionary
        with three entries: data, index and columns.  The data entry is a numpy
        two-dimensional array that includes the statistics described below.
        The index and columns entries provide additional information that can
        be used to create a pandas dataframe containing the statistics.  Each
        row of the data matrix provides the following statistics for each value
        of k considered:

        W : The mean pooled within-cluter sum of squares around the cluster
            means for the input data set.  The value returned for each value of
            k is the mean of B2 clusterings.
        log(W) : the logarithm of W (see above)
        log(W*) : The expectation of log(W) under an appropriate null reference
                  distribution of the data.  This is calculated as the mean log
                  pooled within-cluter sum of squares around the cluster means
                  for B2 generated null reference data sets.
        Gap : The gap statistic calculated as log(W*) - log(W).
        Std Err : The standard error of log(W*).

    Examples
    --------
    >>> from gapstat import gapstat
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> X,_ = make_blobs(n_samples=16, centers=[[4,4],[-4,4],[-4,-4],[4,-4]],
    ...       n_features=2, random_state=2)
    >>>
    >>> k, labels = gapstat(X, clusterer=AgglomerativeClustering(),
    ...                     max_k=5)
    >>> k
    4
    >>> labels
    array([3, 3, 2, 0, 1, 3, 2, 1, 1, 0, 2, 3, 0, 1, 0, 2])
    """
    # validate input parameters

    if max_k <= 0:  # TO DO: also check if it is an integer
        raise ValueError("Maximum number of clusters to consider should be "
                         "a positive integer, got %d instead" % max_k)

    if B1 <= 0:  # TO DO: also check if it is an integer
        raise ValueError("The number of null reference data sets to generate "
                         "should be a positive integer, got %d instead" % B1)

    if B2 <= 0:  # TO DO: also check if it is an integer
        raise ValueError("The number of times to cluster the data set to find "
                         "a stable W value should be a positive integer, got "
                         "%d instead" % B2)

    # check the clusterer and create a default clusterer if necessary
    clusterer = _check_clusterer(clusterer)

    # to determine whether a particular value of k is optimal
    # requires calculating the gap statistic for k+1, so
    # interate through all values of k up to max_k+1

    # check that the number of samples is consistent with (max_k+1)
    X, _, _ = _check_inputs(X=X, k=max_k+1)

    # create arrays to hold statistics
    # -- "pooled within-cluster sum of squares around cluster means"
    W = np.zeros(max_k+1)
    log_W = np.empty(max_k+1)
    log_W[:] = np.nan
    # -- "expected W_k under a null reference distribution of the data"
    log_W_star = np.empty(max_k+1)
    log_W_star[:] = np.nan
    # -- the gap statistic
    gap = np.empty(max_k+1)
    gap[:] = np.nan
    # -- standard error
    s = np.empty(max_k+1)
    s[:] = np.nan
    # -- labels for each value of k
    labels = np.full((max_k+1, X.shape[0]), -1)  # labels for each b
    # -- the estimated optimal number of clusters
    k_hat = None  # if max_k is too small then k_hat will be None

    for k in range(max_k+1):

        # calculate W and log(W)
        # -- k is zero-basd iterator, num clusters is one greater
        W[k], log_W[k], labels[k, :] = _calc_W(X, k+1,
                                               clusterer=clusterer, B=B2)

        # calculate log(W*) and the standard error
        # -- k is zero-basd iterator, num clusters is one greater
        log_W_star[k], s[k] = _calc_exp_W(X, k+1, clusterer=clusterer, B=B1)

        # calculate the gap statistic for k
        gap[k] = log_W_star[k] - log_W[k]
        # if W for ref data is less than W for input matrix
        # then set gap to zero and see if adding more clusters
        # reduces the value of W for the input matrix
        if (gap[k] < 0):
            gap[k] = 0

        # determine whether the previous value of k is the estimated optimal
        # number of clusters
        # -- (1) make sure the optimal has not been found
        # -- (2) make sure there is a previous value (k-1) for comparison
        # -- (3) make sure clustering of X is actually better than the
        # --     clustering of null ref data
        # -- (4) use gap statistic to determine if optimal k has been found
        if ((k_hat is None) &                    # (1)
                (k > 0) &                        # (2)
                (gap[k-1] != 0) &                # (3)
                (gap[k-1] >= (gap[k] - s[k]))):  # (4)

            # found an estimate of the optimal number of clusters!
            # -- # k is zero-based iteration index, num of clusters is +1
            k_hat = k  # previous value of k is the estimate: ((k-1)+1) = k

            # if we are not calculating statistics then stop
            if (not calcStats):
                break

    # -- end for k

    # fit the clusterer using the estimated optimal k &
    # identify labels for optimal k
    if (k_hat is not None):
        # fit the clusterer using k_hat as the number of clusters
        clusterer.set_params(n_clusters=k_hat)
        k_hat_labels = clusterer.fit_predict(X)
    else:
        k_hat_labels = None

    # return the results
    if (calcStats):
        stats = {}
        # create array of k values (index)
        stats["index"] = np.arange(1,max_k+2)
        # create an array of column headers (columns)
        stats["columns"] = np.array(["W", "log(W)", "log(W*)", "Gap", "Std Err"])
        # create a multi-dimensional array with the statistics (data)
        stats["data"] = np.stack((W, log_W, log_W_star, gap, s), axis=1)

        return k_hat, k_hat_labels, stats
    else:
        return k_hat, k_hat_labels

# end function


def gapstat_score(X, labels, k=None, clusterer=None, B=10, calcStats=False):
    """Compute the gap statistic score (metric) for the given clustering.

    The gap statistic is the difference between the log of the pooled
    within-cluster sum of squares for the candiate clustering and the
    expectation of that value under an apprpriate null reference
    distribution.

    For more details on the gap statistic see [1]_.

    Parameters
    ----------
    X : array [n_samples_a, n_features]
        The observations that were clustered.
    labels : array, shape = [n_samples]
        Predicted labels for each observation.
    k : int, optional, default: None
        The number of clusters in the clustering.  If set to None then the
        number of clusters will be calculated based on the supplied labels.
    clusterer : object or None, optional (default=None)
        The clusterer to use to cluster the null referece data sets.
        If None, then the base clusterer is K-Means.
    B : int, optional, default: 10
        The number of null reference data sets that are generated and
        clustered in order to estimate the optimal number of clusters
        for the data set.
    calcStats : boolean, optional, default: False
        Calculate and return the underlying statistics used to calculate
        the gap statistic score.  The statistics include W, log(W), log(W*),
        and standard error.  Otherwise, only the gap statistic score is
        returned.

    Returns
    -------
    gap : float
        The value of the gap statistic for the clustering.

    W : float, optional
        The mean pooled within-cluter sum of squares around the cluster means
        for the provided clustering.  This is only returned when calcStats is
        True.

    log_W : float, optional
        log(W).  This is only returned when calcStats is True.

    log_W_star : float, optional
        The expectation of log(W) under an appropriate null reference
        distribution of the data.  This is calculated as the mean log pooled
        within-cluter sum of squares around the cluster means for B generated
        null reference data sets.  This is only returned when calcStats is
        True.

    s : float, optional
        The standard error of log(W*).  This is only returned when calcStats
        is True.

    Examples
    --------
    >>> from gapstat import gapstat
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> X,_ = make_blobs(n_samples=16, centers=[[4,4],[-4,4],[-4,-4],[4,-4]],
    ...       n_features=2, random_state=2)
    >>>
    >>> ac = AgglomerativeClustering().fit(X)
    >>> gapstat_score(X, ac.labels_)
    -0.6028585939536981

    References
    ----------
    .. [1] Tibshirani, R. , Walther, G. and Hastie, T. (2001), Estimating the
           number of clusters in a data set via the gap statistic. Journal of
           the Royal Statistical Society: Series B (Statistical Methodology),
           63: 411-423. doi:10.1111/1467-9868.00293
    """
    if B <= 0:  # TO DO: also check if it is an integer
        raise ValueError("The number of null reference data sets to generate "
                         "should be a positive integer, got %d instead" % B)

    # check that the inputs are valid and consistent
    X, labels, k = _check_inputs(X=X, y=labels, k=k)

    # check the clusterer and create a default clusterer if necessary
    clusterer = _check_clusterer(clusterer)

    # calculate W for supplied clustering
    W = _pooled_within_cluster_sum_of_squares(X, labels, k)
    log_W = _safeLog(W)

    # calculate log(W*) and standard error
    log_W_star, s = _calc_exp_W(X, k, clusterer, B)

    # calculate the gap statistic for the clustering
    gap = log_W_star - log_W

    if (calcStats):
        return gap, W, log_W, log_W_star, s
    else:
        return gap


def _calc_W(X, k, clusterer=None, B=1):
    """Calculate the expected pooled within-in cluster sum of squares
    for the data set and the specified number of clusters k.

    Parameters
    ----------
    X : array [n_samples_a, n_features]
        The observations that were clustered.
    k : int
        The number of clusters to use when clustering the data sets.
    clusterer : object or None, optional (default=None)
        The clusterer to use to cluster the data set.  If None, then
        the clusterer is K-Means.
    B : int, optional, default: 10
        The number of times the data set should be clustered in order to
        determine an average pooled within cluster sum of squares.  This
        helps smooth out random differences introduced by random starting
        states for some clusterers.

    Returns
    -------
    W : float
        The mean pooled with-in cluster sum of squares for the B
        clusterings that were generated.
    log_W : float
        The mean log(W) for the B clusterings that were generated
    labels : array [n_samples]
        The mode of the labels generated for each of the B clusterings
        that were generated
    """

    # check the clusterer and create a default clusterer if necessary
    clusterer = _check_clusterer(clusterer)

    # handle degenerate case when there is 1 sample per cluster
    if (k == X.shape[0]):
        # return:
        # -- W: one sample per cluster, so W is zero
        # -- log(W): log(0) is undefined, return NaN
        # -- labels: return unique label for each sample
        return 0.0, np.nan, np.array(range(k))

    # cluster the data set B times and calculate the average W

    # arrays to hold stats for the B iterations
    W = np.zeros(B)
    log_W = np.empty(B)
    log_W[:] = np.nan
    labels = np.full((B, X.shape[0]), -1)  # labels for each b

    # set the number for clusters in the clustered data set
    clusterer.set_params(n_clusters=k)

    for b in range(B):
        # generate clusters
        labels[b, :] = clusterer.fit_predict(X)

        # calculate W and log(W) for the b-th iteration
        W[b] = _pooled_within_cluster_sum_of_squares(X, labels[b, :], k)
        log_W[b] = _safeLog(W[b])

    # -- end for b

    # find the mean of W and log(W) for the B clusterings
    avg_W = np.sum(W)/B
    avg_log_W = np.sum(log_W)/B

    # randomly select one of the clusterings to return
    i = randint(0, B-1)
    ret_labels = labels[i, :]

    return avg_W, avg_log_W, ret_labels


def _calc_exp_W(X, k, clusterer=None, B=10):
    """Calculate the expected pooled within-in cluster sum of squares
    for the null reference disribution for the data set and the
    specified number of clusters k.

    Parameters
    ----------
    X : array [n_samples, n_features]
        The observations that were clustered.
    k : int
        The number of clusters to use when clustering the null ref
        data sets.
    clusterer : object or None, optional (default=None)
        The clusterer to use to cluster the null referece data sets.
        If None, then the clusterer is K-Means.
    B : int, optional, default: 10
        The number of null reference data sets that are generated and
        clustered in order to calculate the null reference pooled within
        cluster sum of squares.

    Returns
    -------
    log_W_star : float
        The expected pooled within-in cluster sum of squares for the null
        reference disribution.

    std_err : float
        The standard error for the means of the B null reference data sets
    """

    # check the clusterer and create a default clusterer if necessary
    clusterer = _check_clusterer(clusterer)

    n_samples, n_features = X.shape

    # handle degenerate case when there is 1 sample per cluster
    if (k == n_samples):
        # return:
        # -- log(W*): W* is 0, log(0) is undefined, return NaN
        # -- standard error: return NaN
        return np.nan, np.nan

    # calculate min & max for samples
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # generate B null ref sets, custer each set and calcualte the statistic

    # arrays to hold stats for the B null ref data sets for current k
    null_ref_W = np.zeros(B)  # value of W* for the B null ref sets
    log_null_ref_W = np.empty(B)  # value of log(W*) for the B null ref sets
    log_null_ref_W[:] = np.nan

    # set the number for clusters in the clustered data set
    clusterer.set_params(n_clusters=k)

    for b in range(B):
        # generate a new "null reference data set"
        null_ref = _gen_null_ref(n_samples, n_features, X_min, X_max)

        # generate clusters for the "null reference data set"
        labels = clusterer.fit_predict(null_ref)

        # calculate W* and log(W*) for the b-th null reference data set
        null_ref_W[b] = _pooled_within_cluster_sum_of_squares(null_ref,
                                                              labels, k)
        log_null_ref_W[b] = _safeLog(null_ref_W[b])

    # -- end for b

    # end generation and clustering of B null ref data sets

    # find the mean of log(W*) for the B ref data set samples
    log_W_star = np.sum(log_null_ref_W)/B  # log(W*) (aka, l_bar)

    # calculate the standard deviation
    sd = math.sqrt(np.mean(np.power(log_null_ref_W - log_W_star, 2)))

    # calculate the standard error
    s = sd*math.sqrt(1 + 1/B)

    return log_W_star, s


def _gen_null_ref(n_samples, n_features, f_min, f_max):
    """Generate a data set with the specified number of samples and
    values for features chosen from a uniform distributions with the
    specified minimum and maximum values.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.

    n_features : int
        The number of features to generate.

    f_min : arrray, float [n_features]
        The minimum values for the features.

    f_max : array, float [n_features]
        The maximum value for the features.

    Returns
    -------
    null_ref : array, float [ n_sample, n_features ]
        The generated samples.
    """

    # create 2D array to hold null reference data set
    null_ref = np.empty((n_samples, n_features))
    null_ref[:] = np.nan

    # generate a "null reference data set"
    for f in range(n_features):
        null_ref[:, f] = np.random.uniform(low=f_min[f], high=f_max[f],
                                           size=n_samples)
    # null ref set generated ---

    return null_ref


def _check_clusterer(clusterer=None):
    """Check that the clusterer is a valid clusterer (it implements
    the required methods).  If no cluster is provided, create
    a default clusterer.

    Parameters
    ----------
    clusterer : object or None, optional (default=None)
        The clusterer to use to cluster the data sets.
        If None, then the clusterer is K-Means.

    Returns
    -------
    clusterer : object
        The supplied clusterer or a default clusterer if none was provided.
    """

    if (clusterer is None):  # create default clusterer if necessary
        # default Cluster is KMeans
        clusterer = KMeans()
    else:
        # make sure base clusterer implements set_params()
        getattr(clusterer, 'set_params')

        # make sure base clusterer implements fit_predict()
        getattr(clusterer, 'fit_predict')

        # make sure base clusterer has n_clusters attribute
        getattr(clusterer, 'n_clusters')

    return clusterer


def _check_inputs(X=None, y=None, k=None):
    """Input validation for gapstat.

    Depending on the inputs provided, use one or more of the following
    validation utilities to validate the inputs:

        sklearn.utils.check_array()
        sklearn.utils.check_X_y()
        sklearn.utils.validation.column_or_1d()

    In addition, if k is provided, validate the following:

        0 < k <= X.shape[0]
        k == number of unique value in y

    Parameters
    ----------
    X : array [n_samples, n_features], optional
        The data set to be clustered
    y : array-like, shape=[n_samples], optional
        The labels identifying the cluster that each sample belongs to.
    k : int
        The number of clusters.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.
    k_validated: int
        The calculated or validated k.
    """
    if (X is None) & (y is None):
        raise ValueError("One of 'X' or 'y' must not be 'None'")

    n_labels = None
    if (y is not None):
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_labels = len(le.classes_)

    if (X is not None) & (y is not None):
        X, y = check_X_y(X, y)
        if (k is not None):
            if (not 0 < k <= X.shape[0]):
                raise ValueError("Number of clusters (k) is %d. Valid values "
                                 "are 1 to n_samples (inclusive)" % k)
            if (n_labels != k):
                raise ValueError("Number of unique labels (%d) does not equal "
                                 "the number of clusters (k=%d)."
                                 % (n_labels, k))
        else:  # (k is None)
            k = n_labels

    if (X is not None) & (y is None):
        X = check_array(X)
        if (k is not None) & (not 0 < k <= X.shape[0]):
            raise ValueError("Number of clusters (k) is %d. Valid values "
                             "are 1 to n_samples=%d (inclusive)"
                             % (k, X.shape[0]))

    if (X is None) & (y is not None):
        y = column_or_1d(y)
        if (k is not None) & (n_labels != k):
            raise ValueError("Number of unique labels (%d) does not equal the "
                             "number of clusters (k=%d)." % (n_labels, k))
        else:
            k = n_labels

    return X, y, k


def _pooled_within_cluster_sum_of_squares(X, labels, k):
    """Calculate the pooled within-cluster sum of squares (W) for
    the clustering defined by the specified labels.

    Parameters
    ----------
    X : array-like, sparse matrix or dataframe, shape=[n_samples, n_features]
        The observations that were clustered.
    labels : array-like, shape=[n_samples]
        The labels identifying the cluster that each sample belongs to.
    k: integer
        The number of unique labels - number of clusters.
    """
    n_samples, _ = X.shape

    # initialize W to zero
    W = 0

    # -- iterate over the clusters and calculate the pairwise distances for
    # -- the points
    for c_label in range(k):
        c_k = X[labels == c_label]
        d_k = euclidean_distances(c_k, c_k, squared=True)
        n_k = len(c_k)
        # multiply by 0.5 because each distance is included in the sum twice
        W = W + (0.5*d_k.sum())/(2*n_k)

    # return the result
    return W
# end pooled_within_cluster_sum_of_squares


def _safeLog(x):
    """Return the log of the specified number.  If the number
    is zero (or close to zero) return NaN.

    Parameters
    ==========
    x : float

    Returns
    =======
    log_x : float
        The value of log(x) or np.nan if x is zero.
    """
    if (math.isclose(x, 0.0)):
        # return a very small number
        return np.nan
    else:
        return math.log(x)


def _get_column_indices(X):
    """Get the column indices for the input matrix.  Determines the data
    type of the input matrix and uses the appropriate method to retrieve
    the column indices.
    """

    try:
        # this will fail if X is not a pandas DataFrame
        return list(X.columns)
    except AttributeError:
        pass

    # X is an array-like
    return list(range(X.shape[1]))
# -- end function
