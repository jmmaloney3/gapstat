# ---
# --- gapstat_test.py - tests for gapstat.py
# ---
# --- Author:  John Maloney
# --- created: 2019-02-10
# ---

import math
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering

import gapstat


def test_pooled_within_cluster_sum_of_squares():
    """Test the pooled with-in cluster sum of squares calculation.
    """
    # test using quad1 data set
    T1, labels1, n_clusters1 = _quad1(center=[0, 0], offset=[0.5, 0.5])
    _run_pooled_within_cluster_sum_of_squares_test(T1, labels1, n_clusters1)

    # test using quad4 data set
    T4, labels4, n_clusters4 = \
        _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    _run_pooled_within_cluster_sum_of_squares_test(T4, labels4, n_clusters4)

    # test using quad16 data set
    T16, labels16, n_clusters16 = _quad16(center=[0, 0], offset1=[0.5, 0.5],
                                          offset2=[2, 2], offset3=[4, 4])
    _run_pooled_within_cluster_sum_of_squares_test(T16, labels16, n_clusters16)


def test_gapstat_score():
    """Test the gap statistic calculation.
    """
    # test using quad4 data set
    T4, L4, K4 = _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    _run_gapstat_score_test(T4, L4, K4)

    # test using quad16 data set
    T16, L16, K16 = _quad16(center=[0, 0], offset1=[0.5, 0.5],
                            offset2=[2, 2], offset3=[4, 4])
    _run_gapstat_score_test(T16, L16, K16)


def test_calc_W():
    """Test the W calculation different data sets.
    """
    # test using quad1 data set
    T1, L1, K1 = _quad1(center=[0, 0], offset=[0.5, 0.5])
    _run_calc_W_test(T1, L1, K1)

    # test using quad4 data set
    T4, L4, K4 = _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    _run_calc_W_test(T4, L4, K4)

    # test using quad16 data set
    T16, L16, K16 = _quad16(center=[0, 0], offset1=[0.5, 0.5],
                            offset2=[2, 2], offset3=[4, 4])
    _run_calc_W_test(T16, L16, K16)

    # test when number of samples equals number of clusters
    T6 = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])

    K6 = T6.shape[0]
    _run_calc_W_test(T6, None, K6)

    # test when each cluster contains duplicate samples
    T2 = np.array([[1, 1], [1, 1], [1, 1],
                  [2, 2], [2, 2], [2, 2],
                  [3, 3], [3, 3], [3, 3],
                  [4, 4], [4, 4], [4, 4]])
    K2 = 4
    _run_calc_W_test(T2, None, K2)


def test_calc_exp_W():
    """Test the expected W calculation using the different datasets.
    """
    # test using quad1 data set
    T1, _, K1 = _quad1(center=[0, 0], offset=[0.5, 0.5])
    _run_calc_exp_W_test(T1, K1)

    # test using quad4 data set
    T4, _, K4 = _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    _run_calc_exp_W_test(T4, K4)

    # test using quad16 data set
    T16, _, K16 = _quad16(center=[0, 0], offset1=[0.5, 0.5],
                          offset2=[2, 2], offset3=[4, 4])
    _run_calc_exp_W_test(T16, K16)


def test_gen_null_ref():
    """Test null reference data set generation.
    """
    # generate a null reference data set
    n_samples = 10
    n_features = 5
    f_min = [-1, -0.5, 0, 0.5, 1]
    f_max = [0,   0.5, 1, 1.5, 2]

    null_ref = gapstat._gen_null_ref(n_samples, n_features, f_min, f_max)

    assert null_ref.shape[0] == n_samples
    assert null_ref.shape[1] == n_features
    assert (f_min < null_ref).all().all()
    assert (f_max > null_ref).all().all()


def test_gapstat():
    """Test the gapstat function using different datasets.
    """
    # test using quad4 data set
    T4, _, K4 = _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    _run_gapstat_test(T4, K4)

    # test using quad16 data set
    T16, _, K16 = _quad16(center=[0, 0], offset1=[0.5, 0.5],
                          offset2=[2, 2], offset3=[4, 4])
    _run_gapstat_test(T16, K16)


def test_gstat_kmeans():
    """Test the GapStatClustering using K-Mean as the base clusterer.
    """
    # test fit()
    _run_fit_test()

    # test predict() and fit_predict()
    _run_predict_test()

    # test transform and fit_transform()
    _run_transform_test()


def test_gstat_agglomerative():
    """Test the GapStatClustering using AgglomerativeClustering as the
    base clusterer.
    """
    # test fit()
    _run_fit_test(base_clusterer=AgglomerativeClustering())

    # test predict() and fit_predict()
    _run_predict_test(base_clusterer=AgglomerativeClustering())

    # test transform and fit_transform()
    _run_transform_test(base_clusterer=AgglomerativeClustering())


def test_gstat_affinitypropagation():
    """Test the GapStatClustering using AffinityPropagation as the
    base clusterer.
    """
    # affinitypropagation is incompatble with GapStatClustering
    # because it doesn't allow number of clusters to be specified
    with pytest.raises(AttributeError):
        gapstat.GapStatClustering(base_clusterer=AffinityPropagation())


def test_gstat_birch():
    """Test the GapStatClustering using Birch as the
    base clusterer.
    """
    # test fit()
    _run_fit_test(base_clusterer=Birch())

    # test predict() and fit_predict()
    _run_predict_test(base_clusterer=Birch())

    # test transform and fit_transform()
    _run_transform_test(base_clusterer=Birch())


def test_gstat_dbscan():
    """Test the GapStatClustering using DBSCAN as the
    base clusterer.
    """
    # dbscan is incompatble with GapStatClustering
    # because it doesn't allow number of clusters to be specified
    with pytest.raises(AttributeError):
        gapstat.GapStatClustering(base_clusterer=DBSCAN())


def test_gstat_featureagglomeration():
    """Test the GapStatClustering using FeatureAgglomeration as the
    base clusterer.
    """
    # FeatureAgglomeration is incompatble with GapStatClustering
    # because it transforms the data set before clusering
    # which breaks some of the gapstat logic
    # Instead, use AgglomerativeClustering and transform the
    # data set before fitting the data
    with pytest.raises(AttributeError):
        gapstat.GapStatClustering(base_clusterer=FeatureAgglomeration())


def test_gstat_minibatchkmeans():
    """Test the GapStatClustering using MiniBatchKMeans as the
    base clusterer.
    """
    # test fit()
    _run_fit_test(base_clusterer=MiniBatchKMeans())

    # test predict() and fit_predict()
    _run_predict_test(base_clusterer=MiniBatchKMeans())

    # test transform and fit_transform()
    _run_transform_test(base_clusterer=MiniBatchKMeans())


def test_gstat_meanshift():
    """Test the GapStatClustering using MeanShift as the
    base clusterer.
    """
    # dbscan is incompatble with GapStatClustering
    # because it doesn't allow number of clusters to be specified
    with pytest.raises(AttributeError):
        gapstat.GapStatClustering(base_clusterer=MeanShift())


def test_gstat_spectralclustering():
    """Test the GapStatClustering using SpectralClustering as the
    base clusterer.
    """
    # test fit()
    _run_fit_test(base_clusterer=SpectralClustering())

    # test predict() and fit_predict()
    _run_predict_test(base_clusterer=SpectralClustering())

    # test transform and fit_transform()
    _run_transform_test(base_clusterer=SpectralClustering())


def test_input_validation():
    """Test cases where the input is invalid and ensure that the
    appropriate exception is raised.
    """
    # create test data
    X, y, n_clusters = _quad4(center=[0, 0], offset1=[0.5, 0.5],
                              offset2=[2, 2])

    # corrupt data to cause check_X_y to throw exception
    # -- too few labels
    y = y[0:-1]

    # test gapstat_score()
    with pytest.raises(ValueError):
        gapstat.gapstat_score(X, y)

    # test gapstat()
    with pytest.raises(ValueError):
        gapstat.gapstat(X, max_k=X.shape[0]+1)

# --
# -- Utility Functions
# --


def _run_fit_test(base_clusterer=None):
    """Test GapStatClustering.fit() method using the specified base clusterer.
    """
    # construct test data
    T, _, K = _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    n_samples = T.shape[0]

    # create gapstat clusterer with base clusterer
    gstat = gapstat.GapStatClustering(base_clusterer=base_clusterer)
    base_clusterer = gstat.base_clusterer

    # test when max_k is too small
    gstat.set_params(max_k=K-1)
    with pytest.raises(NotFittedError):
        gstat.fit(T)

    # test fit()
    gstat.set_params(max_k=K)
    gstat = gstat.fit(T)
    _check_labels(gstat.labels_, gstat.n_clusters_, n_samples, K)


def _run_predict_test(base_clusterer=None):
    """Test GapStatClustering.predict() and GapStatClustering.fit_predict()
    methods using the specified base clusterer.
    """
    # construct test data
    T, _, K = _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    n_samples = T.shape[0]

    # create gapstat clusterer with base clusterer
    gstat = gapstat.GapStatClustering(base_clusterer=base_clusterer)
    base_clusterer = gstat.base_clusterer

    # test predict() before fit()
    # adapt test case to the base_cluster capabilities
    if (hasattr(base_clusterer, 'predict')):
        # test predict() before fit()
        with pytest.raises(NotFittedError):
            gstat.predict(T)
    else:
        # test unsupported predict()
        with pytest.raises(AttributeError):
            gstat.predict(T)

    # test when max_k is too small
    gstat.set_params(max_k=K-1)
    with pytest.raises(NotFittedError):
        gstat.fit_predict(T)

    # test fit_predict()
    gstat.set_params(max_k=K)
    predicted_labels = gstat.fit_predict(T)
    _check_labels(predicted_labels, gstat.n_clusters_, n_samples, K)
    _check_labels(gstat.labels_, gstat.n_clusters_, n_samples, K)

    # test predict()
    # adapt test case to the base_cluster capabilities
    if (hasattr(base_clusterer, 'predict')):
        predicted_labels = gstat.predict(T)
        _check_labels(predicted_labels, gstat.n_clusters_, n_samples, K)
        _check_labels(gstat.labels_, gstat.n_clusters_, n_samples, K)


def _run_transform_test(base_clusterer=None):
    """Test GapStatClustering.transform() and GapStatClustering.fit_transform()
    methods using the specified base clusterer.
    """
    # construct test data
    T, _, K = _quad4(center=[0, 0], offset1=[0.5, 0.5], offset2=[2, 2])
    n_samples = T.shape[0]

    # create gapstat clusterer with base clusterer
    gstat = gapstat.GapStatClustering(base_clusterer=base_clusterer)
    base_clusterer = gstat.base_clusterer

    # test transform() before fit()
    # adapt test case to the base_cluster capabilities
    if (hasattr(base_clusterer, 'transform')):
        # test transform() before fit()
        with pytest.raises(NotFittedError):
            gstat.transform(T)
    else:
        # test unsupported transform()
        with pytest.raises(AttributeError):
            gstat.transform(T)

    # test when max_k is too small
    # adapt test case to the base_cluster capabilities
    if (hasattr(base_clusterer, 'fit_transform')):
        # test when max_k is too small
        gstat.set_params(max_k=K-1)
        with pytest.raises(NotFittedError):
            gstat.fit_transform(T)
    else:
        # test unsupported fit_transform()
        with pytest.raises(AttributeError):
            gstat.fit_transform(T)

    # determine expected number of columns in transformed data
    # -- in most cases this is the number of clusters
    # -- for birch this is the number of subclusters
    if (isinstance(base_clusterer, Birch)):
        n_features = len(base_clusterer.subcluster_centers_)
    else:
        n_features = K

    # test fit_transform()
    # adapt test case to the base_cluster capabilities
    if (hasattr(base_clusterer, 'fit_transform')):
        gstat.set_params(max_k=K)
        transformed_T = gstat.fit_transform(T)
        _check_transformed(transformed_T, n_samples, n_features)

    # test transform()
    # adapt test case to the base_cluster capabilities
    if (hasattr(base_clusterer, 'transform')):
        transformed_T = gstat.transform(T)
        _check_transformed(transformed_T, n_samples, n_features)


def _run_pooled_within_cluster_sum_of_squares_test(T, labels, n_clusters):
    """Test the pooled with-in cluster sum of squares calculation
    using the provided data set.
    """
    # calculate the pooled within cluster sum of squares
    W = gapstat._pooled_within_cluster_sum_of_squares(T, labels, n_clusters)

    # calculate the answer
    # the result should be the "sum of squares around the cluster means"
    expected_W = _pooled_sum_sq_around_mean(T, labels, n_clusters)

    # check test results
    assert expected_W == W


def _run_gapstat_score_test(T, labels, n_clusters):
    """Test the gap statistic calculation using the provided dataset.
    """
    gap, W, log_W, log_W_star, s = \
        gapstat.gapstat_score(T, labels, calcStats=True)

    expected_W = _pooled_sum_sq_around_mean(T, labels, n_clusters)

    assert gap == (log_W_star - log_W)
    assert W == expected_W
    assert log_W_star > 0.0
    assert s > 0.0


def _run_calc_W_test(T, L, n_clusters):
    """Test the W calculation using the specified data set.
    """
    W, log_W, labels = gapstat._calc_W(T, n_clusters)

    # check W and log(W) calculation
    if (L is None):
        expected_W = _pooled_sum_sq_around_mean(T, labels, n_clusters)
    else:
        expected_W = _pooled_sum_sq_around_mean(T, L, n_clusters)

    assert W == expected_W
    if (W == 0.0):
        assert np.isnan(log_W)
    else:
        assert log_W == math.log(W)

    # check the calculated labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    assert len(le.classes_) == n_clusters


def _run_calc_exp_W_test(T, n_clusters):
    """Test the expected W calculation using the specified dataset.
    """
    log_W_star, s = gapstat._calc_exp_W(T, n_clusters)

    assert not np.isnan(log_W_star)
    assert not np.isnan(s)


def _run_gapstat_test(T, n_clusters):
    """Test the gapstat function using the specified data set.
    """
    max_k = n_clusters + 2
    est_k, labels = gapstat.gapstat(T, max_k=max_k)
    assert est_k == n_clusters

    # count the number of clusters
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    assert len(le.classes_) == n_clusters


def _check_labels(gstat_labels, gstat_k, expected_n, expected_k):
    """Check that the calculated labels and number of clusters (k) equals the
    expected values.
    """
    # check n_clusters calculated by gstat
    assert gstat_k == expected_k

    # check the number of labels
    assert gstat_labels.shape[0] == expected_n

    # check the number of unique labels
    le = LabelEncoder()
    le.fit_transform(gstat_labels)
    assert len(le.classes_) == expected_k


def _check_transformed(transformed_data, expected_n_rows, expected_n_cols):
    """Check that the transformed data has the expected shape.
    """
    assert transformed_data.shape[0] == expected_n_rows
    assert transformed_data.shape[1] == expected_n_cols


def _pooled_sum_sq_around_mean(T, labels, n_clusters):
    """Calculate the pooled sum of the squares around the cluster
    mean for all clusters.
    """
    expected_W = 0
    for c in range(n_clusters):
        expected_W = expected_W + _sum_sq_around_mean(T[labels == c])

    return expected_W


def _sum_sq_around_mean(C):
    """Calculate the sum of the squares around the cluster mean
    for the specified clusters.
    """
    # calculate cluster mean
    c_mean = np.mean(C, axis=0)
    # calculate "sum of squares around the cluster means"
    sum_sq_from_mean = 0
    for i in range(len(C)):
        sum_sq_from_mean = sum_sq_from_mean + (C[i, 0] - c_mean[0])**2
    return sum_sq_from_mean


def _quad1(center=[0, 0], offset=[1, 1], label=0):
    """Generate a data set with four data points centered on the specified
    center point.  Each point is offset from the center using the specified
    offset.
    """
    T = np.array([[center[0]+offset[0], center[1]-offset[1]],
                  [center[0]+offset[0], center[1]+offset[1]],
                  [center[0]-offset[0], center[1]+offset[1]],
                  [center[0]-offset[0], center[1]-offset[1]]])

    # create label arrray
    L = np.full(4, label)

    # return data set T, label array L and num of clusters
    return T, L, 1


def _quad4(center=[0, 0], offset1=[1, 1], offset2=[4, 4], label0=0):
    """Generate a data set with four sets of four data points for a total of
    16 data points in four groups.  The quad1 function is used to generate
    each set of data points.  The offset2 parameter is used to calcualte
    the center of each quad1 data set and offset1 specifies the offset used
    to generate the quad1 datapoints.
    """
    C1 = center + np.array([offset2[0], -offset2[1]])
    T1, L1, K1 = _quad1(C1, offset1, label0)

    C2 = center + np.array([offset2[0], offset2[1]])
    T2, L2, K2 = _quad1(C2, offset1, label0+1)

    C3 = center + np.array([-offset2[0], offset2[1]])
    T3, L3, K3 = _quad1(C3, offset1, label0+2)

    C4 = center + np.array([-offset2[0], -offset2[1]])
    T4, L4, K4 = _quad1(C4, offset1, label0+3)

    T = np.append(T1, T2, axis=0)
    L = np.append(L1, L2, axis=0)
    T = np.append(T,  T3, axis=0)
    L = np.append(L,  L3, axis=0)
    T = np.append(T,  T4, axis=0)
    L = np.append(L,  L4, axis=0)

    return T, L, K1+K2+K3+K4


def _quad16(center=[0, 0], offset1=[1, 1], offset2=[4, 4], offset3=[8, 8]):
    """Generate a data set with 16 sets of four data points for a total of
    64 data points in 16 groups.  The quad4 function is used to generate
    four sets of data at a time.  The offset3 parameter is used to calcualte
    the center of each quad4 data set and offset2 specifies the offset used
    to generate the quad4 datapoints.   The offset1 parameter is used by
    the quad4 function to generate each quad1 dataset (see the quad1
    description for more details).
    """
    C1 = center + np.array([offset3[0], -offset3[1]])
    T1, L1, K1 = _quad4(C1, offset1, offset2, 0)

    C2 = center + np.array([offset3[0], offset3[1]])
    T2, L2, K2 = _quad4(C2, offset1, offset2, 4)

    C3 = center + np.array([-offset3[0], offset3[1]])
    T3, L3, K3 = _quad4(C3, offset1, offset2, 8)

    C4 = center + np.array([-offset3[0], -offset3[1]])
    T4, L4, K4 = _quad4(C4, offset1, offset2, 12)

    T = np.append(T1, T2, axis=0)
    L = np.append(L1, L2, axis=0)
    T = np.append(T,  T3, axis=0)
    L = np.append(L,  L3, axis=0)
    T = np.append(T,  T4, axis=0)
    L = np.append(L,  L4, axis=0)

    return T, L, K1+K2+K3+K4
