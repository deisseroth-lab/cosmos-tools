import cosmos.traces.clustering_utils as cu
import numpy as np
from cosmos.traces.cosmos_traces import CosmosTraces
import os
import pdb

def test_order_sources_by_clust_1():
    """
    Test that the ordering of sources both
    based on cluster and super-cluster yields
    the same groupings of sources.
    """
    # Generate example clustering [ncells], super_clustering [ncells],
    # and clust_ordering [nclust].

    ncells = 1000
    nclusts = 10
    nsuperclusts = 5
    clustering = np.tile(np.arange(nclusts), (int(ncells/nclusts)))
    super_clustering = np.tile(np.arange(nsuperclusts), (int(nclusts/nsuperclusts)))
    clust_ordering = np.array([0, 5, 6, 1, 7, 2, 3, 8, 4, 9])
    ordered_clustering, ordered_super_clustering = \
                        cu.order_sources_by_clust(clustering,
                                                  super_clustering,
                                                  clust_ordering)

    clust_inds = cu.get_cluster_index_ranges(ordered_super_clustering)

    # Test that ordered_clustering and ordered_super_clustering
    # include the same cells in each super_cluster.
    for super_clust_ind in np.unique(ordered_super_clustering):
        i0 = clust_inds[super_clust_ind]
        i1 = clust_inds[super_clust_ind + 1]
        incl_super_clust = np.unique(
            np.argsort(ordered_super_clustering)[i0:i1])
        incl_clust = np.unique(np.argsort(ordered_clustering)[i0:i1])

        assert(np.max(np.abs(incl_super_clust - incl_clust)) == 0)


def test_compare_cluster_memberships():
    """
    TODO
    :return:
    """
    pass

def test_compare_in_vs_out_of_cluster_comparison():
    """
    TODO
    :return:
    """
    pass

def test_order_super_clusters_1():
    """
    TODO
    """
    # ncells = 1000
    # nclusts = 10
    # nsuperclusts = 5
    # clustering = np.tile(np.arange(nclusts), (int(ncells / nclusts)))
    # super_clustering = np.tile(np.arange(nsuperclusts),
    #                            (int(nclusts / nsuperclusts)))
    # clust_means = np.eye(ncells)
    # cu.order_super_clusters(super_clustering,
    #                         clust_means,
    #                         method='peak',
    #                         do_plot=True,
    #                         title_str=None)
    assert(True)


def test_order_clusters_1():
    """
    TODO

    """

    # cu.order_clusters(clust_means, do_plot=False,
    #                vertical_lines=None,
    #                titlestr=None)

    assert(True)