import cosmos.traces.trace_analysis_utils as utils
import cosmos.imaging.atlas_registration as reg
from cosmos.traces.cell_plotter import CellPlotter
import numpy as np
import scipy.stats
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
from sklearn.decomposition import PCA, NMF
import pickle
import warnings
from scipy.stats import zscore
import time
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.multitest import multipletests

def load_clustering_results(dataset_id, sets, clustering_dir, protocol_as_key=False):
    """
    Load struct containing results of clustering from cluster_analysis.ipynb.

    :param dataset_id: int.
    :param sets: list of dicts, each entry contains:
                 'method', 'protocol', 'randseed', 'n_components', 'l1'
    :param clustering_dir: string. directory path of saved out clustering results.
    :param protocol_as_key: bool. If false then just index into `all_results` using
                            an integer id. If true then use the protocol of the set.
                            Assumes that each set uses a different 'protocol'.
    :return: all_results: a dict containing the clustering results for each set in sets.
    """
    print('Loading: ')
    all_results = dict()
    for i in range(len(sets)):
        s = sets[i]
        filename = os.path.join(clustering_dir, '{}_{}_expt{}_r{}_n{}_l{}.pkl'.format(
                                                                                   s['method'],
                                                                                   s['protocol'],
                                                                                   dataset_id,
                                                                                   s['randseed'],
                                                                                   s['n_components'],
                                                                                   s['l1']))
        with open(filename, 'rb') as f:
            clust_results = pickle.load(f)
        if protocol_as_key:
            all_results[s['protocol']] = clust_results
        else:
            all_results[i] = clust_results

        print(filename)

    return all_results

def get_trial_sets(BD, use_all_trials=True):
    # Setup some analysis specific variables.
    if BD is not None:

        # Define trial sets.
        if use_all_trials:
            clean_trials = np.ones(BD.success.shape)
            min_block_trial = 0
            success = np.ones(BD.success.shape)
        else:
            clean_trials = np.zeros(BD.success.shape)
            clean_trials[BD.get_clean_trials(min_selectivity=0.95)] = 1
            min_block_trial = 7
            success = BD.success

        lick_spout1 = np.logical_and.reduce((BD.go_trials.astype('bool'),
                                             success,
                                             BD.spout_positions == 1,
                                             BD.ind_within_block >= min_block_trial,
                                             clean_trials))
        lick_spout3 = np.logical_and.reduce((BD.go_trials.astype('bool'),
                                             success,
                                             BD.spout_positions == 3,
                                             BD.ind_within_block >= min_block_trial,
                                             clean_trials))
        lick_spout4 = np.logical_and.reduce((BD.go_trials.astype('bool'),
                                             success,
                                             BD.spout_positions == 4,
                                             BD.ind_within_block >= min_block_trial,
                                             clean_trials))
        nolick = np.logical_and.reduce((~BD.go_trials.astype('bool'),
                                        success))

        trial_sets = (lick_spout1, lick_spout3, lick_spout4, nolick)
        trial_names = ('go1', 'go2', 'go3', 'ng')
    else:
        trial_sets = None
        trial_names = None

    return trial_sets, trial_names

def plot_ordered_correlation_matrix(rates,
                                    ordering,
                                    do_zscore=True,
                                    clim=[-0.2, 0.2],
                                    ylabel='Source #',
                                    xlabel='Source #'):
    """
    Plot correlation matrix, ordered by clustering results.

    :param rates: [ncells, ntime, ntrials]
    :param ordering: [ncells], the order of the rows/columns of the correlation matrix.
    :param do_zscore:
    :param clim:
    :return:
    """

    rates_flat = np.reshape(rates,
                            (rates.shape[0], rates.shape[1] * rates.shape[2]),
                            order='F')
    if do_zscore:
        rates_flat = zscore(rates_flat, axis=1)

    C = np.corrcoef(rates_flat)
    plt.figure()
    plt.imshow(C[:, ordering][ordering, :], clim=[-0.2, 0.2],
               cmap=plt.cm.bwr)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)

def get_pre_lick_sources(summed_licks, spike_rates,
                         task_classes,
                         trial_sets, which_trials,
                         which_classes, which_sets,
                         earliest_frame):
    """
    Plot all sources that have peak activity between 'earliest_frame'
    and lick onset (averaged within each trial type).
    :param summed_licks: [ntrials, nframes] Licks summed across all spouts.
    :param spike_rates: [nsources, nframes, ntrials]
    :param task_classes: the task-class assignment of each source.
    :param trial_sets: (ntrial_types), bool vector for each trial types
    :param which_trials:
    :param which_classes:
    :param which_sets:
    :param earliest_frame:
    :return:
    """
    do_debug = False

    source_ids = []
    corresponding_task_classes = []
    total_pre_lick = 0
    total_class_sources = 0
    for ind, c in enumerate(which_classes):
        sources = np.where(task_classes == c)[0]
        set_trials = np.logical_and(trial_sets[which_sets[ind]],
                                    which_trials)
        class_rates = spike_rates[sources, :, :][:, :, set_trials]
        class_licks = summed_licks[set_trials, :]

        mr = np.mean(class_rates, axis=2)
        peak_times = np.argmax(mr, axis=1)

        # Get lick onsets
        onsets = []
        for trial in range(class_licks.shape[0]):
            licks = np.where(class_licks[trial, :] > 0)[0]
            if len(licks) > 0:
                onsets.append(np.min(licks))
            else:
                onsets.append(np.nan)

        mc = np.mean(class_licks, axis=0)
        lick_onset = np.min(np.where(mc > 0)[0])


        frames_buffer = 2  # Add a buffer before the first lick to deal with the lick before the mouse actually reaches the spout
        pre_lick_sources = np.logical_and(peak_times > earliest_frame,
                                          peak_times < lick_onset - frames_buffer)

        source_ids.extend(list(sources[np.where(pre_lick_sources)[0]]))
        num_pre_lick = np.sum(pre_lick_sources)
        corresponding_task_classes.extend([c]*num_pre_lick)

        total_pre_lick += num_pre_lick
        total_class_sources += len(sources)

        if do_debug:
            print(np.sort(peak_times))
            plt.figure(), plt.plot(mc), plt.xlim([0, 205])
            print('Lick onset: {}'.format(lick_onset))
            print(sources[np.where(pre_lick_sources)[0]])
            print(source_ids)

    return source_ids, corresponding_task_classes, total_class_sources



def get_sources_more_active_pre_vs_post(traces, trial_sets,
                                        t_ranges, trials_to_use,
                                        pthresh=1e-3):
    """
    Get indices of sources that are more active during pre-lick
    vs. post-lick on go trials (but not necesarilly on nogo trials).

    :param traces: [nsources x nframes x ntrials]
    :param trial_sets: list of boolean arrays, indicating which
                       trials are included in each trial type.
                       Right now,
                       assumes the first 3 are go_trials and the
                       4th is nogo.
    :param t_ranges: dict containing 'pre' and 'post' frame ranges
                     (i.e. np.arange(15, 72) and np.arange(76, 133)
    :param trials_to_use: [ntrials] boolean array. If only want to
                         use a subset of the trials (i.e. the ones
                         where the mouse only licked after a certain
                         frame).
    :param pthresh: float. pre-bonferonni-corrected pvalue.
    :return:
        sorted_sig_cells: ID of significant cells, sorted by pvalue
    """


    tt = dict()
    tt['pre'] = traces[:, t_ranges['pre'], :]
    tt['post'] = traces[:, t_ranges['post'], :]

    time_means = dict()
    for key in tt.keys():
        time_means[key] = np.mean(tt[key], axis=1)

    trial_means = dict()
    for key in tt.keys():
        trial_means[key] = np.mean(tt[key], axis=2)


    trials_p = []
    mds = [] # Mean difference between pre/post
    for ind, trial_set in enumerate(trial_sets):
        if trials_to_use is None:
            which_trials = trial_set
        else:
            which_trials = np.logical_and(trial_set, trials_to_use)
        d = time_means['pre'] - time_means['post']
        md = np.mean(d, axis=1)
        _, p = scipy.stats.ttest_1samp(d[:, which_trials],
                                       np.zeros(d.shape[0]),
                                       axis=1)
        if ind < 3:
            p[md < 0] = 1  # One-sided test for go trials

        trials_p.append(p)
        mds.append(md)

    all_trials_p = np.vstack(trials_p)
    thresh = pthresh / all_trials_p.shape[1]
    all_sig = all_trials_p < thresh
    how_many_sig = np.sum(all_sig, axis=0)

    # sig = np.max(all_sig, axis=0)
    # sig_cells = np.where(sig)[0]
    sig_cells = np.where(how_many_sig > 0)[0] ## Select a source if any trial types are significant.
    sorted_sig_cells = sig_cells[np.argsort(p[sig_cells])]

    return sorted_sig_cells


def get_sources_decrease_on_go_and_flat_on_nogo(which_cells,
                                               traces,
                                               go_trials,
                                               nogo_trials,
                                               ranges,
                                               use_std_thresh=True,
                                               do_plot=True):
    """
    Get sources whose activity decreases on go_trials but
    is flat (or increases) on nogo_trials, when comparing the
    specified pre and post ranges.
    :param which_cells: list of cell ides.
    :param traces: [nsources x nframes x ntrials]
    :param go_trials: bool array.
    :param nogo_trials: bool array.
    :param ranges: dict containing 'pre' and 'post' frame ranges
                     (i.e. np.arange(15, 72) and np.arange(76, 133)
    :param use_std_thresh: bool. If True, then only includes
                           sources where the decrease magnitude
                           is greater than 2 stdev.
    :param do_plot: bool.
    :return: good_cells: list of IDs of sources that meet
                         the criteria.
    """


    good_cells = []
    for i in which_cells:
        go_trace = np.mean(traces[i, :, go_trials], axis=0).T
        nogo_trace = np.mean(traces[i, :, nogo_trials], axis=0).T

        means = dict()
        means['go'] = [np.mean(go_trace[ranges['pre']]),
                       np.mean(go_trace[ranges['post']])]
        means['nogo'] = [np.mean(nogo_trace[ranges['pre']]),
                         np.mean(nogo_trace[ranges['post']])]

        stds = dict()
        stds['go'] = [np.std(go_trace[ranges['pre']]),
                      np.std(go_trace[ranges['post']])]
        stds['nogo'] = [np.std(nogo_trace[ranges['pre']]),
                        np.std(nogo_trace[ranges['post']])]
        # print(stds)

        if use_std_thresh:
            is_good_cell = np.logical_and.reduce(
                (means['go'][0] - means['go'][1] > 2 * stds['go'][0],
                 means['nogo'][0] < means['nogo'][1] + stds['nogo'][0]))
        else:
            is_good_cell = np.logical_and.reduce((means['go'][0]
                                                  > means['go'][1],
                                                  means['nogo'][0]
                                                  < means['nogo'][1]))

        if is_good_cell:
            good_cells.append(i)
            if do_plot:
                plt.figure()
                plt.plot(go_trace, 'r')
                plt.plot(nogo_trace, 'b')

                plt.title('{} --- Go: {:.3f}, {:.3f}, nogo: {:.3f}, {:.3f}'.format(
                    is_good_cell,
                    means['go'][0],
                    means['go'][1],
                    means['nogo'][0],
                    means['nogo'][1]))

    return good_cells


def plot_centroids(which_cells, CT, max_radius=3):
    """
    Plot centroids of specified cells
    and gray out all other sources.
    :param which_cells:
    :param CT:
    :return:
    """
    source_coloring = np.ones((CT.ncells, 4))
    RGBA = np.copy(np.squeeze(source_coloring))
    RGBA[:, :] = np.array([0.9, 0.9, 0.9, 0.3])

    RGBA[which_cells, :] = np.array([1, 0, 0, 1])

    plt.figure()
    utils.centroids_on_atlas(RGBA, np.arange(RGBA.shape[0]),
                             CT.centroids,
                             CT.atlas_tform,
                             max_radius=max_radius,
                             rotate90=True)

def plot_pre_lick_sources(lick_rates, spike_rates,
                          task_classes, trial_sets,
                          which_classes, which_sets,
                          earliest_frame, latest_frame,
                          fps, save_dir=None):
    """
    Plot all sources that have peak activity between 'earliest_frame'
    and 'latest_frame'. Only use trials where licking ensues after
    'latest_frame'.
    :param task_classes:
    :param trial_sets:
    :param which_trials:
    :param which_classes:
    :param which_sets:
    :param earliest_frame:
    :param earliest_frame:
    :param fps:
    :param save_dir:
    :return:
    """

    summed_licks = sum([lick_rates[i] for i in range(4)])
    pre_lick_trials = np.sum(summed_licks[:, :latest_frame], axis=1) > 0
    no_pre_lick_trials = ~pre_lick_trials

    ### Get sources which exhibit pre-lick firing
    (source_ids,
     corresponding_classes,
     total_class_sources) = get_pre_lick_sources(summed_licks, spike_rates,
                                                 task_classes,
                                                 trial_sets,
                                                 no_pre_lick_trials,
                                                 which_classes, which_sets,
                                                 earliest_frame)

    total_pre_lick = len(source_ids)
    print('{} -- Fraction pre lick: {}/{} --> {:.4f}'.format(save_dir,
                                                             total_pre_lick,
                                                             total_class_sources,
                                                             total_pre_lick / total_class_sources))

    ### Plot each source for all trial types


    color_template = get_color_template()

    for s, cc in zip(source_ids, corresponding_classes):
        plt.figure(figsize=(5, 7))
        ax_traces = plt.subplot(4, 1, 4)
        for ind, c in enumerate(which_classes):
            set_trials = np.logical_and(
                trial_sets[which_sets[ind]],
                no_pre_lick_trials)

            class_rates = spike_rates[s, :, set_trials]
            class_licks = summed_licks[set_trials, :]

            # Get lick onsets
            onsets = []
            for trial in range(class_licks.shape[0]):
                licks = np.where(class_licks[trial, :] > 0)[0]
                if len(licks) > 0:
                    onsets.append(np.min(licks))
                else:
                    onsets.append(np.nan)

            plt.subplot(4, 1, ind+1)
            plt.imshow(
                gaussian_filter1d(class_rates, 1.5, axis=1,
                                         mode='constant'),
                cmap='gray_r',
                aspect='auto',
                extent=[0, class_rates.shape[1]/fps, 0, class_rates.shape[0]])
            plt.plot(onsets/fps, np.arange(len(onsets))+0.5, 'r.', markersize=1)
            plt.axvline(earliest_frame/fps, color='k', linestyle='--')
            plt.ylabel('Trial')
            plt.xlim([0, 6])
            plt.xticks(np.arange(6), '')

            mr = np.mean(class_rates, axis=0)
            plt.sca(ax_traces)
            plt.plot(np.arange(len(mr))/fps, mr, color=color_template[c])

        plt.sca(ax_traces)
        ax2 = plt.gca().twinx()
        ml = np.mean(summed_licks, axis=0)
        ax2.plot(np.arange(len(ml))/fps, ml, color='k')
        plt.xlim([0, 6])
        plt.xlabel('Time (s)')
        plt.axvline(earliest_frame/fps, color='k', linestyle='--')
        plt.suptitle('Source {}, class {}'.format(s, cc))


        if save_dir is not None:
            savename = 'source_{}.pdf'.format(s)
            plt.gcf().set_size_inches(w=2.5,
                                      h=3.5)  # Control size of figure in inches
            plt.savefig(os.path.join(save_dir, savename),
                        transparent=True, rasterized=True, dpi=600)


            # source_ids = np.hstack(source_ids)
        # source_coloring = np.ones(
        #     (len(all_nmf['mr2']['ordered_super_clustering']), 4))
        # RGBA = np.copy(np.squeeze(source_coloring))
        # RGBA[:, :] = np.array([0.9, 0.9, 0.9, 0.3])
        # RGBA[source_ids, :] = np.array([1, 0, 0, 1])
        #
        # plt.figure()
        # utils.centroids_on_atlas(RGBA, np.arange(RGBA.shape[0]),
        #                          all_nmf['mr2']['centroid_atlas_coords'],
        #                          None,
        #                          max_radius=10,
        #                          rotate90=True)


def get_task_clusters_per_region(all_dataset_names, allCT, sets,
                                 which_datasets, clustering_dir):
    """
    Load task-class clusters for the datasets specified in which_datasets,
    and for each task-class, for each dataset, determine how many
    sources are in each region.

    :param all_dataset_names: all datasets loaded in the notebook.
                              corresponding to allCT. i.e. [7, 11, 18, 19]
    :param allCT: all datasets loaded in the notebook.
    :param sets: dict specifying the parameters of the clustering.
                 i.e. {'method':'classify', 'protocol':'mr2', 'randseed':'', 'n_components':'', 'l1':''}
    :param which_datasets: which of all loaded datasets to load.
    :param clustering_dir: where clustering files are saved out.
    :return: all_clusters_region_dist: dict. keys: dataset_id, vals: dict.
                                                  keys: cluster_id. vals: dict.
                                                       keys: region_name, val: # sources
             all_total_cells_per_region: dict. keys: dataset_id, vals: dict.
                                                    keys: region_name, val: # sources
    """
    all_clusters_region_dist = {}
    all_total_cells_per_region = {}

    for dataset_id in which_datasets:
        CT_ind_spatial = int(
            np.where(np.array(all_dataset_names) == dataset_id)[0])
        all_nmf_spatial = load_clustering_results(dataset_id, sets,
                                                  clustering_dir,
                                                  protocol_as_key=True)

        region_names = allCT[CT_ind_spatial].regions
        regions_of_cells = np.array(allCT[CT_ind_spatial].region_of_cell)
        clust_assignments = all_nmf_spatial['mr2']['ordered_super_clustering'] ### Should probably make this depend on 'sets'.

        (clusters_region_dist,
         total_cells_per_region) = get_task_cluster_cells_per_region(
            region_names, regions_of_cells, clust_assignments)
        all_clusters_region_dist[dataset_id] = clusters_region_dist
        all_total_cells_per_region[dataset_id] = total_cells_per_region

    return (all_clusters_region_dist, all_total_cells_per_region)


def load_centroids_and_task_labels(all_dataset_names, allCT, sets,
                                   which_datasets, clustering_dir):
    """
    Load task-class clusters for the datasets specified in which_datasets,
    and also the centroids associated with each source.
    Scales the coordinates to be in mm.

    :param all_dataset_names: all datasets loaded in the notebook.
                              corresponding to allCT. i.e. [7, 11, 18, 19]
    :param allCT: all datasets loaded in the notebook.
    :param sets: dict specifying the parameters of the clustering.
                 i.e. {'method':'classify', 'protocol':'mr2', 'randseed':'', 'n_components':'', 'l1':''}
    :param which_datasets: which of all loaded datasets to load.
    :param clustering_dir: where clustering files are saved out.
    :return: all_clusters_region_dist: dict. keys: dataset_id, vals: dict.
                                                  keys: cluster_id. vals: dict.
                                                       keys: region_name, val: # sources
             all_total_cells_per_region: dict. keys: dataset_id, vals: dict.
                                                    keys: region_name, val: # sources
    """
    mm_per_pixel = 0.01375  ### 11um pixels with 40/50 demagnification.
                            ### Verified with USAF calibration target (20180612).

    all_centroids = {}
    all_labels = {}

    for dataset_id in which_datasets:
        CT_ind_spatial = int(
            np.where(np.array(all_dataset_names) == dataset_id)[0])
        all_nmf_spatial = load_clustering_results(dataset_id, sets,
                                                  clustering_dir,
                                                  protocol_as_key=True)

        clust_assignments = all_nmf_spatial['mr2']['ordered_super_clustering'] ### Should probably make this depend on 'sets'.
        centroids = allCT[CT_ind_spatial].centroids*mm_per_pixel
        print('Converting pixels to mm.')

        all_centroids[dataset_id] = centroids
        all_labels[dataset_id] = clust_assignments

    return (all_centroids, all_labels)


def organize_clusters_per_region(which_dsets,
                                 all_clusters_region_dist,
                                 all_total_cells_per_region,
                                 regions=['MO', 'PTLp', 'RSP', 'SSp', 'VIS']):
    """
    Convert the output of get_task_clusters_per_region()
    into arrays
        all_spatial_dist: [ndatasets x nclusters x nregions] and
        all_total_cells: [ndatasets x 1 x nregions]

    :param all_clusters_region_dist:
    :param all_total_cells_per_region:
    :param regions: List of regions (i.e. ['MO', 'PTLp', 'RSP', 'SSp', 'VIS'])
    :return:
    """
    ndatasets = len(all_clusters_region_dist.keys())
    nclusters = len(all_clusters_region_dist[
                        list(all_clusters_region_dist.keys())[0]].keys())
    nregions = len(regions)
    all_spatial_dist = np.zeros((ndatasets, nclusters, nregions))
    all_total_cells = np.zeros((ndatasets, nclusters, nregions))

    clusts = np.arange(nclusters)
    for d_ind, dset in enumerate(which_dsets):
        for c_ind, clust in enumerate(clusts):
            for r_ind, region in enumerate(regions):
                all_spatial_dist[d_ind, c_ind, r_ind] = \
                all_clusters_region_dist[dset][clust][region]

    for d_ind, dset in enumerate(which_dsets):
        for c_ind, clust in enumerate(clusts):
            for r_ind, region in enumerate(regions):
                all_total_cells[d_ind, c_ind, r_ind] = \
                all_total_cells_per_region[dset][region]

    return (all_spatial_dist, all_total_cells)

def plot_clusters_per_region(all_spatial_dist,
                             all_total_cells,
                             regions,
                             task_class_names,
                             do_normalize=True,
                             bar_width=0.15,
                             start_cluster=1,
                             group_colors=['g', 'b', 'r', 'k', 'y'],
                             group_by_region=False):
    """Make barchart plot of results from organize_clusters_per_region(),
    averaging across mice (datasets).

    :param start_cluster: int. If you want to exclude cluster 0, then set this to 1.
    """
    vals = np.copy(all_spatial_dist)
    if do_normalize:
        vals /= all_total_cells

    d_means = np.mean(vals, axis=0)
    d_err = scipy.stats.sem(vals, axis=0)

    if not group_by_region:
        nregions = len(regions)
        r = np.arange(all_spatial_dist.shape[1] - start_cluster) - bar_width*nregions/2
        plot_handles = []
        for i in range(nregions):
            p = plt.bar(r, d_means[start_cluster:, i],
                         color=group_colors[i],
                         width=bar_width,
                         yerr=d_err[start_cluster:, i])
            r = [x + bar_width for x in r]
            plot_handles.append(p)

        plt.legend(plot_handles, regions)
        plt.xticks(np.arange(len(task_class_names[start_cluster:])),
                   task_class_names[start_cluster:])
        if do_normalize:
            plt.ylabel('Fraction of total sources in region')
        else:
            plt.ylabel('Number of sources')
        plt.title('Distribution of task-classes by region')
    else:
        nclusts = d_means.shape[0] - 1
        r = np.arange(nclusts) - bar_width * nclusts / 2.0
        plot_handles = []

        for i in range(start_cluster, start_cluster + nclusts):
            print(r)
            print(group_colors[i])
            print(d_means[i, :])
            print(d_err[i, :])
            x = np.array(r)
            if np.mod(len(r), 2) == 1:
                x = x + bar_width/2
            p = plt.bar(x, d_means[i, :],
                        color=group_colors[i],
                        width=bar_width,
                        yerr=d_err[i, :])
            r = [x + bar_width for x in r]
            plot_handles.append(p)
        plt.xticks(np.arange(len(regions)), regions)
        plt.legend(plot_handles, task_class_names[start_cluster:])
        if do_normalize:
            plt.ylabel('Fraction of regional sources')
        else:
            plt.ylabel('Number of sources')
        plt.title('Distribution of task-classes by region')


def thresh_corrs(corrs, pvals, pthresh=None, valthresh=None):
    """Helper function to threshold correlation values"""

    if pthresh is not None:
        not_sig = pvals > (pthresh / len(corrs) / len(corrs))
        corrs[not_sig] = np.nan
    if valthresh is not None:
        corrs[np.where(corrs < valthresh*np.nanmax(corrs))] = np.nan
        # corrs[np.where(corrs<0.1)] = np.nan

    return corrs


def summarize_corr_trial_avg_vs_single_trial(data_info, allCT,
                                             all_centroids, datasets,
                                             fig_save_dir):
    """
    Plot spatial correlation map for a specified seed source,
    using either single-trial or trial-averaged traces.
    Plot correlation vs. distance.
    Plot the single-trial traces for the specified source.
    Plot raw traces at moments of high correlation with a neighbor source.
    Plot the spatial location of the seed and neighbor.

    :param data_info: dict. Must contain keys 'dataset_id', 'which_source',
                      'window_size', 'partner_index', 'traces_ylim', 'corr_ylim'.
    :param allCT: list containing CosmosTraces objects for all loaded datasets.
    :param all_centroids: output from load_centroids_and_task_labels().
    :param datasets: List specifying the ordering of datasets in allCT and all_centroids.
    :param fig_save_dir: If not None, location for saving figures.
    :return:
    """

    # Gather info about which sources to plot.
    dataset_id = data_info['dataset_id']
    which_source = data_info['which_source']
    window_size = data_info['window_size']
    partner_index = data_info['partner_index']
    traces_ylim = data_info['trace_ylim']
    corr_ylim = data_info['corr_ylim']

    # Set up saving.
    if fig_save_dir is not None:
        savedir = os.path.join(fig_save_dir, str(dataset_id))
        os.makedirs(savedir, exist_ok=True)
    else:
        savedir = None

    CT_ind = int(np.where(np.array(datasets) == dataset_id)[0])
    CT = allCT[CT_ind]
    centroids = all_centroids[dataset_id]

    ### Compute correlation between sources. ###
    (all_c, all_d, all_p,
     all_source_ids, smoothed_spikes) = get_all_correlation_vs_dist(
         ['full', '4way'],
         CT,
         centroids,
         which_hemisphere=None,
         do_binarize=False,
         do_zscore=False,
         return_flattened=False)

    ### Now make plots of correlation map for the specified source. ###
    cell_id = all_source_ids['full'][which_source]
    savename = 'id' + str(dataset_id) + '_corr_vs_dist_{}'.format(cell_id)

    # Plot correlation spatial map, and correlation vs. distance.
    compare_corr_trial_avg_vs_single_trial(which_source, CT, dataset_id,
                                           all_c, all_d, all_p,
                                           all_source_ids,
                                           max_radius=10, bin_range=[0, 7],
                                           bin_size=0.5, ylim=corr_ylim,
                                           do_pthresh=False,
                                           do_valthresh=False,
                                           use_abs_val=True)
    if fig_save_dir is not None:
        print(fig_save_dir)
        plt.gcf().set_size_inches(w=5, h=4)
        plt.savefig(os.path.join(savedir, savename + '.pdf'),
                    transparent=True, rasterized=True, dpi=600)

    # Plot single-trial traces for the specified source.
    plot_formatted_cell_across_trials_wrapper(cell_id, CT, ylims_avg=[0, 7])
    if fig_save_dir is not None:
        plt.gcf().set_size_inches(w=0.75, h=1.3)
        plt.savefig(os.path.join(savedir, savename + '_traces.pdf'),
                    transparent=True, rasterized=True, dpi=600)


    ### Plot trace of seed and neighbor source from a small time window. ###
    flat_smooth_spikes = utils.flatten_traces(smoothed_spikes)
    flat_spikes = utils.flatten_traces(CT.St)
    flat_fluor = utils.flatten_traces(CT.Ft)

    # Order sources by correlation with the seed.
    c_source = np.copy(all_c['full'][which_source, :])
    c_source[which_source] = 0
    sorted_sources = np.argsort(-c_source)

    sources_to_corr = [which_source, sorted_sources[partner_index]]
    sources_to_plot = [which_source, sorted_sources[partner_index]]
    traces_to_corr = flat_smooth_spikes

    if fig_save_dir is not None:
        savedir = os.path.join(fig_save_dir, str(dataset_id), str(partner_index))
        os.makedirs(savedir, exist_ok=True)
    else:
        savedir = None

    frames_to_plot = plot_sources_at_high_correlation_timepoints(
        sources_to_corr,
        traces_to_corr,
        sources_to_plot,
        flat_spikes, flat_fluor,
        savedir=savedir,
        window_size=window_size,
        n_to_plot=5,
        clim=[0, 3],
        do_overlay_traces=True,
        ylim=traces_ylim,
        do_zscore=True)

    # Now plot the location of the sources.
    radii = np.array([1, 1.05])
    CT.centroids_on_atlas(radii, sources_to_corr, max_radius=10,
                          set_alpha=True)

    if savedir is not None:
        savename = 'sources_{}_{}.pdf'.format(sources_to_corr[0],
                                              sources_to_corr[1])
        plt.savefig(os.path.join(savedir, savename),
                    transparent=True, rasterized=True, dpi=600)





def get_all_correlation_vs_dist(which_settings,
                                CT, centroids,
                                which_hemisphere,
                                do_binarize=False,
                                do_zscore=False,
                                return_flattened=False):
    """
    Wrapper around get_correlation_vs_dist() that fills dicts
    with the correlations for multiple settings (i.e. single-trial
    and trial-averaged).
    :param which_settings: list. i.e. ['full', '4way']
    :param CT: CosmosTraces object
    :param centroids: atlas aligned and scaled centroids (i.e. in units of mm)
                      from load_centroids_and_task_labels()
    :param which_hemisphere: None means both hemispheres. Otherwise 0 or 1.
    :param do_binarize: bool. Binarize spikes.
    :param do_zscore: bool. Zscore traces.
    :param returned_flattened:
    :return:
    all_c: dict. keys are which_settings. val is the correlation matrices across all sources.
    all_d: dict. keys are which_settings. val is a matrix the distances between all sources.
    all_p: dict. keys are which_settings. val is the pvalue for each correlation.
    all_source_ids: dict. keys are which_settings. val a list of the source_id corresponding to
                    each row of the correlation matrix.
    smoothed_spikes: the processed traces used for computing the correlation. This is returned so
                     that you can use those exact traces for subsequent computation.

    """
    all_c = dict()
    all_d = dict()
    all_p = dict()
    all_source_ids = dict()
    for which_traces in ['full', '4way']:
        c, d, p, which_sources, smoothed_spikes = get_correlation_vs_dist(
            CT,
            centroids,
            which_traces=which_traces,
            which_hemisphere=None,
            do_binarize=False, do_zscore=False,
            return_flattened=False,
            return_spikes=True)
        key = which_traces
        all_c[key] = c
        all_d[key] = d
        all_p[key] = p
        all_source_ids[key] = which_sources

    return all_c, all_d, all_p, all_source_ids, smoothed_spikes


def get_correlation_vs_dist(CT, centroids, which_traces,
                            which_hemisphere=None, do_binarize=False,
                            do_zscore=True, return_flattened=True,
                            return_spikes=False):
    """
    For each pair of centroids, computes the distance between
    the centroids, and the correlation between their traces.

    :param CT: CosmosTraces object.
    :param centroids: atlas aligned and scaled centroids (i.e. in units of mm)
                      from load_centroids_and_task_labels()
    :param which_traces: 'full' (single-trial) or '4way' (trial-averaged, but
                         separated by the 4 trial types)
    :param which_hemisphere: 0 or 1.
    :param do_binarize: bool. If True, will binarize spikes.
    :param do_zscore: Zscore the traces before computing the correlation.
    :param return_flattened: bool. Flatten the correlation matrices.
    :param return_spikes: bool. Return the processed traces (i.e. smoothed spikes)
                          used for actually computing the correlation.
    :return:
        c: [num_centroid_pairs], correlation between each pair
        d: [num_centroid_pairs], distance between each pair
        p: [num_centroid_pairs], pvalue of each correlation
    """

    spikes = CT.St
    if do_binarize:
        spikes = (spikes > 0).astype(float)

    smooth_spikes = gaussian_filter1d(spikes, 1.5, axis=1, mode='constant')
    trial_sets, trial_names = get_trial_sets(CT.bd)


    rates = smooth_spikes
    if which_traces == 'full':
        rates_flat = np.reshape(rates,
                                (rates.shape[0], rates.shape[1]*rates.shape[2]),
                                order='F')
    elif which_traces == '4way':
        rates_flat = concatenate_trial_type_avgs(trial_sets, rates,
                                                    do_plot=False)
    else:
        raise NotImplementedError('{} not implemented'.format(which_traces))

    if do_zscore:
        rates_flat = scipy.stats.zscore(rates_flat)

    # Compute correlation between all pairs
    # corr_full = np.corrcoef(rates_flat)
    corr, p_corr = scipy.stats.spearmanr(rates_flat, axis=1)

    dists = scipy.spatial.distance.cdist(centroids, centroids, 'euclidean')


    # Extract sources from specified hemisphere
    if which_hemisphere is not None:
        which_sources = np.where(CT.hemisphere_of_cell == which_hemisphere)[0]
    else:
        which_sources = np.arange(len(CT.hemisphere_of_cell))

    which_corr = corr[which_sources, :][:, which_sources]
    which_dists = dists[which_sources, :][:, which_sources]
    which_p = p_corr[which_sources, :][:, which_sources]

    if return_flattened:
        inds = np.triu_indices(which_dists.shape[0], k=1)
        c = which_corr[inds[0], inds[1]]
        d = which_dists[inds[0], inds[1]]
        p = which_p[inds[0], inds[1]]
    else:
        c = which_corr
        d = which_dists
        p = which_p


    if return_spikes:
        return c, d, p, which_sources, smooth_spikes
    else:
        return c, d, p, which_sources

def summarize_binned_correlation_vs_dist(all_c, all_d, bin_range, bin_size, use_abs_val=True):
    """
    Summarize correlation vs. distance by binning all source-pairs
    based on their distance, and computing the mean and sem of
    the correlations within that bin.
    Uses output from get_correlation_vs_dist()

    :param all_c: dict. each entry is the output from get_correlation_vs_dist()
                  the correlation between pairs, using  either single-trial or
                  trial-averaged traces, and various subsets of sources.
    :param all_d: dict. each entry is the output from get_correlation_vs_dist()
                  the distance between pairs, using  either single-trial or
                  trial-averaged traces, and various subsets of sources.
    :param bin_size: in units of mm. distance to use for binning.
    :param use_abs_val: bool. Use absolute value of correlation so that you are
                        measuring magnitude as opposed to direction. Is this
                        necessary?
    :return:
    """

    all_m = dict()
    all_s = dict()

    bins = np.arange(bin_range[0], bin_range[1], bin_size)

    for key in all_c.keys():
        d = all_d[key]
        all_m[key] = np.array([])
        all_s[key] = np.array([])
        for val in bins:
            in_bin = np.logical_and((d - val) < bin_size, (d - val) > 0)
            inds = np.where(in_bin)[0]
            corr = all_c[key][inds]
            if use_abs_val:
                corr = np.abs(corr)

            mean_corr = np.nanmean(corr)
            sem_corr = scipy.stats.sem(corr, nan_policy='omit')

            # Use absolute value of correlation
            all_m[key] = np.append(all_m[key], mean_corr)
            all_s[key] = np.append(all_s[key], sem_corr)

    return all_m, all_s, bins

def plot_correlation_vs_dist(all_m, all_s, bins, do_normalize=True,
                             colors=None):
    """
    Plot binned summary of correlation vs. distance.
    Uses output from summarize_binned_correlation_vs_dist()

    :param all_m: dict. mean bin value for each set of sources.
    :param all_s: dict. mean sem values for each set of sources.
    :param bins: the binned distances.
    :param do_normalize: bool. Normalize to the max correlation.
    :return:
    """
    labels = list(all_m.keys())
    labels.sort()
    for key in labels:
        m = all_m[key]
        s = all_s[key]
        if do_normalize:
            n = np.nanmax(m)
        else:
            n = 1

        if colors is not None:
            c = colors[key]
        else:
            c = None
        plt.plot(bins, m/n, color=c)
        plt.fill_between(bins, m/n-s/n, m/n+s/n,
                         alpha=0.5, color=c)

    plt.ylabel('Normalized correlation')
    plt.xlabel('Distance (mm)')
    plt.axvline(0.15, linestyle='--', color='k')
    plt.axvline(0.0, linestyle='--', color='k')
    plt.legend(labels, bbox_to_anchor=(1.3, 1))


def plot_corr_trial_avg_vs_single_trial(which_source, CT, dataset_id,
                                        all_c, all_d, all_p, all_source_ids,
                                        max_radius=10,
                                        do_pthresh=False, do_valthresh=False,
                                        use_abs_val=True, do_save=True,
                                        fig_save_dir=None, smoothed_spikes=None,
                                        bin_range=[0.15, 6], bin_size=0.5):
    """
    Plot seeded correlation for a specific source, for trial-averaged and single-trial correlations.
    Plot the raw correlation vs distance.
    Plot the single trial traces, divided by trial type.

    :param which_neuron:
    :param CT:
    :param all_c:
    :param all_d:
    :param all_p:
    :param all_source_ids:
    :param do_pthresh:
    :param do_valthresh:
    :param use_abs_val:
    :param do_save:
    :param fig_save_dir:
    :param smoothed_spikes: Optionally provide precomputed smoothed_spikes (for efficiency)
    :return:
    """

    cell_id = all_source_ids['full'][which_source]
    savename = 'id' + str(dataset_id) + '_corr_vs_dist_{}'.format(cell_id)

    compare_corr_trial_avg_vs_single_trial(which_source, CT, dataset_id,
                                              all_c, all_d, all_p,
                                              all_source_ids,
                                              max_radius=10,
                                              do_pthresh=False,
                                              do_valthresh=False,
                                              use_abs_val=True,
                                              bin_range=bin_range,
                                              bin_size=bin_size)


    if do_save and fig_save_dir is not None:
        plt.gcf().set_size_inches(w=5, h=4)
        plt.savefig(fig_save_dir + savename + '.pdf', transparent=True,
                    rasterized=True, dpi=600)

    plot_formatted_cell_across_trials_wrapper(cell_id, CT, ylims_avg=None, smoothed_spikes=smoothed_spikes)
    if do_save and fig_save_dir is not None:
        plt.gcf().set_size_inches(w=0.75, h=1.3)
        plt.savefig(fig_save_dir + savename + '_traces.pdf', transparent=True,
                    rasterized=True, dpi=600)

    print(fig_save_dir)

def compare_corr_trial_avg_vs_single_trial(which_source, CT, dataset_id,
                                           all_c, all_d, all_p, all_source_ids,
                                           max_radius=50,
                                           do_pthresh=False, do_valthresh=False, use_abs_val=True,
                                           bin_range=[0.15, 6], bin_size=0.5, ylim=[0, 0.5]):
    """
    Plot seeded correlation for a specific source, for trial-averaged and single-trial correlations.
    Plot the raw correlation vs distance.

    :param which_neuron:
    :param CT:
    :param all_c:
    :param all_d:
    :param all_p:
    :param all_source_ids:
    :param do_pthresh:
    :param do_valthresh:
    :param use_abs_val:
    :return:
    """

    all_ms = dict() # All means across trialavg/singletrial

    fig = plt.figure(figsize=(10, 8))
    gs = []
    gs.append(plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1))
    gs.append(plt.subplot2grid((2, 2), (0, 1), colspan=1))
    gs.append(plt.subplot2grid((2, 2), (1, 0), colspan=1))
    gs.append(plt.subplot2grid((2, 2), (1, 1), colspan=1))
    # gs.append(plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=2))

    colors = ['r', 'b']
    for iter, key in enumerate(['4way', 'full']):
        corrs = np.copy(all_c[key][which_source, :])
        dists = np.copy(all_d[key][which_source, :])
        pvals = np.copy(all_p[key][which_source, :])
        source_ids = all_source_ids[key]

        pthresh = 1e-3 if do_pthresh else None
        valthresh = 0.5 if do_valthresh else None

        corrs[which_source] = 0.01 # Zero out the seed neuron so it does not obstruct the view
        corrs = thresh_corrs(corrs, pvals, pthresh=pthresh, valthresh=valthresh)
        normed_corrs = corrs/np.nanmax(corrs)

        # Plot the spatial distribution
        if use_abs_val:
            normed_corrs = np.abs(normed_corrs)

        plt.subplot(gs[iter])
        CT.centroids_on_atlas(normed_corrs, source_ids, max_radius=max_radius,
                              set_alpha=True, highlight_inds=[which_source])
        plt.title('{}: {} ({})'.format(dataset_id, source_ids[which_source], which_source))

        # Now summarize correlation vs. distance
        if use_abs_val:
            corrs_to_use = np.abs(corrs)
        else:
            corrs_to_use = corrs

        all_m, all_s, bins = summarize_binned_correlation_vs_dist(
            {key: corrs_to_use},
            {key: dists},
            bin_range=bin_range,
            bin_size=bin_size,
            use_abs_val=use_abs_val)

        plt.subplot(gs[iter+2])
        plt.plot(dists, corrs_to_use, '.', color=[0.7, 0.7, 0.7], markersize=1)
        plt.plot(bins+bin_size/2, all_m[key], color=colors[iter])
        plt.ylim(ylim)
        plt.xlim([bin_range[0]-0.5, bin_range[1]+0.5])
        plt.xticks(np.arange(0, np.max(bin_range)+1, 2))
        # plt.ylim([0, 0.2])
        if iter == 0:
            plt.ylabel('|correlation|')
        plt.xlabel('Separation distance (mm)')

        all_ms[key] = all_m # Log for final summary plot



def plot_formatted_cell_across_trials_wrapper(cell_id, CT, ylims_avg=[0, 7], smoothed_spikes=None):
    """
    Wrapper to simplfiy call to trace_analysis_utils.plot_formatted_cell_across_trials.

    :param cell_id: global id of the source (i.e. index into CT)
    :param CT: CosmosTraces object.
    :param ylims_avg: ylim of the plot of average trace across trials.
    :return:
    """

    lick_onsets = utils.get_lick_onsets(CT.bd.spout_lick_rates)
    if smoothed_spikes is None:
        smoothed_spikes = gaussian_filter1d(CT.St, 1.5, axis=1, mode='constant') ## This is inefficient to call every time, but convenient for now...

    trial_sets, names = get_trial_sets(CT.bd, use_all_trials=True)
    trial_colors = ['orange', 'c', 'r', 'g']
    event_frames = CT.fps * np.array([CT.bd.stimulus_times[0],
                                      CT.bd.stimulus_times[0] + 1.5])
    utils.plot_formatted_cell_across_trials(cell_id, smoothed_spikes, CT.Tt,
                                            trial_sets, names, trial_colors,
                                            event_frames, CT.centroids,
                                            CT.atlas_tform,
                                            clim=[0, 4],
                                            lick_onsets=lick_onsets,
                                            xlims=[-2, 4],
                                            ylims_trials=[0, 25],
                                            ylims_avg=ylims_avg)

def plot_cluster_spread_comparison(all_clust, keys,
                                   min_clust_size=5, do_plot=True):
    """
    Plot histograms of spatial spread of clusterings.
    Compare clusterings defined by 'keys', which indexes
    into all_clust, which is a dict containing clustering results.

    :param all_clust: dict. each entry is a clustering result with certain parameters.
    :param keys: list of strings. which clustering results to compare/overlay.
    :param min_clust_size: int. only include clusters that contain a minimum
                           number of sources.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        colors = [(1, 0, 1, 0.7),
                  (1, 0.5, 0, 0.7)]
        clust_spreads = {}
        for ind, key in enumerate(keys):
            med_cluster_spread, _ = compute_cluster_spread(
                                                    all_clust[key]['ordered_clustering'],
                                                    all_clust[key]['centroid_atlas_coords'],
                                                    do_split_hemispheres=False)
            clust_size = compute_cluster_sizes(all_clust[key]['ordered_clustering'])
            clust_spread = med_cluster_spread[clust_size > min_clust_size]
            clust_spreads[key] = clust_spread

            if do_plot:
                plt.hist(clust_spread, color=colors[ind],
                         density=True, bins=np.linspace(0, 120, 15))

        if do_plot:
            handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
            labels = keys
            plt.xlabel('Median cluster spread, distance [au]')  ### ---> TODO: Put this in terms of mm based on the atlas?
            plt.ylabel('Probability')
            plt.xticks([0, 50, 100])
            legend = plt.legend(handles, labels, frameon=False,
                                bbox_to_anchor=(1.2, 0.5), handlelength=0.5,
                                handletextpad=0.3, labelspacing=0.2)
        return clust_spreads


def concatenate_trial_type_avgs(trial_sets, rates, do_plot=False,
                                get_first_half=False, get_second_half=False):
    """
    Concatenate trial types to obtain the mean for each trial type,
    where trial types are defined based on the provided trial_sets.
    :param trial_sets: list of boolean arrays each of length [ntrials].
                       each array indicates which trials are included in that
                       set of trials.
    :param rates: the full data array. [ncells x ntime x ntrials]
    :param do_plot: bool. Optionally plot the concatenated array.
    :returns type_mean: [ncells x ntime_per_trial*len(trial_sets)]. The average
                        trace for each cell, with each trial_set concatenated.
    """
    type_means = []
    for trial_set in trial_sets:
        trial_inds = np.where(trial_set)[0]
        if get_first_half:
            # trial_inds = trial_inds[:int(len(trial_inds) / 2)]
            trial_inds = trial_inds[0::2]
        elif get_second_half:
            # trial_inds = trial_inds[int(len(trial_inds) / 2):]
            trial_inds = trial_inds[1::2]

        # type_means.append(np.mean(rates[:, :, trial_set], axis=2))
        type_means.append(np.mean(rates[:, :, trial_inds], axis=2))
    type_mean = np.hstack(type_means)


    if do_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(type_mean, aspect='auto')

    return type_mean


def get_cluster_basis(clustering):
    """
    Convert cluster labels into a basis set.

    :param clustering: [ncells] A cluster assignment for each source.
    """
    basis = np.zeros((len(clustering), np.max(clustering) + 1))
    for source, clust in enumerate(clustering):
        basis[source, clust] = 1
    return basis


def get_cluster_explained_variances(rates, which_traces, all_nmf, trial_sets):
    """
    Compute how much variance a clustering basis set explains
    for a given set of traces, indicated by the 'which_traces' parameter.
    :param rates: [ncells x ntime x ntrials] The full dataset of
                    smoothed spike rates for each source for each trial.
    :param which_traces: str. '4way' - Average trace for each of the four
                                       task conditions, concatenated.
                              'baseline' - The unaveraged inter-trial-interval
                                           segments of the full trace.
                              'full' - The unaveraged full trace.
    :param all_nmf: dict. has a key for each clustering basis.
                    the entry is also a dict that contains a
                    field 'clustering', which is the assignment
                    of each source to a cluster.
    :param trial_sets: tuple of ntrial_types arrays. Each array is boolean,
                       of length ntrials, and indicates whether
                       each trial is included in that trial type.
    :return evrs: dict. for each clustering basis set in all_nmf,
                        the explained variance of each cluster.
    """
    if which_traces == '4way':
        traces = concatenate_trial_type_avgs(trial_sets, rates,
                                             do_plot=False)
    elif which_traces == 'baseline':
        baseline_inds = np.hstack((np.arange(3, 64), np.arange(170, 203)))
        baseline_rates = rates[:, baseline_inds, :]
        baseline_flat = np.reshape(baseline_rates,
                                   (baseline_rates.shape[0],
                                    baseline_rates.shape[1] *
                                    baseline_rates.shape[2]), order='F')
        traces = baseline_flat
    elif which_traces == 'full':
        rates_flat = np.reshape(rates, (
        rates.shape[0], rates.shape[1] * rates.shape[2]), order='F')
        traces = rates_flat

    evrs = {}
    for clust in all_nmf.keys():
        cluster_basis = get_cluster_basis(all_nmf[clust]['clustering'])
        print(clust, str(cluster_basis.shape))
        evrs[clust] = utils.explained_variance_of_basis(traces, cluster_basis)

    return evrs

def factorize_data(data, n_components, method='NMF', do_plot=False, randseed=1,
                   l1_ratio=0.0):
    """
    Compute NMF decomposition on the provided matrix,
    with specified number of components.
    Zhat = W*H is the best NMF approximation of Z = data.T
    :param data: [ncells x ntime].
    :param n_components: int. number of components for the decomposition.
    :param method: 'NMF' (not yet included: 'PCA', maybe 'kmeans')
    :param do_plot: bool. If True then plots explained variance.

    :returns model: the trained sklearn model.
             H: [n_components x cells] Weight of each cell in a cluster.
             W: [time x n_components] The time series of each cluster.
             evr: [n_components]. Explained variance of each cluster.

    """
    X = data
    if method == 'NMF':
        model = NMF(n_components=n_components, verbose=0, random_state=randseed,
                    l1_ratio=l1_ratio, alpha=l1_ratio)
        W = model.fit_transform(X.T)
        H = model.components_
        evr = utils.explained_variance_of_basis(X, H.T)
    else:
        raise('method {} not yet implemented in factorize_data().'.format(method))

    if do_plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(
            evr[np.argsort(-evr)],'r')
        plt.title('Explained variance')
        plt.xlabel('basis #')
        plt.subplot(1, 2, 2)
        plt.plot(np.cumsum(
            evr[np.argsort(-evr)]), 'r')
        plt.title('cumsum explained var')
        plt.xlabel('basis #')

        Zhat = np.matmul(W, H)
        Z = data.T
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(Zhat[:100, 0], 'b')
        plt.title('Reconstructed trace')
        plt.subplot(2, 1, 2)
        plt.plot(Z[:100, 0], 'r')
        plt.title('True trace')


    return model, H, W, evr


def hierarchical_cluster(data, n_clusters):
    """
    Cluster data into n_clusters using hierarchical
    clustering with a correlation based metric.

    :param data: [ncells x ntime]
    :param n_clusters: int.
    :return: clustering - cluster assignment for each cell
             Z - the linkage datastruct from hierarchical ordering.
    """
    Z = linkage(data, method='weighted', metric='correlation')
    # Z = linkage(data, method='single', metric='correlation')
    # Z = linkage(data, method='ward', metric='euclidean')

    clustering = cut_tree(Z, n_clusters=n_clusters)
    clustering = clustering.flatten()

    return clustering, Z

def cluster_from_factorization(H, W, data_to_recompute_cluster_means=None):
    """
    Assign each cell to a cluster based on the output
    from factorize_data().
    Currently just uses whichever component has the highest
    coefficient for each cell.

    :param H: [clusters x cells]. Weight of each cell in a cluster.
    :param W: [time x clusters]. The time series of each cluster.
    :param data_to_recompute_cluster_means: Optional. Set to None to exclude.
                If None, then will simply use the NMF output W as the time series
                for each cluster.
                Alternatively, you can compute a slightly different
                time series by applying the factorized basis to the
                full time series and then averaging within each trial type.
                If this is desired, provide a dict with the following entries:
                    model - the learned sklearn model
                    rates_unaveraged - [ncells, ntime] (the full time series)
                    trial_sets - list of boolean arrays each of length ntrials.
                             each array indicates which trials are included in that
                             set of trials.
                    ntrials - number of total trials
                    trial_ntime - number of frames in each trial
                i.e.:
                recompute_clusts = {'model':model, 'rates_unaveraged':rates_flat,
                                    'trial_sets':reduced_trial_sets,
#                                   'ntrials':CT.ntrials, 'trial_ntime':CT.St.shape[1]}

    """
    if data_to_recompute_cluster_means is not None:
        model = data_to_recompute_cluster_means['model']
        rates_unaveraged = data_to_recompute_cluster_means['rates_unaveraged']
        trial_sets = data_to_recompute_cluster_means['trial_sets']
        ntrials = data_to_recompute_cluster_means['ntrials']
        trial_ntime = data_to_recompute_cluster_means['trial_ntime']

        Wflat = model.transform(rates_unaveraged.T)
        X_transform = Wflat
        X_transform_trial = np.reshape(X_transform.T,
                                       (X_transform.shape[1], trial_ntime,
                                        ntrials),
                                       order='F')
        cluster_means = []
        for trial_set in trial_sets:
            cluster_means.append(
                np.mean(X_transform_trial[:, :, trial_set], axis=2))
        clust_means = np.hstack(cluster_means)
    else:
        clust_means = W.T

    clustering = np.argmax(H, axis=0)  ## Cluster neurons by assigning to nmf basis
                                       ## with the largest weight.
                                       ## Potentially, you could do better than this.

    return clustering, clust_means


def order_clusters(clust_means, do_plot=True, vertical_lines=None,
                   titlestr=None, trange=None, use_min=False, figsize=(10, 5),
                   clust_means_to_plot=None):
    """
    Order clusters by peak-time of the cluster time series.
    :param clust_means: [nclusters x ntime]. The time series of each cluster.
    :param do_plot: bool. Optionally plot the ordered clusters.
    :param vertical_lines: None or list of ints. Optionally draw vertical lines.
    :param trange: np.array of frames.
                   If not None, then only computes ordering based off
                   the signal in the provided range of frames.
    :return ordering: [nclusters]. ordering[0] is the new position of cluster 0.
    """
    if trange is None:
        trange = np.arange(clust_means.shape[1])

    if clust_means_to_plot is None:
        clust_means_to_plot = clust_means

    # Order clusters by peak time.
    ordering = np.argsort(np.argmax(clust_means[:, trange], axis=1))
    if use_min:
        ordering = np.argsort(np.argmin(clust_means[:, trange], axis=1))

    if do_plot:
        plt.figure(figsize=figsize)
        plt.imshow(scipy.stats.zscore(
                   clust_means_to_plot[ordering, :], axis=1),
                   aspect='auto', clim=[-1, 5])
        if vertical_lines is not None:
            [plt.axvline(x, color='r') for x in vertical_lines]
        if titlestr is not None:
            plt.title(titlestr)

    return ordering


def plot_cluster_member_traces(rates, clustering, which_cluster, which_trials,
                               event_frames, cmap):
    """
    Plot all of the member sources of a cluster across trials.

    :param rates: the full data array. [ncells x ntime x ntrials]
    :param clustering: [ncells] A cluster assignment for each source.
    :param which_cluster: int. ID of the cluster to plot.
    :param which_trials: int or list. which trials to plot.
    :param event_frames: list of floats. Frame number of task events for one trial,
                         i.e. [trial_start, odor_start, reward_onset].
    :param cmap: string. name of the colormap to use.
    """

    which_sources = np.where(clustering == which_cluster)[0]
    cluster_rates = rates[which_sources, :, :]
    trial_rates = cluster_rates[:, :, which_trials]
    trial_rates = np.reshape(trial_rates, (trial_rates.shape[0],
                                           trial_rates.shape[1] *
                                           trial_rates.shape[2]), order='F')
    trial_starts = np.arange(len(which_trials)) * cluster_rates.shape[1]
    plt.imshow(trial_rates, cmap=cmap, aspect='auto')
    plt.title(which_trials)
    if event_frames is not None:
        for e in event_frames:
            [plt.axvline(x + e, color='r', linewidth=0.5) for x in trial_starts]
    return trial_rates


def plot_cluster_means_across_trials(rates, clustering, which_cluster,
                                     trial_sets, trial_set_names=None,
                                     event_frames=None, cmap='jet'):
    """
    Plot individual trials of the cluster mean.

    :param rates: the full data array. [ncells x ntime x ntrials]
    :param clustering: [ncells] A cluster assignment for each source.
    :param which_cluster: int. ID of the cluster to plot.
    :param trial_sets: tuple of ntrial_types arrays. Each array is boolean,
                       of length ntrials, and indicates whether
                       each trial is included in that trial type.
    :param trial_set_names: tuple of ntrial_types strings. The name of
                            each trial type.
    :param event_frames: list of floats. Frame number of task events for one trial,
                         i.e. [trial_start, odor_start, reward_onset].
    :param cmap: string. name of the colormap to use.
    """
    ntrial_types = len(trial_sets)
    nr = 7
    nc = ntrial_types
    plt.figure()
    gs = []
    for i in range(ntrial_types):
        gs.append(plt.subplot2grid((nr, nc), (0, i), colspan=1, rowspan=7))

    which_sources = np.where(clustering == which_cluster)[0]
    maxy = 0
    for trial_set in trial_sets:
        maxval = np.max(
            np.mean(rates[which_sources, :, :][:, :, trial_set], axis=0))
        if maxval > maxy:
            maxy = maxval

    for i in range(len(trial_sets)):
        plt.subplot(gs[i])
        cluster_mean = np.mean(rates[which_sources, :, :][:, :, trial_sets[i]],
                               axis=0).T
        plt.imshow(cluster_mean,
                   aspect='auto',
                   interpolation='nearest',
                   cmap=cmap)
        #         plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.gca().set_yticks([])
        #         plt.gca().set_xticks([])
        plt.xlim([0, 181])
        plt.clim([0, 0.5*maxy])
        if event_frames is not None:
            [plt.axvline(ef, color='k', linewidth=0.5) for ef in event_frames]
        if trial_set_names is not None:
            plt.title(trial_set_names[i])
        if i == 0:
            plt.ylabel('trial')


#         mean_trace = np.mean(W_trial[pc,:,np.where(trial_sets[i])[0]], axis=0)
#         mean_traces.append(mean_trace)
#         if show_trace:
#             plt.subplot(gs[i+nstart])
#             plt.plot(mean_trace)
#             [plt.axvline(ef,  color='r', linewidth=1, alpha=0.5) for ef in event_frames]
#             plt.ylim([0, maxy])


#     return mean_traces

def generate_super_clusters(protocol, clust_means, event_frames, trial_nframes):
    """
    Assigns clusters to super-clusters, one hierarchical level higher.
    These super-clusters are defined based on specific aspects of the task,
    i.e. clusters that are active before the odor, or only after the
    odor.

    :param protocol: string. Defines how to generate superclusters.
                     Options: '2way', '4way', 'oeg2way'.
    :param clust_means: [nclust x ntime] The time series of each cluster.
    :param event_frames: list of floats. Frame number of task events for one trial,
                         i.e. [trial_start, odor_start, reward_onset].
    :param trial_nframes: int. Number of frames in one trial.

    :return super_clust_info: a dict containing:
                -super_clustering: [nclusters]. Assignment of each
                              cluster to one of the superclusters.
                -titles: list of strings. The name of each supercluster.
    """

    if protocol == '2way':
        odor_buffer = 3 ## Include a few frames past odor onset (before motor initiation).
        odor_start = int(event_frames[1]) + odor_buffer
        reward_start = int(event_frames[2])
        pre_go_odor = [[0, odor_start]]
        pre_nogo_odor = [[trial_nframes, trial_nframes+odor_start]]
        post_go_odor_onset = [[odor_start, trial_nframes]]
        post_nogo_odor_onset = [[odor_start+trial_nframes, 2*trial_nframes]]

        during_odor = [[x*trial_nframes+odor_start, x*trial_nframes+odor_start+30]
                       for x in np.arange(4)]
        post_odor_onset = [[x*trial_nframes+odor_start, x*trial_nframes+trial_nframes]
                           for x in np.arange(4)]

        peaks = np.argmax(clust_means, axis=1)
        pre_go_odor_clusts = utils.in_ranges(pre_go_odor, peaks)
        pre_nogo_odor_clusts = utils.in_ranges(pre_nogo_odor, peaks)
        post_go_odor_onset_clusts = utils.in_ranges(post_go_odor_onset, peaks)
        post_nogo_odor_onset_clusts = utils.in_ranges(post_nogo_odor_onset, peaks)

        ### Now assign each cluster to a super-cluster.
        super_clustering = np.zeros(clust_means.shape[0]).astype(int)
        super_clustering[pre_go_odor_clusts] = 0
        super_clustering[pre_nogo_odor_clusts] = 1
        super_clustering[post_go_odor_onset_clusts] = 2
        super_clustering[post_nogo_odor_onset_clusts] = 3

        super_cluster_titles = ['pre-go-odor', 'pre-nogo-odor',
                                'post-go-odor', 'post-nogo-odor']

    elif protocol == '4way' or protocol == 'rand_4way':
        odor_buffer = 3 ## Include a few frames past odor onset (before motor initiation).
        odor_start = int(event_frames[1]) + odor_buffer
        reward_start = int(event_frames[2])
        pre_odor = [[x * trial_nframes, x * trial_nframes + odor_start] for x in
                    np.arange(4)]
        during_odor = [[x * trial_nframes + odor_start,
                        x * trial_nframes + odor_start + 30] for x in
                       np.arange(4)]
        post_odor_onset = [
            [x * trial_nframes + odor_start, x * trial_nframes + trial_nframes]
            for x in np.arange(4)]

        peaks = np.argmax(clust_means, axis=1)
        pre_odor_clusts = utils.in_ranges(pre_odor, peaks)

        ### Define clusters where the response during the odor
        ### is independent of the trial type.
        ### Potentially you could do this more wisely,
        ### i.e. using an anova across trials
        type_sums = np.vstack([np.max(clust_means[:, r[0]:r[1]], axis=1)
                               for r in post_odor_onset]).T
        type_sums = type_sums / np.sum(type_sums, axis=1)[:, np.newaxis]
        # indep_cutoff = 0.6
        indep_cutoff = 0.7
        spout_independent = np.logical_and(
            np.sum(type_sums < indep_cutoff, axis=1) == 4,
            utils.in_ranges(during_odor, peaks))  ### In original space

        spout1 = np.logical_and(utils.in_ranges([post_odor_onset[0]], peaks),
                                ~spout_independent)
        spout2 = np.logical_and(utils.in_ranges([post_odor_onset[1]], peaks),
                                ~spout_independent)
        spout3 = np.logical_and(utils.in_ranges([post_odor_onset[2]], peaks),
                                ~spout_independent)
        nogo = np.logical_and(utils.in_ranges([post_odor_onset[3]], peaks),
                              ~spout_independent)

        ### Now assign each cluster from the nmf to one of these clusters
        super_clustering = np.zeros(clust_means.shape[0]).astype(int)
        super_clustering[pre_odor_clusts] = 0
        super_clustering[spout_independent] = 1
        super_clustering[spout1] = 2
        super_clustering[spout2] = 3
        super_clustering[spout3] = 4
        super_clustering[nogo] = 5

        super_cluster_titles = ['pre-odor', 'spout-independent', 'spout1',
                                'spout2', 'spout3', 'nogo']


    elif protocol == 'oeg2way':
        pass
    else:
        raise('Protocol {} not yet implemented in generate_superclusters().'.format(protocol))

    super_clust_info = {'super_clustering':super_clustering,
                        'titles': super_cluster_titles}

    return super_clust_info


def order_super_clusters(super_clustering, clust_means, method='peak',
                         do_plot=True, title_str=None, tranges=None,
                         clust_means_to_plot=None):
    """
    Each super cluster includes a set of clusters.
    This function orders the clusters within each super cluster,
    i.e. based on the peak time of the cluster mean trace.
    :param super_clustering: [nclusters]. Assignment of each
                              cluster to one of the superclusters.
    :param clust_means: [nclust x ntime] The time series of each cluster.
    :param method: str. 'peak' is the only one implemented now.
    :param do_plot: bool. Set to True to plot the ordered clusters.
    :param tranges: dict. For each super cluster, an np.array of frames.
               If not None, then only computes ordering based off
               the signal in the provided range of frames.

    :return clust_ord: [nclusters] The new cluster ordering, that is
                       now ordered within each supercluster. Each
                       entry indicates which cluster is in that
                       position.
    """
    if clust_means_to_plot is None:
        clust_means_to_plot = clust_means

    clust_ord = []
    for i in np.unique(super_clustering):
        if tranges is not None:
            trange = tranges[i]
        else:
            trange = np.arange(clust_means.shape[1])
        clusts = np.where(super_clustering == i)[0]
        # max_sort = np.argsort(np.argmax(clust_means[clusts, :], axis=1))
        max_sort = np.argsort(np.argmax(clust_means[clusts, :][:, trange], axis=1))
        clust_ord.append(clusts[max_sort])

    clust_ord = np.hstack(clust_ord)

    if do_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(clust_means_to_plot[clust_ord, :], aspect='auto')
        if title_str is not None:
            plt.title(title_str)

    return clust_ord


def order_sources_by_clust(clustering,
                           super_clustering,
                           clust_ordering):
    """
    Reassign cluster labels to sources, so that they now represent
    a provided ordering of clusters.
    I.e. if what was originally cluster #1, when ordered should now
    be cluster #5, propagate this change to all sources in cluster #1.

    :param clustering: [ncells]. Assignment of each cell to a cluster.
    :param super_clustering: [nclusters]. Assignment of each clusters to a super cluster.
    :param clust_ordering: [nclusters]. Assignment of ordered position of each cluster.
                                  If clust_ordering[5] = 10, then that means cluster #10
                                  should be in position 5.
                                  This can be the output of order_super_clusters(), if
                                  you want to order by super cluster,
                                  or order_clusters(), if you just want to order
                                  by the simple clusters.

    :return ordered_clustering: [ncells]. For each cell, the ordered position of its cluster.
    :return ordered_super_clustering: [ncells] For each cell, the position of its supercluster.
    """

    ### Assign each cell a new position based on its cluster and the
    ### ordered position of that cluster.
    ordered_clustering = np.zeros(clustering.shape).astype(int)
    for i in range(len(clustering)):
        cluster_of_source = clustering[i]
        ordered_clustering[i] = np.where(clust_ordering == cluster_of_source)[0]

    if super_clustering is not None:
        ### For each source, assign it to a super_cluster.
        ordered_super_clustering = np.zeros(ordered_clustering.shape).astype(int)
        for i in range(len(clustering)):
            ordered_super_clustering[i] = super_clustering[clustering[i]]
    else:
        ordered_super_clustering = None

    return ordered_clustering, ordered_super_clustering


def get_cluster_index_ranges(ordered_clustering):
    """
    Get a list of start and end indices corresponding
    to each cluster when the sources are ordered
    according to the provided ordering.

    :param ordered_clustering: [nsources] The assignment
                               of each source to a cluster,
                               where the clusters have been
                               ordered (i.e. see order_sources_by_clust()).

    :return clust_inds: [nclusts + 1]. A list of indices.
                         To obtain the the sources in the first cluster,
                         index as:
                         >>np.argsort(ordered_clustering)[clust_inds[0]:clust_inds[1]],
                         and for cluster c, index as:
                         >>np.argsort(ordered_clustering)[clust_inds[c]:clust_inds[c+1]],

    """

    clust_inds = np.where(np.diff(np.sort(ordered_clustering)))[0] + 1
    clust_inds = np.insert(clust_inds, 0, 0)
    clust_inds = np.append(clust_inds, len(ordered_clustering))

    return clust_inds

def get_color_template():
    color_template = [[0.11, 0.11, 0.11, 1.0],
                      [250./255, 165./255, 26./255, 1.0],
                      [30./255, 188./255, 189./255, 1.0],
                      [237./255, 34./255, 36./255, 1.0],
                      [25./255, 129./255, 64./255, 1.0],
                      [0, 0, 1.0, 1.0]]

    # color_template = [[1.0, 0.2, 0, 1.0],
    #                   [1.0, 0, 1.0, 1.0],
    #                   [1.0, 0.5, 0.2, 1.0],
    #                   [1.0, 0, 0.3, 1.0],
    #                   [0.0, 1.0, 0.5, 1.0],
    #                   [0, 0, 1.0, 1.0],
    #                   ]  ### Can change this is if you want custom colors,
    #                     ### May need to increase it if you have more colors...

    return color_template

def assign_colors_to_sources(ordered_clustering,
                             ordered_super_clustering,
                             cmap='jet',
                             same_within_super_cluster=False,
                             set_to_gray=None,
                             specify_discrete_colors=False):
    """
    Assign a color to each source based on the cluster
    and super-cluster it is a member of.

    :param ordered_clustering: [ncells]. For each cell, the ordered position of its cluster.
                               Output of order_sources_by_clust().
    :param ordered_super_clustering: [ncells] For each cell, the position of its supercluster.
                                     Output of order_sources_by_clust()
    :param cmap: Colormap used for assigning color to each cluster.
    :param same_within_super_cluster: Assign same color to each member of a super-cluster
    :param set_to_gray: None, or a list of super_cluster_indices that should
                        set their color to a transparent gray.

    :return source_coloring: [ncells x 4] RGBA value for each source.
    """

    ### Make clusters within the same super cluster have
    ### similar color values.
    clust_labels = np.sort(ordered_clustering)
    if ordered_super_clustering is None:
        ordered_clust_labels = np.expand_dims(clust_labels, axis=1)
    else:
        super_clust_labels = np.sort(ordered_super_clustering)
        ordered_clust_labels = np.expand_dims(super_clust_labels * 50
                                              + 4.001 * clust_labels, axis=1)
        if same_within_super_cluster:
            ordered_clust_labels = super_clust_labels
            
            
    if specify_discrete_colors:
        color_template = get_color_template()
        # color_template = [[1.0, 0.2, 0, 1.0],
        #                   [1.0, 0, 1.0, 1.0],
        #                   [1.0, 0.5, 0.2, 1.0],
        #                   [1.0, 0, 0.3, 1.0],
        #                   [0.0, 1.0, 0.5, 1.0],
        #                   [0, 0, 1.0, 1.0],
        #                   ] ### Can change this is if you want custom colors,
        #                     ### May need to increase it if you have more colors...

        if set_to_gray is not None:
            for k in set_to_gray:
                color_template[k] = [0.8, 0.8, 0.8, 0.5]

        colors = [color_template[k] for k in ordered_clust_labels]
    else:
        clusts = np.unique(ordered_clust_labels)
        plt.figure(figsize=(0.5, 0.5))
        f = plt.imshow(np.unique(ordered_clust_labels)[:, np.newaxis],
                       cmap=cmap,
                       aspect='auto')
        plt.axis('off')
        colors = [f.cmap(f.norm(c)) for c in clusts]


    sorted_source_coloring = np.zeros((ordered_clust_labels.shape[0], 1, 4))
    for i, clust in enumerate(clust_labels):
        clust_ind = np.where(clust == np.unique(clust_labels))[0]
        sorted_source_coloring[i, 0, :] = np.array(colors[int(clust_ind)][:])

    ### Reorganize the colors so that they match the global ordering of sources
    ### (i.e. the one loaded up in CosmosTraces).
    source_coloring = np.zeros(sorted_source_coloring.shape)
    for source in range(source_coloring.shape[0]):
        sorted_ind = np.where(np.argsort(ordered_clustering) == source)[0]
        source_coloring[source, :, :] = sorted_source_coloring[sorted_ind, :, :]

    return source_coloring

#
# def assign_colors_to_sources_orig(ordered_clustering,
#                              ordered_super_clustering,
#                              cmap='jet',
#                              same_within_super_cluster=False):
#     """
#     Assign a color to each source based on the cluster
#     and super-cluster it is a member of.
#
#     :param ordered_clustering: [ncells]. For each cell, the ordered position of its cluster.
#                                Output of order_sources_by_clust().
#     :param ordered_super_clustering: [ncells] For each cell, the position of its supercluster.
#                                      Output of order_sources_by_clust()
#     :param cmap: Colormap used for assigning color to each cluster.
#     :param same_within_super_cluster: Assign same color to each member of a super-cluster
#     :param set_to_gray: None, or a list of super_cluster_indices that should
#                         set their color to a transparent gray.
#
#     :return source_coloring: [ncells x 4] RGBA value for each source.
#     """
#
#     ### Make clusters within the same super cluster have
#     ### similar color values.
#     clust_labels = np.sort(ordered_clustering)
#     if ordered_super_clustering is None:
#         ordered_clust_labels = np.expand_dims(clust_labels, axis=1)
#     else:
#         super_clust_labels = np.sort(ordered_super_clustering)
#         ordered_clust_labels = np.expand_dims(super_clust_labels * 50
#                                               + 4 * clust_labels, axis=1)
#         if same_within_super_cluster:
#             ordered_clust_labels = np.expand_dims(super_clust_labels * 50
#                                                   + 0.001 * clust_labels, axis=1)
#
#
#     clusts = np.unique(ordered_clust_labels)
#     plt.figure(figsize=(0.5, 0.5))
#     f = plt.imshow(np.unique(ordered_clust_labels)[:, np.newaxis],
#                    cmap=cmap,
#                    aspect='auto')
#     plt.axis('off')
#     colors = [f.cmap(f.norm(c)) for c in clusts]
#
#     sorted_source_coloring = np.zeros((ordered_clust_labels.shape[0], 1, 4))
#     for i, clust in enumerate(clust_labels):
#         clust_ind = np.where(clust == np.unique(clust_labels))[0]
#         sorted_source_coloring[i, 0, :] = np.array(colors[int(clust_ind)][:])
#
#     ### Reorganize the colors so that they match the global ordering of sources
#     ### (i.e. the one loaded up in CosmosTraces).
#     source_coloring = np.zeros(sorted_source_coloring.shape)
#     for source in range(source_coloring.shape[0]):
#         sorted_ind = np.where(np.argsort(ordered_clustering) == source)[0]
#         source_coloring[source, :, :] = sorted_source_coloring[sorted_ind, :, :]
#
#     return source_coloring


def cluster_snn(data, k=3):
    """
    Shared nearest neighbor clustering (from Will).
    Takes as input a dimensionality reduced form of the data.
    :param data: [ncells x ntime (or nprincipal_components)]
    :param k:
    :return:
    """
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(data)
    neighbor_graph = nbrs.kneighbors_graph(data)
    g = ig.Graph()
    g = ig.GraphBase.Adjacency(neighbor_graph.toarray().tolist(), mode=ig.ADJ_UNDIRECTED)
    sim = np.array(g.similarity_jaccard())
    g = ig.GraphBase.Weighted_Adjacency(sim.tolist(), mode=ig.ADJ_UNDIRECTED)
    return np.array(g.community_multilevel(weights="weight", return_levels=False))


def get_cluster_averages(data, source_clustering, do_zscore=True,
                         do_plot=False):
    """
    Return the average trace across sources in each cluster.

    :param data: [ncells x ntime]
    :param source_clustering: [ncells] Assigns a cluster to each source.
                              i.e. ordered_super_clustering or
                                   ordered_clustering, the outputs from
                                   order_sources_by_clust().
    :param do_plot: bool
    """

    clust_inds = get_cluster_index_ranges(source_clustering)
    ordered_source_means = data[np.argsort(source_clustering), :]

    if do_zscore:
        M = scipy.stats.zscore(ordered_source_means, axis=1)
    else:
        M = ordered_source_means

    nclusts = len(clust_inds) - 1
    clust_avgs = np.zeros((nclusts, M.shape[1]))
    for c in range(nclusts):
        clust_avgs[c, :] = np.median(M[clust_inds[c]:clust_inds[c + 1], :],
                                     axis=0)

    if do_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(scipy.stats.zscore(clust_avgs, axis=1), aspect='auto',
                   clim=[-1, 5])
        plt.title('super clust avgs dataset')

    return clust_avgs


def plot_cluster_averages(data, ordered_clustering, ordered_super_clustering,
                          trial_set_inds, by_trial_type=True,
                          event_frames=None,
                          event_labels=None,
                          time_labels=None,
                          vertical_lines=None):
    """
    Plot average trace for each cluster for a specified set of trials.

    :param data: [ncells x time]
    :param ordered_clustering: [ncells] Assigns a cluster to each source.
                                   The outputs from order_sources_by_clust().
    :param ordered_super_clustering: [ncells] Assigns a supercluster to each source.
    :param trial_set_inds: [ntrialtypes] The start index for each trial type
                                         if data consists of multiple trial types
                                         concatenated together.
    :param by_trial_type: bool. If true, then each trial type has a separate plot,
                                with all cluster traces plotted for each.
                                If false, then each cluster has a separate plot,
                                with all trial types plotted for each.
    :param event_frames: list of frame indices at which to draw vertical lines.
    """
    # import pdb
    # pdb.set_trace()
    clust_avgs = get_cluster_averages(data, ordered_super_clustering,
                                      do_zscore=True, do_plot=False)
    clust_avgs = scipy.stats.zscore(clust_avgs, axis=1)
    # clust_avgs = scipy.signal.savgol_filter(clust_avgs, 11, 3)

    source_coloring = assign_colors_to_sources(ordered_clustering,
                                               ordered_super_clustering,
                                               cmap='jet')
    super_clust_colors = get_super_clust_colors(ordered_super_clustering,
                                                source_coloring)


    if by_trial_type:
        fig = plt.figure(figsize=(20, 3))
        for c in range(clust_avgs.shape[0]):
            plt.plot(clust_avgs[c,:],
                    color=super_clust_colors[c, :],
                    linewidth=1)
        if vertical_lines is not None:
            if type(vertical_lines) is dict:
                for key, val in vertical_lines.items():
                    if key == 'k':
                        [plt.axvline(x, color=key, linewidth=0.5) for x in val]
                    else:
                        [plt.axvline(x, color=key, linewidth=0.5,
                                     linestyle=(0, (1, 1))) for x in val]
            else:
                [plt.axvline(x, color='r', linewidth=0.5) for x in vertical_lines]
        plt.xlim([0, clust_avgs.shape[1]])
        if time_labels is not None:
            plt.gca().set_xticks(time_labels['positions'])
            plt.gca().set_xticklabels(time_labels['labels'])
            plt.gca().set_xlabel('time [s]')
        plt.gca().set_ylabel('z-score')
        for axis in ['top', 'bottom', 'left', 'right']:
            plt.gca().spines[axis].set_linewidth(0.5)

    # nt = int(np.mean(np.diff(trial_set_inds)))
    # if by_trial_type:
    #     # Separate subplot for each trial type.
    #     fig = plt.figure(figsize=(20, 3))
    #     for i in range(4):
    #         plt.subplot(1, len(trial_set_inds), i + 1)
    #         inds = np.arange(2, nt-5)
    #         for c in range(clust_avgs.shape[0]):
    #             plt.plot(
    #                 inds,
    #                 clust_avgs[c, inds+trial_set_inds[i]].T,
    #                 color=super_clust_colors[c, :],
    #                 linewidth=1)
    #         #   plt.title(trial_names[i])
    #         plt.ylim([-2.5, 5])
    #         fig.subplots_adjust(hspace=0, wspace=0.04)
    #         if i > 0:
    #             plt.yticks([])
    #         plt.xlim([0, nt])
    #         if event_frames is not None:
    #             [plt.axvline(x, linewidth=0.5, linestyle=(0, (1, 1)),
    #                          color='k') for x in event_frames]
    #         if event_labels is not None:
    #             plt.xticks(event_frames, event_labels)
    # else:
    #     # Separate subplot for each cluster.
    #     plt.figure(figsize=(10, 2))
    #     nclusts = clust_avgs.shape[0]
    #     for clust in range(nclusts):
    #         plt.subplot(1, nclusts, clust + 1)
    #         for trialtype in range(len(trial_set_inds)):
    #             plt.plot(clust_avgs[clust, trialtype * nt:(trialtype + 1) * nt])
    #         if event_frames is not None:
    #             [plt.axvline(x) for x in event_frames]
    #         #     plt.title(super_clust_titles[clust])


def plot_stacked_group_composition(group_composition,
                                   subgroup_colors,
                                   res=1000, transpose=True):
    """
    Display a stacked bar chart representing how groups
    are composed of shared sub-groups.

    :param group_composition: [nsubgroups, ngroups]
    :param subgroup_colors: [nsubgroups x 4] The RGBA color for each subgroup
                         in subgroups.
                         i.e. output from get_super_clust_colors()
    """

    im = np.zeros((group_composition.shape[1], res, 4))

    subgroup_indices = np.cumsum(group_composition, axis=0)
    subgroup_indices = np.vstack(
        (np.zeros(subgroup_indices.shape[1]), subgroup_indices))
    subgroup_indices /= np.max(subgroup_indices)
    subgroup_indices *= res
    subgroup_indices = subgroup_indices.astype(int)

    for group in range(group_composition.shape[1]):
        for subgroup in range(group_composition.shape[0]):
            inds = np.arange(subgroup_indices[subgroup, group],
                             subgroup_indices[subgroup + 1, group])
            im[group, inds, :] = subgroup_colors[subgroup, :]

    if not transpose:
        im = np.swapaxes(im, 0, 1)
    plt.imshow(im, aspect='auto', extent=[0, 1, im.shape[0], 0])


def sort_rows_greedily(d):
    """
    Sort the ordering of the rows
    of data by successively selecting
    the next most similar row (that
    has not yet been selected).

    :param data: [nrows x ncols] To be sorted.
    :return sorted_data: [nrows x ncols] The now-sorted input matrix.
    :return sorted_inds: [nrows] The indices for sorting.
    """

    d = np.copy(d)
    nrows = d.shape[0]
    sorted_inds = np.zeros(nrows)
    sorted_data = np.zeros(d.shape)
    unsorted_data = np.copy(d)
    unsorted_inds = np.arange(nrows)

    for i in range(nrows):
        if i == 0:
            which_row = np.argmax(d[:, 0])
        else:
            ### Select the closest row
            prev_row = sorted_data[i - 1, :]
            dot_prod = np.matmul(unsorted_data, prev_row)
            which_row = np.argmax(dot_prod)

        sorted_data[i, :] = unsorted_data[which_row, :]
        sorted_inds[i] = unsorted_inds[which_row]
        unsorted_inds = np.delete(unsorted_inds, which_row)
        unsorted_data = np.delete(unsorted_data, which_row, 0)

    return sorted_data, sorted_inds.astype(int)


def get_between_cluster_correlations(clustering, corr_matrix):
    """
    Compute the mean correlation coefficient of sources within
    a cluster (to each other) vs. with sources in each of
    the clusters.

    :param clustering: [ncells] A cluster assignment for each source.
    :param corr_matrix: [ncells x ncells] Correlation matrix between
                        each source.
    :return: clust_corr: [nclusts x nclusts] a matrix with the
                         mean correlation coefficient between
                         sources in each cluster.
    """

    nclusts = np.max(clustering) + 1
    clust_corrs = np.zeros((nclusts, nclusts))
    for base_c in np.unique(clustering):
        sources_in_base = np.where(clustering == base_c)[0]
        for target_c in np.unique(clustering):
            sources_in_target = np.where(clustering == target_c)[0]

            corrs = []
            for s in sources_in_base:
                for t in sources_in_target:
                    if s != t:  # Exclude self-comparisons
                        corrs.append(corr_matrix[s, t])

            clust_corrs[base_c, target_c] = np.mean(np.array(corrs))

    return clust_corrs


def corr_structure_across_time(data, window_radius=500, do_plot=False):
    """
    Analyze how the correlation structure of the data
    changes across time.
    Specifically, compute a correlation matrix using
    temporally windowed subsets of the data.
    Then compute the correlation between those correlation
    matrices.
    The result is a matrix that represents the similarity in
    correlation structure between various timepoints.

    :param data: [ncells x nt]
    :param window_radius: int. Each time window has twice this number
                          of timepoints.
    :return corr_corr: [nwindows x nwindows] the correlation
                       matrix of temporally windowed
                       correlation matrices.
    :return corr_vals: [nwindows x noff_diag]
                       the off-diagonal values of each
                       temporally windowed correlation matrix.
    """
    wd, w_ind = utils.get_windowed_data(data, window=window_radius)

    corr_vals = []  # The off-diagonal terms of each windowed corr matrix.
    allC = []
    for i in range(len(wd)):
        # Compute correlation of the time window.
        c = np.corrcoef(wd[i])
        allC.append(c)

        # Extract the upper off-diagonal components.
        d = np.triu(c, k=1)
        e = d[d != 0]
        e[np.where(np.isnan(e))[0]] = 0
        corr_vals.append(e)

    corr_vals = np.vstack(corr_vals)
    if do_plot:
        plt.figure()
        plt.imshow(corr_vals, aspect='auto')
        plt.title('off-diagonal values of each windowed corr matrix')
        plt.xlabel('off-diagonal index')
        plt.ylabel('time window')

        plt.figure()
        plt.plot(np.mean(corr_vals, axis=1))
        plt.title('mean correlation within each time window')
        plt.xlabel('time window')

    corr_corr = np.corrcoef(corr_vals)
    return corr_corr, corr_vals


def rand_circshift(data, rand_seed=1):
    """
    Circshift each row of data by a randomly chosen
    shift for each row.


    :param data: [ncells x ntime]
    :param rand_seed: int. seed of the permutation.
    :return: shifted_data.
    """

    shifts = np.random.random_integers(0, data.shape[1], size=data.shape[0])
    shifted_data = np.copy(data)

    for ind, shift in enumerate(shifts):
        shifted_data[ind, :] = np.roll(shifted_data[ind, :], shift)

    return shifted_data


def compare_in_vs_next_best_cluster(clustering, corr_matrix, do_plot=False, do_debug=False):

    """
    Compare the mean correlation coefficient of sources
    within a cluster (to each other)
    vs with that of the cluster is that is most correlated
    of all of the other clusters.

    :param clustering: [ncells] A cluster assignment for each source.
    :param corr_matrix: [ncells x ncells] Correlation matrix between
                        each source.
    :return: diffs:  [nclusters] For each cluster, the difference between
                     the correlation between sources within that cluster
                     and the correlation with sources in the cluster of the
                     other clusters that has the highest mean correlation with
                     the target cluster.
    """
    clust_corrs = get_between_cluster_correlations(clustering, corr_matrix)

    if do_debug:
        plt.imshow(clust_corrs)

        for i in range(clust_corrs.shape[0]):
            plt.figure()
            plt.plot(clust_corrs[:, i], 'k-')
            plt.plot(i, clust_corrs[i, i], 'ro')
            plt.title(i)

    diffs = []
    nclusts = clust_corrs.shape[0]
    for i in range(nclusts):
        inds = np.arange(nclusts)!=i
        d = clust_corrs[i, i] - np.nanmax(clust_corrs[i, inds])
        diffs.append(d)
    diffs = np.array(diffs)
    # diffs[np.isnan(diffs)] = 0

    if do_plot:
        diffs_p = np.delete(diffs, np.where(np.isnan(diffs))[0])

        _, p = scipy.stats.ttest_1samp(diffs_p, 0)
        _, pw = scipy.stats.wilcoxon(diffs_p)
        plt.hist(diffs_p, bins=20)
        plt.title('ttest_1samp p: {:.3e}, wilcoxon: {:.3e}'.format(p, pw))
        plt.xlabel('Difference in correlation')
        plt.ylabel('Cluster count')
    return diffs

def compare_in_vs_out_of_cluster(clustering, corr_matrix, do_plot=False,
                                 do_remove_nan=True):
    """
    Compare the mean correlation coefficient of sources
    within a cluster (to each other)
    vs with sources outside of that cluster.

    :param clustering: [ncells] A cluster assignment for each source.
    :param corr_matrix: [ncells x ncells] Correlation matrix between
                        each source.
    :param do_plot: bool. If true, plot the correlation between clusters.

    :return in_clust: [nclusts] For each cluster, mean correlation between
                                sources within that cluster.
            out_clust: [nclusts*2 - nclusts] For each pair of different clusters,
                                the mean correlation between sources
                                in the pairs of clusters.

    """

    # nclusts = np.max(clustering) + 1
    # clust_corrs = np.zeros((nclusts, nclusts))
    # for base_c in np.unique(clustering):
    #     sources_in_base = np.where(clustering == base_c)[0]
    #     for target_c in np.unique(clustering):
    #         sources_in_target = np.where(clustering == target_c)[0]
    #
    #         corrs = []
    #         for s in sources_in_base:
    #             for t in sources_in_target:
    #                 if s != t:  # Exclude self-comparisons
    #                     corrs.append(corr_matrix[s, t])
    #
    #         clust_corrs[base_c, target_c] = np.mean(np.array(corrs))

    clust_corrs = get_between_cluster_correlations(clustering, corr_matrix)

    if do_plot:
        plt.imshow(clust_corrs)
        plt.title('Correlation of sources between clusters')

    ### Extract diagonal components (i.e. correlations within clusters)
    in_clust = np.diagonal(clust_corrs)
    if do_remove_nan:
        in_clust = np.delete(in_clust, np.where(np.isnan(in_clust))[0])

    ### Extract upper off-diagonal components
    ### (i.e. correlations outside clusters).
    w = np.where(np.triu(clust_corrs, k=1) != 0)
    out_clust = clust_corrs[w[0], w[1]]
    if do_remove_nan:
        out_clust = np.delete(out_clust, np.where(np.isnan(out_clust))[0])

    return in_clust, out_clust


def plot_in_vs_out_of_cluster_comparison(in_clust, out_clust):
    """
    Plot comparison of correlation within clusters vs
    to other clusters.
    Uses output of compare_in_vs_out_of_cluster().
    """
    plt.hist(in_clust, density=True, color=(1.0, 0, 0, 0.5))
    # plt.figure()
    plt.hist(out_clust, density=True, color=(0, 0, 1.0, 0.5))
    h, p = scipy.stats.kruskal(in_clust, out_clust)
    plt.title('kruskal pvalue {:.2e}'.format(p))
    plt.xlabel('Mean correlation between clusters')
    plt.ylabel('Probability')


def plot_clustered_sources_cross_validate(
                           ordered_clustering,
                           ordered_super_clustering,
                           first_source_means,
                           second_source_means,
                           source_coloring,
                           cmap='rocket',
                           clim=[-1, 6],
                           vertical_lines=None,
                           title_str=None,
                           labels=None,
                           time_labels=None,
                           exclude_super_clusters=None):
    """
      Plot the mean traces of all sources/cells,
      ordered by cluster.
      For each cluster, order the sources based on the peak
      time in the average trace, using half of the dataset,
      and then plot the average trace of the other half
      of the dataset.

      Uses the output of order_sources_by_clust().

      :param ordered_clustering: [ncells]. For each cell, the ordered position of its cluster.
      :param ordered_super_clustering: [ncells] For each cell, the position of its supercluster.
      :param first_source_means: [ncells x ntime]. The trace corresponding to each source/cell
                                             potentially concatenated across trial types averaged
                                             across the first half of the dataset.
      :param second_source_means: [ncells x ntime]. The trace corresponding to each source/cell
                                         potentially concatenated across trial types averaged
                                          across the second half of the dataset.
      :param source_coloring: [ncells x 4] RGBA value for each source.
      :param cmap: str. Colormap name for the trace plots.
      :param vertical_lines: list of frame indices for plotting vertical lines over the traces.
      :param title_str: Optional title for the plot.
      :param labels: a dict containing 'labels' and 'positions' for plotting
                     labels of each trial type column.
      :param time_labels: a dict containing 'labels' and 'positions' for plotting
                     the time ticks along the bottom.
      :param exclude_super_clusters: A list of superclusters
                                    (using the 'ordered_super_clustering' ids)
                                    to not show in the plot. Set to None
                                    to ignore.
      """

    clust_end = np.where(np.diff(np.sort(ordered_super_clustering)))[0]
    clust_end = np.insert(clust_end, 0, 0)
    # clust_end = np.insert(clust_end, -1, len(ordered_super_clustering))

    nsources = source_means.shape[0]

    ### Generate plot layout
    fig = plt.figure(figsize=(10, 20))
    gs = []
    gs.append(plt.subplot2grid((50, 50), (0, 0), colspan=3, rowspan=47))
    gs.append(plt.subplot2grid((50, 50), (0, 3), colspan=47, rowspan=47))

    ### Plot cluster labels
    plt.subplot(gs[0])

    if exclude_super_clusters is not None:
        n_to_exclude = 0
        for c in exclude_super_clusters:
            ordered_clustering = np.copy(ordered_clustering)
            print(c)
            ordered_clustering[np.where(ordered_super_clustering == c)[0]] = 1e8
            n_to_exclude += len(np.where(ordered_super_clustering == c)[0])
            print(n_to_exclude)
    plt.imshow(source_coloring[np.argsort(ordered_clustering), :, :], aspect='auto')

    gs[0].get_xaxis().set_visible(False)
    #     plt.yticks(label_ind, region_str)
    gs[0].set_yticks([])
    gs[0].tick_params(length=0)
    # plt.box(on=None)
    for axis in ['top', 'bottom', 'left', 'right']:
        gs[0].spines[axis].set_linewidth(0.5)
    # gs[0].spines['left'].set_visible(False)
    if exclude_super_clusters:
        gs[0].set_ylim([nsources - n_to_exclude, 0])
        [plt.axhline(x - n_to_exclude, color='k', linewidth=0.5) for x in clust_end]

    else:
        [plt.axhline(x, color='k', linewidth=0.5) for x in clust_end]

    # Now, plot.
    plt.subplot(gs[1])
    ordered_source_means = source_means[np.argsort(ordered_clustering), :]
    Z = scipy.stats.zscore(ordered_source_means, axis=1)
    plt.imshow(Z, clim=clim, aspect='auto', cmap=cmap)
    # plt.imshow(Z, clim=[-2, 3], aspect='auto', cmap='bwr')
    if vertical_lines is not None:
        if type(vertical_lines) is dict:
            for key, val in vertical_lines.items():
                if key == 'k':
                    [plt.axvline(x, color=key, linewidth=0.5) for x in val]
                else:
                    [plt.axvline(x, color=key, linewidth=0.5, linestyle=(0, (1, 1))) for x in val]
        else:
            [plt.axvline(x, color='r', linewidth=0.5) for x in vertical_lines]
    # plt.axvline(Z.shape[1], color='k', linewidth=0.5)

    plt.yticks(np.arange(0, nsources, 500))
    # plt.yticks([0, 500, 1000])
    axT = gs[1].twiny()
    axT.set_xlim(gs[1].get_xlim())
    for axis in ['top', 'bottom', 'left', 'right']:
        gs[1].spines[axis].set_linewidth(0.5)
        axT.spines[axis].set_linewidth(0.5)
    gs[1].yaxis.set_label_position("right")
    gs[1].yaxis.tick_right()
    if exclude_super_clusters:
        gs[1].set_ylim([nsources - n_to_exclude, 0])
        [plt.axhline(x - n_to_exclude, color='k', linewidth=0.5) for x in clust_end]
    else:
        [plt.axhline(x, color='k', linewidth=0.5) for x in clust_end]

        # gs[1].set_xticks()

    if labels is not None:
        axT.set_xticks(labels['positions'])
        axT.set_xticklabels(labels['labels'])
        axT.xaxis.set_ticks_position('none')

    if time_labels is not None:
        gs[1].set_xticks(time_labels['positions'])
        gs[1].set_xticklabels(time_labels['labels'])
        gs[1].set_xlabel('time [s]')

    gs[1].set_ylabel('sources')

    # plt.box(on=None)
    fig.subplots_adjust(hspace=0, wspace=0)
    if title_str is not None:
        plt.suptitle(title_str)


def plot_clustered_sources(ordered_clustering,
                           ordered_super_clustering,
                           source_means,
                           source_coloring,
                           cmap='rocket',
                           clim=[-1, 6],
                           vertical_lines=None,
                           title_str=None,
                           labels=None,
                           time_labels=None,
                           exclude_super_clusters=None):
    """
    Plot the mean traces of all sources/cells,
    ordered by cluster.
    Uses the output of order_sources_by_clust().

    :param ordered_clustering: [ncells]. For each cell, the ordered position of its cluster.
    :param ordered_super_clustering: [ncells] For each cell, the position of its supercluster.
    :param source_means: [ncells x ntime]. The trace corresponding to each source/cell
                                           potentially concatenated across trial types.
    :param source_coloring: [ncells x 4] RGBA value for each source.
    :param cmap: str. Colormap name for the trace plots.
    :param vertical_lines: list of frame indices for plotting vertical lines over the traces.
    :param title_str: Optional title for the plot.
    :param labels: a dict containing 'labels' and 'positions' for plotting
                   labels of each trial type column.
    :param time_labels: a dict containing 'labels' and 'positions' for plotting
                   the time ticks along the bottom.
    :param exclude_super_clusters: A list of superclusters
                                  (using the 'ordered_super_clustering' ids)
                                  to not show in the plot. Set to None
                                  to ignore.
    """

    clust_end = np.where(np.diff(np.sort(ordered_super_clustering)))[0]
    clust_end = np.insert(clust_end, 0, 0)
    # clust_end = np.insert(clust_end, -1, len(ordered_super_clustering))

    nsources = source_means.shape[0]

    ### Generate plot layout
    fig = plt.figure(figsize=(10, 20))
    gs = []
    gs.append(plt.subplot2grid((50, 50), (0, 0), colspan=3, rowspan=47))
    gs.append(plt.subplot2grid((50, 50), (0, 3), colspan=47, rowspan=47))

    ### Plot cluster labels
    plt.subplot(gs[0])

    if exclude_super_clusters is not None:
        n_to_exclude = 0
        for c in exclude_super_clusters:
            ordered_clustering = np.copy(ordered_clustering)
            print(c)
            ordered_clustering[np.where(ordered_super_clustering==c)[0]] = 1e8
            n_to_exclude += len(np.where(ordered_super_clustering==c)[0])
            print(n_to_exclude)
    plt.imshow(source_coloring[np.argsort(ordered_clustering), :, :], aspect='auto')

    gs[0].get_xaxis().set_visible(False)
    #     plt.yticks(label_ind, region_str)
    gs[0].set_yticks([])
    gs[0].tick_params(length=0)
    # plt.box(on=None)
    for axis in ['top', 'bottom', 'left', 'right']:
        gs[0].spines[axis].set_linewidth(0.5)
    # gs[0].spines['left'].set_visible(False)
    if exclude_super_clusters:
        gs[0].set_ylim([nsources - n_to_exclude, 0])
        [plt.axhline(x-n_to_exclude, color='k', linewidth=0.5) for x in clust_end]

    else:
        [plt.axhline(x, color='k', linewidth=0.5) for x in clust_end]

    # Now, plot.
    plt.subplot(gs[1])
    ordered_source_means = source_means[np.argsort(ordered_clustering), :]
    Z = scipy.stats.zscore(ordered_source_means, axis=1)
    plt.imshow(Z, clim=clim, aspect='auto', cmap=cmap)
    # plt.imshow(Z, clim=[-2, 3], aspect='auto', cmap='bwr')
    if vertical_lines is not None:
        if type(vertical_lines) is dict:
            for key, val in vertical_lines.items():
                if key == 'k':
                    [plt.axvline(x, color=key, linewidth=0.5) for x in val]
                else:
                    [plt.axvline(x, color=key, linewidth=0.5, linestyle=(0, (1, 1))) for x in val]
        else:
            [plt.axvline(x, color='r', linewidth=0.5) for x in vertical_lines]
    # plt.axvline(Z.shape[1], color='k', linewidth=0.5)

    plt.yticks(np.arange(0, nsources, 500))
    # plt.yticks([0, 500, 1000])
    axT = gs[1].twiny()
    axT.set_xlim(gs[1].get_xlim())
    for axis in ['top', 'bottom', 'left', 'right']:
        gs[1].spines[axis].set_linewidth(0.5)
        axT.spines[axis].set_linewidth(0.5)
    gs[1].yaxis.set_label_position("right")
    gs[1].yaxis.tick_right()
    if exclude_super_clusters:
        gs[1].set_ylim([nsources - n_to_exclude, 0])
        [plt.axhline(x-n_to_exclude, color='k', linewidth=0.5) for x in clust_end]
    else:
        [plt.axhline(x, color='k', linewidth=0.5) for x in clust_end]

# gs[1].set_xticks()

    if labels is not None:
        axT.set_xticks(labels['positions'])
        axT.set_xticklabels(labels['labels'])
        axT.xaxis.set_ticks_position('none')

    if time_labels is not None:
        gs[1].set_xticks(time_labels['positions'])
        gs[1].set_xticklabels(time_labels['labels'])
        gs[1].set_xlabel('time [s]')

    gs[1].set_ylabel('sources')


    # plt.box(on=None)
    fig.subplots_adjust(hspace=0, wspace=0)
    if title_str is not None:
        plt.suptitle(title_str)


def get_super_clust_colors(ordered_super_clustering, centroid_coloring):
    """
    Return RGBA colors corresponding to
    each supercluster.
    :param ordered_super_clustering: [ncells]. Assignment of each source to a supercluster.
    :param centroid_coloring: [ncells x 4]. RGBA for assignment to each source.
                              i.e. output of assign_colors_to_sources().
    """
    super_clust_colors = np.zeros((len(np.unique(ordered_super_clustering)), 4))
    for ind, sc in enumerate(np.unique(ordered_super_clustering)):
        cellid = np.where(ordered_super_clustering==sc)[0][0]
        super_clust_colors[ind, :] = centroid_coloring[cellid, :]
    return super_clust_colors


def compare_cluster_memberships(clustering1,
                                clustering2,
                                do_sort=True,
                                exclude_super_clusters=None):
    """
    Examine how two different clusterings relate.
    Specifically, a cluster in clustering1
    is composed of many sources.
    How do those sources map onto the clusters in
    clustering2?
    For example, clustering1 can be the local clusters
    generated by clustering the full traces,
    whereas clustering2 is the superclusters generated
    by clustering the averaged traces. For each local cluster,
    are the different sources members of multiple
    superclusters, and if so, which ones?
    For each cluster in cluster1, the membership
    in the clusters of cluster2 are plotted.

    :param clustering1: [ncells]. Assignment of sources to
                                   the first clustering.
                                   i.e. ordered_clustering
    :param clustering2: [ncells]. Assignment of sources to
                                   the second clustering.
                                   i.e. ordered_super_clustering
    :param clust2_colors: [nclust2 x 4] The RGBA color for each cluster
                         in clustering2.
                         i.e. output from get_super_clust_colors()
    :param do_sort: bool. Sort clusters when plotting to put
                          similar clusters next to each other.
    :param exclude_super_clusters: A list of superclusters
                                  (using the 'ordered_super_clustering' ids)
                                  to not show in the plot. Set to None
                                  to ignore.

    :returns normed_overlap:[nclust2 x nclust1] For each cluster
                            in clustering1, the fraction of sources
                            that are in each cluster of clustering2.
             sorted_inds: [nclust1]. Indices used for plotting normed_overlap.
    """

    nclust1 = np.max(clustering1) + 1
    nclust2 = np.max(clustering2) + 1

    ### Ignore specified super clusters.
    which_super_clusters = np.arange(nclust2)
    if exclude_super_clusters is not None:
        for c in exclude_super_clusters:
            which_super_clusters = np.delete(which_super_clusters,
                                             np.where(which_super_clusters == c)[0])
        nclust2 -= len(exclude_super_clusters)

    overlap = np.zeros((nclust2, nclust1))
    total_in_clust1 = np.zeros((1, nclust1))

    for clust1 in range(nclust1):
        for ind2, clust2 in enumerate(which_super_clusters):
            in_clust1 = clustering1 == clust1
            in_clust2 = clustering2 == clust2
            in_both = np.logical_and(in_clust1, in_clust2)

            # overlap[clust2, clust1] = len(np.where(in_both)[0])
            overlap[ind2, clust1] = len(np.where(in_both)[0])
            # total_in_clust1[0, clust1] = len(np.where(in_clust1)[0])
            total_in_clust1[0, clust1] += len(np.where(in_both)[0])

    # import pdb; pdb.set_trace()
    # normed_overlap = overlap / np.tile(total_in_clust1, (nclust2, 1))  ### A Potential unit test: The columns(?) should each add up to 1
    normed_overlap = overlap / np.sum(overlap, axis=0)[np.newaxis, :]
    normed_overlap[np.isnan(normed_overlap)] = 0

    if do_sort:
        sort_normed_overlap, sorted_inds = sort_rows_greedily(normed_overlap.T)
        group_composition = sort_normed_overlap.T
    else:
        group_composition = normed_overlap
        sorted_inds = np.arange(group_composition.shape[1])

    # plt.figure(figsize=(5, 5))
    # plot_stacked_group_composition(group_composition, clust2_colors)
    # plt.ylabel('Cluster #')

    return normed_overlap, sorted_inds


def compute_cluster_sizes(clustering):
    """
    Count the number of sources in each cluster.

    :param clustering:
    :return: cluster_sizes
    """
    nclusters = np.max(clustering)+1
    cluster_sizes = np.zeros((nclusters, 1))
    for ccc in range(nclusters):
        cluster_sizes[ccc] = len(np.where(clustering==ccc)[0])

    return cluster_sizes

def compute_cluster_spread(clustering, centroids, do_split_hemispheres=False):
    """
    Compute median distance between the centroids of sources in
    a each cluster.
    :param clustering: [ncells] The cluster assignment of each source.
    :param centroids: [ncells x 2] The centroid coordinates of each source.

    :return median_cluster_spread: [nclusters], the spread of centroids in each cluster
    :return sem_cluster_spread: [nclusters], the variance in the centroid distances in each cluster
    """

    ### Compute spatial spread of each cluster.
    nclusters = np.max(clustering)+1
    median_cluster_spread = np.zeros((nclusters, 1))
    sem_cluster_spread = np.zeros((nclusters, 1))

    # for ind, ccc in enumerate(np.unique(clustering)):
    for ccc in range(np.max(clustering)+1):
        ind = ccc
        cluster_inds = np.where(clustering == ccc)[0]
        clust_centroid = np.mean(centroids[cluster_inds, :], axis=0)
        diffs = centroids[cluster_inds, :] - clust_centroid
        diffs_dist = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        median_cluster_spread[ind] = np.mean(diffs_dist)
        sem_cluster_spread[ind] = scipy.stats.sem(diffs_dist)

    return median_cluster_spread, sem_cluster_spread


def plot_single_cluster_spatial_map(which_clust,
                                    centroids_to_plot,
                                    clustering,
                                    super_clustering=None,
                                    bg_alpha=0.1,
                                    fig_size=[3,3],
                                    radius=10,
                                    ax=None):
    """
    Wrapper that enables plotting of centroids for a
    single cluster.

    :param which_clust: integer id of the cluster. (corresponding to 'clustering')
    :param centroids_to_plot: [ncells x 2] coordinate of each cell.
                              i.e. all_nmf['full']['centroid_atlas_coords']
    :param clustering: [ncells], assignment of each cell to a cluster.
                        i.e. all_nmf['full']['ordered_clustering']
    :param super_clustering: optional. This only is necessary for protocol='4way'
    :param bg_alpha: option. Alpha of the background centroids.
    :param fig_size: [width, height] in inches.
    :param radius: int. radius of the centroids in scatter plot.
    :return:
    """

    centroid_coloring = assign_colors_to_sources(clustering,
                                                 super_clustering,
                                                 cmap='hsv')

    plot_cluster_spatial_maps(centroid_coloring,
                                 clustering,
                                 centroids_to_plot,
                                 radius=radius,
                                 background_color=np.array([0.9, 0.9, 0.9, bg_alpha]),
                                 do_overlay=True,
                                 specific_clusters = [which_clust],
                                 ncols=1,
                                 ax=ax)

    if ax is None:
        plt.gcf().set_size_inches(w=fig_size[0], h=fig_size[1]) # Control size of figure in inches

def plot_cluster_spatial_maps(source_coloring,
                              clustering,
                              centroids_to_plot,
                              radius=10,
                              background_color=np.array([0.9, 0.9, 0.9, 0.9]),
                              subplot_titles=None,
                              do_overlay=False,
                              specific_clusters=None,
                              ncols=3,
                              do_titles=True,
                              ax=None):
    """
    Plot spatial map of the sources included in each cluster or supercluster,
    colored according to the provided source_coloring array.
    Plots each source at the center of mass of its ROI.

    :param source_coloring: [ncells x 4] RGBA value for each source.
    :param clustering: [ncells] Assignment of each cell to
                                a cluster or supercluster.
                                For example, can be one of the
                                outputs of order_sources_by_clust(),
                                such as 'ordered_super_clustering'.
    :param centroids_to_plot: [ncells x 2] The atlas-transformed centroids.
    :param atlas_coords: [ncells x 2] The centroid location of each source, transformed
                         to align to the atlas
                         (i.e. using CT.atlas_tform.inverse((centroids[cell, 1],
                                                             centroids[cell, 0]))[0]
    :param radius: int. radius of the circle to draw at each source location.
    :param background_color: the color for plotting background centroids that
                             are not the highlighted ones.
    :param subplot_titles: [nclusts] title strings for each subplot.
                              Set to None for no titles.
    :param do_overlay: bool. Overlay all clusters onto a single plot.
    :param specific_clusters: list of indices of clusters to include
                              in the plot. Corresponds to indices in
                              'clustering' variable.
    """

    cell_ids = np.arange(centroids_to_plot.shape[0])

    if specific_clusters is None:
        # specific_clusters = np.unique(clustering)
        specific_clusters = np.arange(np.max(clustering)+1)

    # print(specific_clusters)

    nclusts = len(specific_clusters)
    if not do_overlay:
        nrows = int(np.ceil(nclusts/ncols))
    else:
        nrows = 1

    H = 5
    W = 15
    if subplot_titles is None:
        W = 20

    if ax is None:
        fig = plt.figure(figsize=(W, nrows*H))
    for iter, i in enumerate(specific_clusters):
        if not do_overlay:
            plt.subplot(nrows, ncols, iter + 1)
            if do_titles:
                plt.title(i)

        RGBA = np.copy(np.squeeze(source_coloring))
        RGBA[clustering != i, :] = background_color

        if ax is not None:
            plt.sca(ax)
        utils.centroids_on_atlas(RGBA, cell_ids, centroids_to_plot, None,
                                 max_radius=radius,
                                 rotate90=True)
        if subplot_titles is not None:
            if do_titles:
                plt.title(subplot_titles[i])
    if ax is None:
        fig.subplots_adjust(hspace=0.1, wspace=0.1)


def total_num_per_category(categorization):
    """
    If 'categorization' is a list of labels
    for objects, this function counts how many times each
    label appears.

    :param categorization: [nobjects] Each entry
                          is the label (i.e. an int)
                          for that object.
    :returns num_per_category: for each of the possible
                               labels, how many times
                               does it show up.
    """

    labels = np.arange(1, np.max(categorization)+1)
    num_per_category = np.zeros(len(labels))
    for ind, label in enumerate(labels):
        num_per_category[ind] = len(np.where(categorization == label)[0])

    return num_per_category, labels


def plot_cluster_members_averages(rates,
                                  clustering,
                                  centroids_to_plot,
                                  trial_sets,
                                  do_by_trial_type=True,
                                  do_zscore=True,
                                  which_clusts=None,
                                  event_frames=None,
                                  do_sort=False):
    """
    Plot trial averaged response of sources in each cluster
    as well as spatial map of the centroids.
    Can either plot average across all trials, or separate
    by trial_type.

    :param rates:
    :param clustering:
    :param centroids_to_plot:
    :param trial_sets:
    :param do_by_trial_type:
    :param do_zscore:
    :param which_clust:
    :return:
    """

    if which_clusts is None:
        which_clusts = np.sort(np.unique(clustering))
    elif isinstance(which_clusts, int):
        which_clusts = [which_clusts]

    if do_by_trial_type:
        data = concatenate_trial_type_avgs(trial_sets,
                                           rates,
                                            do_plot=False)
    else:
        data = np.mean(rates, axis=2)

    if do_zscore:
        data = scipy.stats.zscore(data, axis=1)


    ### Compute the in vs. out correlation.
    rates_flat = np.reshape(rates,
                            (rates.shape[0],
                             rates.shape[1] * rates.shape[2]),
                             order='F')
    if do_zscore:
        rates_flat = zscore(rates_flat, axis=1)
    corr_matrix = np.corrcoef(rates_flat)
    inclust, outclust = compare_in_vs_out_of_cluster(clustering, corr_matrix,
                                                     do_plot=False,
                                                     do_remove_nan=False)

    plt.figure()
    for c in which_clusts:
        ### Plot the traces
        plt.figure(figsize=(9, 3))
        plt.subplot(1,2,1)
        traces_to_plot = data[clustering==c, :]
        if do_sort:
            peak_times = np.argmax(traces_to_plot, axis=1)
            traces_to_plot = traces_to_plot[np.argsort(peak_times), :]
        plt.imshow(traces_to_plot, aspect='auto')


        if event_frames is not None:
            if do_by_trial_type:
                for i in range(len(trial_sets)):
                    [plt.axvline(i*rates.shape[1]+f, color='r', linestyle='--')
                     for f in event_frames]
                    [plt.axvline(i*rates.shape[1], color='w', linestyle='-')
                     for f in event_frames]
            else:
                [plt.axvline(0, color='r', linestyle='--') for f in event_frames]

        if centroids_to_plot is not None:
            plot_single_cluster_spatial_map(which_clust=c,
                                            centroids_to_plot=centroids_to_plot,
                                            clustering=clustering,
                                            ax=plt.subplot(1, 2, 2))
        titlestr = '{} -- In: {:.3f}, Best out: {:.3f}'.format(
            c, inclust[c], np.max(outclust[c]))
        plt.title(titlestr)



def plot_clusters_means_per_trial_type(rates, clustering, trial_sets,
                                       do_plot=True):
    """
    For each trial_type in trial sets, imshow the average trace of each
    cluster.

    trace within
    :param rates: [ncells x ntime x ntrials]
    :param clustering: [ncells] A cluster assignment for each source.
    :param trial_sets: tuple of ntrial_types arrays. Each array is boolean,
                       of length ntrials, and indicates whether
                       each trial is included in that trial type.

    :return trial_type_avgs: [nclusters x  nframes_per_trial x ntrial_types]
    """

    rates_flat = np.reshape(rates,
                            (rates.shape[0], rates.shape[1] * rates.shape[2]),
                            order='F')
    clust_avgs = get_cluster_averages(rates_flat, clustering)
    clust_avgs_trial = np.reshape(clust_avgs, (
    clust_avgs.shape[0], rates.shape[1], rates.shape[2]), order='F')
    trial_type_avgs = utils.average_within_trial_types(clust_avgs_trial,
                                                       trial_sets)
    if do_plot:
        for trial_type in range(len(trial_sets)):
            plt.figure()
            plt.imshow(zscore(trial_type_avgs[:, :, trial_type], axis=1),
                       aspect='auto')
            plt.ylabel('cluster #')
            plt.xlabel('frame')
            plt.title('trial type: {}'.format(trial_type))

    return trial_type_avgs


def summarize_individual_cluster(which_clust, CT, clustering, trial_sets,
                                 which_trials=None, save_dir=None):
    """
    Plot various aspects of a specified cluster.
    - Average cluster response for each trial type
    - Single-trial traces for each source in the cluster
    - Contours of ROIs sources in cluster

    :param which_clust: int. id of the cluster.
    :param CT: CosmosDataset object.
    :param clustering: [ncells] A cluster assignment for each source.
    :param trial_sets: tuple of ntrial_types arrays. Each array is boolean,
                       of length ntrials, and indicates whether
                       each trial is included in that trial type.
    :param which_trials: Optionally specify a subset of single trials
                         for showing the single-trial traces of the
                         sources in the cluster. If None, will just
                         use the first few trials.


    """

    rates = CT.St
    trial_type_avgs = plot_clusters_means_per_trial_type(rates, clustering,
                                                         trial_sets,
                                                         do_plot=False)

    ### Plot average cluster response for each trial type
    plt.figure()
    plt.imshow(zscore(trial_type_avgs[which_clust, :, :].T, axis=1),
               aspect='auto')
    plt.ylabel('trial type')
    plt.xlabel('frame')
    plt.title('clust {}'.format(which_clust))

    ### Single-trial traces for each source in the cluster
    if which_trials is None:
        which_trials = np.arange(3)

    plt.figure(figsize=(10, 10))
    trial_rates = plot_cluster_member_traces(rates,
                                             clustering,
                                             which_clust,
                                             which_trials=which_trials,
                                             event_frames=[0],
                                             cmap='gray'
                                             )
    plt.ylabel('Source #')
    plt.suptitle('Single-trial trace of each source in cluster')
    plt.xlabel('Frame #')

    ### Single-trial trace averaged across sources in the cluster
    plt.figure()
    plt.plot(np.mean(trial_rates, axis=0))
    [plt.axvline(x, color='r') for x in which_trials * rates.shape[1]]
    plt.ylabel('Activity')
    plt.title('Mean single-trial activity across sources in cluster')
    plt.xlabel('Frame #')

    ### Contours of ROIs sources in cluster.
    which_cells = np.where(clustering == which_clust)[0]
    coords = plot_cluster_contours(CT, which_cells,
                                   which_base_im=1,
                                   name='c' + str(which_clust),
                                   fig_save_path=None,
                                   just_show_highlighted=True,
                                   show_footprints=True,
                                   edge_color=(1, 0, 0, 0.4))
    plt.title('Cluster {}'.format(which_clust))


    ### Plot centroids on atlas
    # centroid_coloring = cu.assign_colors_to_sources(clustering,
    #                                                 clustering,
    #                                                 cmap='hsv')
    #
    # nmf_centroid = all_nmf[clust1]
    # centroids_to_plot = nmf_centroid['centroid_atlas_coords'][:, :]
    #
    # which_clusts = [which_clust]
    # cu.plot_cluster_spatial_maps(centroid_coloring,
    #                              nmf_centroid['ordered_clustering'],
    #                              centroids_to_plot,
    #                              #                              radius=2,
    #                              radius=10.5,  # 1.5#10,
    #                              background_color=np.array(
    #                                  [0.9, 0.9, 0.9, 0.0]),
    #                              do_overlay=True,
    #                              specific_clusters=which_clusts,
    #                              ncols=2)

def plot_single_trials_with_trial_type(source_id, rates, trial_types,
                                       start_f, end_f, event_frame, fps,
                                       trial_type_colors=['#70C169', 'r', 'w', 'c', 'orange'],
                                       cmap='gray_r'):
    """
    For a single source, plot all single trials, and also plot the trial types
    on the right side.
    :param source_id: int.
    :param rates: [nsources x ntime x ntrials]
    :param trial_types: [ntrials]
    :param start_f: int. Start frame for plotting.
    :param end_f: int. End frame for plotting.
    :param event_frame: Odor onset.
    :param fps: frames per second.
    :return:
    """
    single_trials = rates[source_id, :, :]

    start_t = (start_f - event_frame) / fps
    end_t = (end_f - event_frame) / fps

    plt.imshow(single_trials[start_f:end_f, :].T, cmap=cmap,
               extent=[start_t, end_t, len(trial_types), 0],
               aspect='auto')
    trial_colors = []
    for i in range(len(trial_types)):
        trial_color = trial_type_colors[trial_types[i]]
        #     trial_colors.append(trial_type_colors[trial_types[i]])
        plt.plot(end_t * 1.01, i, '_', color=trial_color)
    plt.xlim([start_t, end_t * 1.02])
    plt.gca().yaxis.tick_right()
    plt.axvline(0, color='k', linestyle='--')
    plt.title(source_id)

def plot_trial_type_mean_traces(source_id, rates, trial_types, event_frame, fps,
                                trial_type_colors=['g', 'r', 'w', 'c', 'orange']):
    """
    Plot the mean trace for each trial type for a single source.

    :param source_id:
    :param rates:
    :param trial_types:
    :param event_frame:
    :return:
    """
    single_trials = rates[source_id, :, :]

    trial_type_names = np.unique(trial_types)
    mean_traces = np.zeros((len(trial_type_names), single_trials.shape[0]))
    for i, ttype in enumerate(trial_type_names):
        mean_traces[i, :] = np.mean(
            single_trials[:, np.where(trial_types == ttype)[0]], axis=1)

    for i in range(mean_traces.shape[0]):
        t = np.arange(mean_traces.shape[1]) - event_frame
        t = t / fps
        plt.plot(t, mean_traces[i, :] + i * 3,
                 color=trial_type_colors[trial_type_names[i]])

    plt.xlim([0, 2.5])
    plt.title(source_id)


def  get_task_cluster_cells_per_region(region_names, regions_of_cells,
                                       clust_assignments):
    """
    Get the number of sources in each cluster in each region.
    Also get the total number of cells per region.

    :param region_names: dict. keys: i.e. 'MO', vals: id number for that region.
                         This is CT.regions.
    :param regions_of_cells: [nsources], the region id for each source.
                             This is CT.region_of_cell.
    :param clust_assignments: [nsources]. The cluster that each source is
                              assigned to.
    :return: clusters_region_dist: dict. key: cluster id.
                                         val: dict. key: region name
                                                    val: # sources.
            total_cells_per_region: dict. key: region name
                                          val: # sources.

    """

    # For each cluster, get a the proportion of sources in each region
    clusters_region_dist = {}
    clusts = np.unique(clust_assignments)
    for c in clusts:
        sources_in_clust = np.where(clust_assignments == c)[0]
        regions_in_clust = regions_of_cells[sources_in_clust]

        num_per_region_in_clust = {}
        for region in region_names.keys():
            num_per_region_in_clust[region] = len(
                np.where(regions_in_clust == region_names[region])[0])
        clusters_region_dist[c] = num_per_region_in_clust

    # Get total cells per region
    total_cells_per_region = {}
    for region in region_names.keys():
        total_cells_per_region[region] = len(
            np.where(regions_of_cells == region_names[region])[0])

    return clusters_region_dist, total_cells_per_region


def compute_spatial_dists(which_sources, centroids, hist_bins=None, do_kde=False):
    """Helper function to actually compute the pairwise spatial distance."""
    which_centroids = centroids[which_sources, :]
    dists = scipy.spatial.distance.pdist(which_centroids)
    if do_kde: ## This takes a long time.
        kde = scipy.stats.gaussian_kde(dists)
    else:
        kde = None
    if hist_bins is not None:
        h, _= np.histogram(dists, bins=hist_bins, normed=True)
    else:
        h = None


    return (dists, h, kde)

def get_pairwise_spatial_dist(clust, labels, centroids, do_shuff=False,
                              hist_bins=None, do_kde=False):
    """
    Compute pairwise distances between all points.
    :param data:
    :return:
    """

    if do_shuff:
        shuff_labels = np.random.permutation(labels)
        which_sources = shuff_labels == clust
    else:
        which_sources = labels == clust
    (dists, h, kde) = compute_spatial_dists(which_sources, centroids,
                                        hist_bins=hist_bins,
                                        do_kde=False)

    return (dists, h, kde)


def get_and_plot_empirical_cdfs(all_dists,
                                all_shuffle_dists,
                                bins,
                                dsets,
                                clusts,
                                savepath):
    """
    Get empirical CDFs and associated corrected p-values,
    and then plot cdf overlay on null distribution.

    :param all_dists: output from generate_all_pairwise_dists()
    :param all_shuffle_dists: output from generate_all_pairwise_dists()
    :param bins: list of floats, points at which to evaluate CDF.
    :param dsets: which datasets to use (keys into all_dists)
    :param clusts: which clusts to use (keys into all_dists[dset])
    :param savepath: Plot save location.
    :return: Nothing.
    """
    (all_shuffle_cdf, all_cdf) = get_cdfs(all_dists,
                                          all_shuffle_dists,
                                          bins, dsets,
                                          clusts)

    all_pvals = get_empirical_pvalues(all_shuffle_cdf,
                                      all_cdf,
                                      dsets)
    all_corrected_pvals = get_corrected_pvals(all_pvals)

    coloring = get_color_template()[1:]
    overlay_spatial_cdfs_on_null_dist(all_shuffle_cdf,
                                      all_cdf,
                                      bins,
                                      all_corrected_pvals,
                                      dsets,
                                      clusts,
                                      savepath=savepath,
                                      colors=coloring,
                                      shuff_shade_p=0.05)

def get_and_plot_empirical_hists(all_dists,
                                all_shuffle_dists,
                                bins,
                                dsets,
                                clusts,
                                savepath):
    """
    Get empirical histograms and associated corrected p-values,
    and then plot histogram overlay on null distribution.

    :param all_dists: output from generate_all_pairwise_dists()
    :param all_shuffle_dists: output from generate_all_pairwise_dists()
    :param bins: list of floats, points at which to evaluate histogram.
    :param dsets: which datasets to use (keys into all_dists)
    :param clusts: which clusts to use (keys into all_dists[dset])
    :param savepath: Plot save location.
    :return: Nothing.
    """

    (all_shuffle_hist,
     all_hist) = generate_hists_from_dists(all_shuffle_dists,
                                              all_dists,
                                              dsets,
                                              clusts,
                                              bins)

    hist_pvals = get_empirical_pvalues(all_shuffle_hist,
                                       all_hist,
                                       dsets)
    hist_corrected_pvals = get_corrected_pvals(hist_pvals)

    coloring = get_color_template()[1:]
    overlay_spatial_hists_on_null_dist(all_shuffle_hist,
                                       all_hist,
                                       bins[:-1],
                                       hist_corrected_pvals,
                                       dsets,
                                       clusts,
                                       savepath=savepath,
                                       colors=coloring,
                                       shuff_shade_p=0.05)

def get_cdfs(all_dists, all_shuffle_dists, x, dsets, clusts=None):
    """
    Compute empirical cumulative density functions (CDF).
    (This is similar to generate_hists_from_dists()
     which instead computes pdfs).

    :param all_shuffle_dist, all_dists: output from generate_all_pairwise_dists()
    :param x: define the values at which to evaluate the CDF
    :param dsets: list of dataset ids (i.e. [7, 11, 18, 19])
    :return:
    """
    all_shuffle_cdf = dict()
    all_cdf = dict()
    for dset in dsets:
        t0 = time.time()
        all_shuffle_cdf[dset] = dict()
        all_cdf[dset] = dict()
        if clusts is None:
            clusts = all_shuffle_dists[dset].keys()
        for i, clust in enumerate(clusts):
            shuffs = []
            for shuff in range(all_shuffle_dists[dset][clust].shape[0]):
                h = ECDF(all_shuffle_dists[dset][clust][shuff, :])(x)
                shuffs.append(h)
            all_shuffle_cdf[dset][clust] = np.vstack(shuffs)

            h = ECDF(all_dists[dset][clust])(x)
            all_cdf[dset][clust] = h
        print(time.time() - t0)

    return (all_shuffle_cdf, all_cdf)

def get_shuffle_percentile(shuffle_vals, vals):
    """
    Determine the percentile of each value
    relative to the shuffle distributions.
    This can then be used to compute a p-value.

    :param shuffle_vals: [nshuffles x npoints]
    :param vals: [npoints]
    :return: [npoints] percentile of each point.
    """
    pctiles = []
    for i in range(len(vals)):
        p = scipy.stats.percentileofscore(shuffle_vals[:, i],
                                          vals[i], kind='rank')
        pctiles.append(p)
    return np.hstack(pctiles)

def get_empirical_pvalues(all_shuffle_cdf, all_cdf, dsets):
    """
    Compute empirical p-values of each cdf point relative
    to the shuffle distribution, by determing the percentile
    of each value among the shuffle values.

    :param all_shuffle_cdf: dict. key: dataset name
                                  val: dict. key: cluster id.
                                             val: [npoints] CDF
                            Same format as output from
                            generate_all_pairwise_dists().
    :param all_cdf: dict. key: dataset name
                          val: dict. key: cluster id.
                                     val: [npoints] CDF
    :param dsets: list of dataset ids. i.e. [7, 11, 18, 19]
    :return: dict. key: dataset name
                   val: dict. key: cluster id.
                              val: [npoints] pvals.
    """
    all_pvals = dict()
    for dset in dsets:
        t0 = time.time()
        all_pvals[dset] = dict()
        for i, clust in enumerate(all_shuffle_cdf[dset].keys()):
            pctiles = get_shuffle_percentile(all_shuffle_cdf[dset][clust],
                                             all_cdf[dset][clust])
            pvalues = 2*(0.5 - np.abs(pctiles/100 - 0.5)) ### Two tailed...
            all_pvals[dset][clust] = pvalues
    return all_pvals


def get_corrected_pvals(all_pvals):
    """
    Apply fdr-bh multiple comparisons correction
    across all pvalues within one dataset (i.e.
    across clusters and bins of cdf/histogram
    for each cluster).
    :param all_pvals: dict. key: dataset name
                      val: dict. key: cluster id.
                              val: [npoints] pvals.
                      Output from get_empirical_pvalues()
    :return: corrected pvalues, in the same format as input.
    """
    all_corr_pvals = dict()
    for dset in all_pvals.keys():
        pvals = np.vstack([all_pvals[dset][i] for i in all_pvals[dset].keys()])
        flat_pvals = np.reshape(pvals, (pvals.shape[0] * pvals.shape[1]))
        _, corr_pvals, _, _ = multipletests(flat_pvals, alpha=0.05,
                                            method='fdr_bh',
                                            is_sorted=False, returnsorted=False)
        corr_pvals = np.reshape(corr_pvals, (pvals.shape[0], pvals.shape[1]))

        corr_pvals_dict = dict()
        for i, clust in enumerate(all_pvals[dset].keys()):
            corr_pvals_dict[clust] = corr_pvals[i, :]

        all_corr_pvals[dset] = corr_pvals_dict
    return all_corr_pvals

def generate_all_pairwise_dists(labels, centroids,
                                dsets_spatial, clusts,
                                spatial_stat_dir,
                                nshuff):
    """
    For multiple datasets, and multiple clusters for each dataset,
    compute the pairwise distance between all sources
    in that cluster.
    :param labels: dict. key: dataset
    :param centroids: dict. key: dataset
    :param dsets_spatial: list of dsets to use (i.e. [7, 11, 18, 19]
    :param clusts: list of clusts to use (i.e. range(1, 6)
    :param spatial_stat_dir: string. location to save stuff out.
    :param nshuff: how many shuffles to perform.
    :return: all_shuffle_dists: dict. key: dataset. val: key: cluster
                                                         val: [nshuffles x npairs]
             all_dists: dict. key: dataset. val: key: cluster
                                                  val: [npairs]
    """
    all_shuffle_dists = dict()
    all_dists = dict()
    for dd in dsets_spatial[:]:
        all_shuffle_dists[dd] = dict()
        all_dists[dd] = dict()

        clusts = np.arange(1, 6)
        for i, clust in enumerate(clusts):
            t0 = time.time()
            all_shuffle_dists[dd][clust] = []
            all_dists[dd][clust] = []

            for sh in range(nshuff):
                if np.mod(sh, 500) == 0:
                    print(
                        'dset: {}, clust: {}, shuff: {}'.format(dd, clust, sh))

                dists, _, _ = get_pairwise_spatial_dist(clust,
                                                        np.copy(labels[dd]),
                                                        np.copy(
                                                                centroids[dd]),
                                                        do_shuff=True,
                                                        hist_bins=None,
                                                        do_kde=False)
                all_shuffle_dists[dd][clust].append(dists)

            dists, _, _ = get_pairwise_spatial_dist(clust,
                                                    np.copy(labels[dd]),
                                                    np.copy(centroids[dd]),
                                                    do_shuff=False,
                                                    hist_bins=None,
                                                    do_kde=False)
            all_dists[dd][clust] = dists

            all_shuffle_dists[dd][clust] = np.vstack(
                all_shuffle_dists[dd][clust])
            print(time.time() - t0)
            fname = os.path.join(spatial_stat_dir,
                                 '{}_clust_{}_pairwise_dists_nshuff_{}.npz'.format(
                                     dd, clust, nshuff))
            with open(fname, 'wb') as f:
                np.savez(f, shuffle_dists=all_shuffle_dists[dd][clust],
                         dists=all_dists[dd][clust])
            print(time.time() - t0)
    return (all_shuffle_dists, all_dists)


def generate_hists_from_dists(all_shuffle_dists, all_dists,
                              dsets, clusts, bins):
    """Generate histograms of the pairwise distances (output of
    generate_all_pairwise_dists().)

    Args:
        all_shuffle_dist, all_dists: output from generate_all_pairwise_dists()
        dsets: list of dataset ids (i.e. [7, 11, 18, 19])
        bins: define the bins for the histogram
    """

    all_hist = dict()
    all_shuffle_hist = dict()

    for dset in dsets:
        t0 = time.time()
        all_shuffle_hist[dset] = dict()
        all_hist[dset] = dict()
        for i, clust in enumerate(clusts):
            shuffs = []
            for shuff in range(all_shuffle_dists[dset][clust].shape[0]):
                h, _ = np.histogram(all_shuffle_dists[dset][clust][shuff, :],
                                    bins=bins, density=True)
                shuffs.append(h)
            # all_shuffle_hist[dset][clust] = shuffs
            all_shuffle_hist[dset][clust] = np.vstack(shuffs)

            h, _ = np.histogram(all_dists[dset][clust], bins=bins, density=True)
            all_hist[dset][clust] = h
        print(time.time() - t0)

    return (all_shuffle_hist, all_hist)

def overlay_on_null_dist(shuffles, vals,
                         corrected_pvals,
                         bins,
                         color,
                         shuff_shade_p,
                         pthresh=0.05,
                         marker='.'):
    """
    Overlay empirical distribution (i.e.
    histogram or cdf) onto the range
    specified by permutation shuffles.
    Also, color the significant parts
    of the empirical distribution based
    on provided corrected pvals.

    :param shuffles: [nshuffles x nvals] shuffle
                     distributions.
    :param vals: [nvals] discretized empirical distribution.
    :param corrected_pvals: [nvals] fdr-bh corrected
                            pvals.
    :param bins: x-values for the distribution.
    :param color: color for the line.
    :param shuff_shade_p: cutoff for plotting the
                          shuffled distribution (note,
                          this does not use corrected
                          pvalue).
    :param pthresh: p-value for computing significance.
    :return:
    """
    top = np.percentile(shuffles,
                        100 * (1 - shuff_shade_p / 2), axis=0)
    bot = np.percentile(shuffles,
                        100 * shuff_shade_p / 2, axis=0)

    sig_vals = np.copy(vals)
    sig_vals[corrected_pvals > pthresh] = np.nan

    plt.fill_between(bins,
                     top,
                     bot,
                     color=[0.76, 0.76, 0.76, 1],
                     linewidth=0)

    if color is None:
        plt.plot(bins, vals, color='k')
    else:
        plt.plot(bins, vals, color=color, linewidth=0.5)

    plt.plot(bins, sig_vals, marker, color='k',
             markersize=1, linewidth=0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

def overlay_spatial_cdfs_on_null_dist(all_shuffle_cdf,
                                      all_cdf,
                                      bins,
                                      all_corrected_pvals,
                                      dsets,
                                      clusts,
                                      savepath=None,
                                      colors=None,
                                      shuff_shade_p=0.05,
                                      ):
    """
    Plot the null CDF distribution as determined by the shuffles.
    Overlay the actual pairwise distance CDF.

    :param all_shuffle_cdf: output from get_cdfs()
    :param all_cdf: output from get_cdfs()
    :param bins:  define the bins for the CDF [npoints]
    :param all_corrected_pvals: dict. key: dataset
                                        val: dict. key: cluster id
                                                    val: [npoints] p-vals
    :param dsets: which dsets to plot. list of ids.
    :param clusts: which clusts to plot. list of ids.
    :param savepath: If not None, save here.
    :param colors: i.e. can use get_color_template()
    :param shuff_shade_p: float. Determine the cutoff for plotting
                          the null dist, based on the shuffles.
    :return:
    """
    for dset in dsets:
        plt.figure(figsize=(3*len(clusts), 5))
        for i, clust in enumerate(clusts):
            plt.subplot(1, len(clusts), i + 1)

            if colors is not None:
                color = colors[i]

            overlay_on_null_dist(all_shuffle_cdf[dset][clust],
                                 all_cdf[dset][clust],
                                 all_corrected_pvals[dset][clust],
                                 bins,
                                 color,
                                 shuff_shade_p,
                                 pthresh=0.05,
                                 marker='-')

            plt.suptitle(dset)
            plt.title('clust {}'.format(clust))
            plt.xticks([0, 3, 6])
            plt.ylim([0, 1])
            plt.xlim([-0.5, 7.5])

            if i > 0:
                plt.gca().axes.yaxis.set_ticklabels([])
            if i == 0:
                plt.ylabel('Cumulative fraction of pairs')
                plt.xlabel('Pairwise distance (mm)')

            if savepath is not None:
                a = 1
                plt.gcf().set_size_inches(w=a*3.5 / 5 * len(clusts),
                                          h=a*0.75)  # Control size of figure in inches
                plt.savefig(os.path.join(savepath,
                            '{}_pairwise_spatial_dist_cdf.pdf'.format(
                                             dset)))

def overlay_spatial_hists_on_null_dist(all_shuffle_hist,
                                       all_hist,
                                       bins,
                                       all_corrected_pvals,
                                       dsets,
                                       clusts,
                                       savepath=None,
                                       colors=None,
                                       shuff_shade_p=0.05,
                                       ):
    """
    Plot the null histogram distribution as determined by the shuffles.
    Overlay the actual pairwise distance histogram.

    :param all_shuffle_hist: output from generate_hists_from_dists()
    :param all_hist: output from generate_hists_from_dists()
    :param bins:  define the bins for the histogram [npoints]
    :param all_corrected_pvals: dict. key: dataset
                                        val: dict. key: cluster id
                                                    val: [npoints] p-vals
    :param dsets: which dsets to plot. list of ids.
    :param clusts: which clusts to plot. list of ids.
    :param savepath: If not None, save here.
    :param colors: i.e. can use get_color_template()
    :param shuff_shade_p: float. Determine the cutoff for plotting
                          the null dist, based on the shuffles.
    :return:
    """
    for dset in dsets:
        plt.figure(figsize=(3*len(clusts), 5))
        for i, clust in enumerate(clusts):
            plt.subplot(1, len(clusts), i + 1)

            if colors is not None:
                color = colors[i]

            overlay_on_null_dist(all_shuffle_hist[dset][clust],
                                 all_hist[dset][clust],
                                 all_corrected_pvals[dset][clust],
                                 bins,
                                 color,
                                 shuff_shade_p,
                                 pthresh=0.05)

            plt.suptitle(dset)
            plt.title('clust {}'.format(clust))
            plt.xticks([0, 3, 6])
            plt.ylim([0, 0.5])
            plt.xlim([-0.5, 7.5])

            if i > 0:
                plt.gca().axes.yaxis.set_ticklabels([])
            if i == 0:
                plt.ylabel('Fraction of pairs')
                plt.xlabel('Pairwise distance (mm)')

            if savepath is not None:
                a = 1
                plt.gcf().set_size_inches(w=a*3.5 / 5 * len(clusts),
                                          h=a*0.75)  # Control size of figure in inches
                plt.savefig(os.path.join(savepath,
                            '{}_pairwise_spatial_dist_hist.pdf'.format(
                                             dset)))

def plot_simulated_clusters(fake_dset,
                            fake_labels,
                            CT,
                            savedir):
    """
    Plot the spatial distribution of simulated
    clusters (used as positive control
    for spatial autocorrelation analysis).
    :param fake_dset: int. dataset id.
    :param fake_labels: [nsources] cluster id of each source.
    :param fake_cluster_info:
    :param cluster_figs_dir: save directory.
    :return:Nothing
    """

    which_sources = fake_labels.astype('bool')
    plot_centroids(which_sources, CT, max_radius=1)

    if savedir is not None:
        plt.gcf().set_size_inches(w=1, h=1) # Control size of figure in inches
        os.makedirs(savedir, exist_ok=True)
        savename = '{}_simulated_cluster_map.pdf'.format(fake_dset)
        plt.savefig(os.path.join(savedir, savename),
                    transparent=True, rasterized=True, dpi=600)

def get_all_fake_labels_and_centroids(which_fake_dsets,
                                      template_labels,
                                      template_centroids,
                                      template_CT,
                                      savedir=None):
    """
    Wrapper function to generate multiple simulated clusterings.

    :param which_fake_dsets: list of simulation ids, see
                             get_fake_labels_and_centroids() for more info.
    :param template_labels: output of load_centroids_and_task_labels()
                            for one real dataset.
    :param template_centroids: output of load_centroids_and_task_labels()
                            for one real dataset. Used as a template
                            of centroids from which to simulate clusters.
    :param template_CT: CosmosTraces object for the template dataset.
    :param savedir: For plotting centroids on atlas.

    :return:
    all_fake_labels: dict. Keys: ids in which_fake_dsets. Values: [nsources]
                           cluster label for each source.
    all_fake_centroids: Keys: ids in which_fake_dsets. Values: [nsources, 2]
                        centroid for each source.
    """
    all_fake_labels = dict()
    all_fake_centroids = dict()
    for fake_dset in which_fake_dsets:  ## Different sizes and a bilateral

        (fake_labels,
         fake_centroids) = get_fake_labels_and_centroids(
            template_labels,
            template_centroids,
            fake_dset)
        if fake_labels is None:
            continue

        all_fake_labels[fake_dset] = fake_labels
        all_fake_centroids[fake_dset] = fake_centroids

        plot_simulated_clusters(fake_dset,
                                fake_labels,
                                template_CT,
                                savedir)

    return all_fake_labels, all_fake_centroids


def get_fake_labels_and_centroids(example_labels, example_centroids, fake_dset):
    """Make hand-generated cluster spatial arrangements.

    Args:
        example_labels: [nsources]
        example_centroids: [nsources x 2]
        fake_dset: The id of the fake dataset (see the if statements in the function).
                        Note: Do not make these overlap with any existing datasets
                        (i.e. [7, 11, 18, 19, 35]) since it will overwrite their
                        spatial analysis files.
    """
    np.random.seed(1)
    fake_labels = np.zeros(example_labels.shape)
    fake_centroids = np.copy(example_centroids)
    if fake_dset == 130:
        d = 0.5
        c = [1.5, 4]
        num_extra = 30
        fake_cluster = get_points_in_circle(fake_centroids, d, c)
        fake_labels[fake_cluster[np.arange(0, len(fake_cluster), 3)]] = 1 # 33%
        fake_labels = include_random_points(fake_labels, num_extra)

    elif fake_dset == 131:
        d = 0.75
        c = [1.5, 4]
        num_extra = 30
        fake_cluster = get_points_in_circle(fake_centroids, d, c)
        fake_labels[fake_cluster[np.arange(0, len(fake_cluster), 3)]] = 1 # 33%
        fake_labels = include_random_points(fake_labels, num_extra)

    elif fake_dset == 132:
        d = 1
        c = [1.5, 4]
        num_extra = 30
        fake_cluster = get_points_in_circle(fake_centroids, d, c)
        fake_labels[fake_cluster[np.arange(0, len(fake_cluster), 3)]] = 1  # 33%
        fake_labels = include_random_points(fake_labels, num_extra)

    elif fake_dset == 133:
        d = 2
        c = [1.5, 4]
        num_extra = 30
        fake_cluster = get_points_in_circle(fake_centroids, d, c)
        fake_labels[fake_cluster[np.arange(0, len(fake_cluster), 3)]] = 1  # 33%
        fake_labels = include_random_points(fake_labels, num_extra)

    elif fake_dset == 134:
        d1 = 1
        c1 = [1.5, 4]
        d2 = 1
        c2 = [6.5, 4]
        num_extra = 30
        fake_cluster = get_points_in_circle(fake_centroids, d1, c1)
        fake_labels[fake_cluster[np.arange(0, len(fake_cluster), 3)]] = 1  # 33%
        fake_cluster = get_points_in_circle(fake_centroids, d2, c2)
        fake_labels[fake_cluster[np.arange(0, len(fake_cluster), 3)]] = 1  # 33%
        fake_labels = include_random_points(fake_labels, num_extra)

    else:
        fake_labels = None
        fake_centroids = None

    return (fake_labels, fake_centroids)


def include_random_points(fake_labels, num_extra):
    """
    Set random subset of points to 1.
    :param fake_labels:
    :param num_extra:
    :return:
    """
    inds = np.random.random_integers(0, len(fake_labels) - 1, num_extra)
    fake_labels[inds] = 1
    return fake_labels


def get_points_in_circle(centroids, d, c):
    """
    Get ids of centroids within a circle of diameter d and
    center c.
    :param centroids:
    :param d:
    :param c:
    :return:
    """
    r = d/2
    ids = np.where(np.sqrt((centroids[:, 0] - c[0])**2 \
                         + (centroids[:, 1] - c[1])**2) < r)[0]
    return ids

def plot_sources_at_high_correlation_timepoints(sources_to_corr,
                                                traces_to_corr,
                                                sources_to_plot,
                                                spikes, fluor,
                                                savedir=None,
                                                window_size=176,
                                                n_to_plot=5,
                                                clim=[0, 3],
                                                do_overlay_traces=False,
                                                colors=['g', 'm'],
                                                ylim=None,
                                                do_zscore=False):
    """
    Find a timepoint of high correlation (between sources_to_corr),
    and then plot all traces (in source_to_plot) at that timepoint.

    :param sources_to_corr: list of source ids.
    :param traces_to_corr: [all_sources x time*trials]. Which traces to use
                           for computing the correlation.
    :param sources_to_plot: list of source ids. If None, will just use
                            sources_to_corr.
    :param traces_to_plot: [all_sources x time*trials]. Which traces to use
                           for plotting (in addition to plotting traces_to_corr).
                           If None, will just use traces_to_corr.
    :param savedir: Save location, if not None.
    :param window_size: Size of window (in frames) for computing correlation.
    :param n_to_plot:
    :param clim:
    :return: The frames centered in windows of high correlation.
    """

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

    dt = 0.034 # Frame length in seconds

    ### Compute windowed correlation between sources.
    wc, w_ind = utils.get_windowed_corr(traces_to_corr[sources_to_corr, :],
                                        window=int(window_size))

    # Center the peak correlation within the high correlation windows.
    frames_to_plot = []
    corrs_to_plot = []
    buffer = window_size + 10
    for n in np.argsort(-np.array(wc))[:n_to_plot]:
        do_recenter = True
        if do_recenter:
            window = traces_to_corr[sources_to_corr, w_ind[n] - buffer:w_ind[n] + buffer]
            window_mean = np.mean(window, axis=0)
            window_peak = np.argmax(window_mean)
            ind = window_peak + w_ind[n] - buffer
        else:
            ind = w_ind[n]
        frames_to_plot.append(ind)
        corrs_to_plot.append(wc[n])

    # Now plot each window.
    plot_method = 'do_traces_and_spikes'
    for iter, frame in enumerate(frames_to_plot):
        if frame + buffer < traces_to_corr.shape[1] and frame - buffer > 0:
            plt.figure(figsize=(4, 4))
            if plot_method == 'do_traces_and_spikes':
                for i in range(len(sources_to_plot)):
                    source = sources_to_plot[i]
                    x0 = frame - buffer
                    x1 = frame + buffer
                    x = dt * np.arange(x0, x1)

                    # Plot raw fluorescence
                    if do_zscore:
                        yticklocs = np.arange(0, np.max(ylim))
                        trace = scipy.stats.zscore(fluor[source, :])
                    else:
                        yticklocs = [0, 20, 40, 60]
                        trace = fluor
                    plt.plot(x, trace[x0:x1], color=colors[i], linewidth=1)

                    # Now plot inferred spikes
                    bin_spikes = (spikes[source, :] > 0).astype(float)
                    bin_spikes[bin_spikes == 0] = np.nan

                    if do_zscore:
                        plt.plot(x, -4 - 2* i + bin_spikes[x0:x1],
                                 '|', markersize=9, markeredgewidth=1,
                                 color=colors[i])
                    else:
                        plt.plot(x, -15 - 10*i + bin_spikes[x0:x1],
                                 '|', markersize=9, markeredgewidth=1, color=colors[i])

                if do_zscore:
                    plt.ylabel('Zscore')
                plt.xlabel('Time in session (s)')
                if ylim is not None:
                    plt.ylim(ylim)
                xticklocs = dt*frame + np.arange(-6, 7, 2)

            elif plot_method == 'do_overlay_traces': # Old
                for i in range(len(sources_to_plot)):
                    plt.plot(dt * np.arange((frame - buffer), (frame + buffer)),
                             traces[sources_to_plot[i], frame - buffer:frame + buffer] + 0)

                plt.ylabel('Event rate (Hz)')
                xticklocs = [dt * (frame - 100), dt * frame, dt * (frame + 100)]
            else: # Old
                plt.imshow(traces[sources_to_plot, frame - buffer:frame + buffer],
                           aspect='auto', cmap='Greys', clim=clim,
                           extent=[dt * (frame - buffer), dt * (frame + buffer), 0,
                                   len(sources_to_plot)])
                plt.ylabel('Source')

                xticklocs = [dt * frame - 6, dt * frame - 4, dt * frame - 2, dt * frame,
                             dt * frame + 2, dt * frame + 4, dt * frame + 6]
                plt.xticks(xticklocs,
                           np.floor(np.array(xticklocs)))

            plt.title('R: {:.3f}'.format(corrs_to_plot[iter]))
            savename = 'traces_{}.pdf'.format(frame)
            if savedir is not None:
                plt.gcf().set_size_inches(w=2,
                                          h=2)  # Control size of figure in inches
                plt.xticks(xticklocs,
                           np.floor(np.array(xticklocs)).astype(int))
                plt.yticks(yticklocs,
                           np.floor(np.array(yticklocs)).astype(int))

                plt.savefig(os.path.join(savedir, savename),
                            transparent=True, rasterized=True, dpi=600)

    return frames_to_plot


def plot_high_correlation_timepoints(traces, which_cells, corr_cells=None, savedir=None,
                                     n_to_plot=15, window_size=100,
                                     do_overlay_traces=False,
                                     do_zscore=False,
                                     clim=[0, 3]):
    """
    For a specified subset of cells
    (i.e. defined based on np.where(clustering == which_clust)[0]),
    find the timepoints of highest correlation, and then imshow the individual traces
    around that timepoint.

    :param traces: [ncells x nframes] traces of all sources (i.e. CT.S, the full traces)                                  )
    :param which_cells: [ncells_in_cluster] List of indices of cells to include
    :param corr_cells: List of indices to use to calculate high correlation
    :param n_to_plot: int. The number of timepoints for which to make plots
    :param window_size: int. Number of frames before and after the high-correlation timepoint
                             to include in each plot.

    """
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

    ### Compute windowed correlation between sources.
    if corr_cells is None:
        corr_cells = which_cells
    wc, w_ind = utils.get_windowed_corr(traces[corr_cells, :], window=int(window_size)) #window=50

    plt.figure()
    plt.plot(w_ind, wc)
    plt.title('Correlation across time')
    plt.xlabel('Frame')
    savename = 'correlation_across_time.pdf'
    if savedir is not None:
        plt.gcf().set_size_inches(w=2, h=1)  # Control size of figure in inches
        plt.savefig(os.path.join(savedir, savename),
                    transparent=True, rasterized=True, dpi=600)

    ### Plot timepoints of high correlation
    dt = 0.034
    if do_zscore:
        smoothZ = zscore(traces, axis=1)
    else:
        smoothZ = traces
    for n in np.argsort(-np.array(wc))[:n_to_plot]:
        do_recenter = True
        if do_recenter:
            buffer = window_size
            window = smoothZ[which_cells, w_ind[n] - buffer:w_ind[n] + buffer]
            window_mean = np.mean(window, axis=0)
            window_peak = np.argmax(window_mean)
            ind = window_peak + w_ind[n] - buffer
        else:
            ind = w_ind[n]

        if ind+buffer < smoothZ.shape[1] and ind - buffer > 0:
            plt.figure(figsize=(4, 2))
            if do_overlay_traces:
                for i in range(len(which_cells)):
                    plt.plot(dt * np.arange((ind - buffer), (ind + buffer)),
                             smoothZ[which_cells[i], ind - buffer:ind + buffer] + 0)

                # plt.plot(dt * np.arange((ind - buffer), (ind + buffer)), smoothZ[which_cells, ind - buffer:ind + buffer].T)
                plt.ylabel('Event rate (Hz)')
                xticklocs = [dt * (ind - 100), dt * ind, dt * (ind + 100)]
            else:
                plt.imshow(smoothZ[which_cells, ind - buffer:ind + buffer],
                           aspect='auto', cmap='Greys', clim=clim,
                           extent=[dt * (ind - buffer), dt * (ind + buffer), 0,
                                   len(which_cells)])
                plt.ylabel('Source')
                # xticklocs = [dt * (ind - buffer), dt * ind, dt * (ind + buffer)]
                # plt.xticks(xticklocs,
                #            ['', np.floor(dt * ind), ''])

                xticklocs = [dt*ind - 6, dt*ind - 4, dt*ind - 2, dt * ind, dt*ind + 2, dt*ind + 4,  dt*ind + 6]
                plt.xticks(xticklocs,
                           np.floor(np.array(xticklocs)))


            plt.title('R: {:.3f}'.format(wc[n]))


            plt.xlabel('Time (s)')

            savename = 'traces_{}.pdf'.format(w_ind[n])

            if savedir is not None:
                plt.gcf().set_size_inches(w=1, h=1.5)  # Control size of figure in inches
                plt.savefig(os.path.join(savedir, savename),
                            transparent=True, rasterized=True, dpi=600)

def plot_cluster_contours(CT, which_cells, which_base_im, name,
                          fig_save_path=None,
                          just_show_highlighted=True,
                          do_display_numbers=False,
                          show_footprints=False,
                          edge_color=(1,1,1,1)):
    """
    Plot the contours of the spatial footprint of the sources in a cluster
    (or any arbitrary set of sources). Optionally include an atlas and brain image.
    The output of this can be imported into imagej and overlaid
    on a raw video of the neural activity.

    :param CT: CosmosTraces object (which contains the footprints of each source)
    :param which_cells: array with the indices of the sources in a cluster
    :param which_base_im: Specify which base image to plot under the contours.
                          1 - atlas outline and brain image
                          2 - just atlas outline
                          3 - no atlas outline
                          4 - just brain image
                          5 - atlas outline, brain image, and footprints
    :param name: string. Identify the dataset and cluster id.
    :param fig_save_path: If not None, location to save out the plot.
    :param just_show_highlighted: bool. Only show the contours of sources
                                  in the cluster. (Takes longer if this is false.)
    :param do_display_numbers: bool. Display the source_id next to each contour.
    :param show_footprints: bool. If True then display the
                            footprint corresponding to each source
                            in addition to the contour of that footprint.
    """

    if which_base_im == 1:
        base_im = reg.overlay_atlas_outline(CT.atlas_outline,
                                            CT.mean_image)
    elif which_base_im == 2:
        base_im = np.zeros(CT.mean_image.shape)
        base_im[1] = 1
        base_im = reg.overlay_atlas_outline(CT.atlas_outline,
                                            base_im)
    elif which_base_im == 3:
        base_im = np.zeros(CT.mean_image.shape)
    elif which_base_im == 4:
        base_im = CT.mean_image
    elif which_base_im == 5:
        base_im = reg.overlay_atlas_outline(CT.atlas_outline,
                                            CT.mean_image)
        show_footprints = True
        edge_color = (1, 0, 0, 0.4)

    CP = CellPlotter(CT.S, CT.S, CT.footprints,
                     base_im, spikes=None,
                     errbar=None,
                     date='Clusters',
                     name=name,
                     fig_save_path=fig_save_path,
                     suffix=name + '_' + str(which_base_im) + '_' + str(int(show_footprints)) +'.png',
                     cmap=plt.cm.winter)

    CP.set_highlighted_neurons(which_cells)

    plt.figure(figsize=(20, 20))
    coords = CP.plot_contours(highlight_neurons=not just_show_highlighted, \
                     display_numbers=do_display_numbers,
                     ax=plt.subplot(111),
                     atlas_outline=None,
                     just_show_highlighted=just_show_highlighted,
                     highlight_color=(0, 0, 0, 0),
                     edge_color=edge_color,
                     contour_linewidth=2,
                     maxthr=0.5,
                     no_borders=True,
                     show_footprints=show_footprints,
                     rotate_image=False)
    return coords

def save_clustering(filename,
                    W, H, W_trial, Z, evr,
                    ordering, clustering,
                    centroid_atlas_coords,
                    ordered_clustering,
                    ordered_super_clustering,
                    trial_sets,
                    trial_names,
                    super_clust_info,
                    trial_set_inds,
                    corr_coef):
    """
    Save out variables necessary for plotting
    and processing clustering results.

    :param filename:  Save location (.pkl file)
    :param W: [time x clusters]. The time series of each cluster.
    :param H: [clusters x cells]. Weight of each cell in a cluster.
    :param W_trial: [clusters x ntime x ntrials]
    :param Z: [time x neurons]. Zhat = W*H is the best approx of Z = rates_flat.T
    :param evr: [clusters] explained variance of each cluster basis.
    :param ordering: [clusters] Ordering of the clusters (i.e. according to peak time).
                                ordering[0] is the new position of cluster 0.
    :param clustering: [ncells]. Assignment of each cell to a cluster.
    :param centroid_atlas_coords: [ncells x 2] Atlas-transformed centroid of each source
                                  (for plotting spatial distribution).
    :param ordered_clustering: [ncells]. For each cell, the ordered position of its cluster.
    :param ordered_super_clustering: [ncells] For each cell, the ordered position of its supercluster.
                                      Potentially None, if no superclustering was performed.
    :param trial_sets: [n_trial_types]. The assignment of trial to each trial type.
    :param trial_names: [n_trial_types]. The name of each trial type.
    :param super_clust_info: dict that contains: 'super_clustering' - the assignment
                                                        of each cluster to a supercluster
                                                'super_clust_titles' - the name of
                                                        each supercluster

    :param trial_set_inds: list. start index for each of the concatenated trial types.
    :param corr_coef: [ncells x ncells] The correlation coefficient of each source
                                        with the others sources.
    """

    nmf_results = {'W': W,
                   'H': H,
                   'W_trial': W_trial,
                   'Z': Z,
                   'evr': evr,
                   'ordering': ordering,
                   'clustering': clustering,
                   'centroid_atlas_coords': centroid_atlas_coords,
                   'ordered_clustering': ordered_clustering,
                   'ordered_super_clustering': ordered_super_clustering,
                   'trial_sets': trial_sets,
                   'trial_names': trial_names,
                   'super_clust_info': super_clust_info,
                   'trial_set_inds': trial_set_inds,
                   'corr_coef': corr_coef}

    with open(filename, 'wb') as f:
        pickle.dump(nmf_results, f)
        
        
        
def plot_task_classification_for_one_dataset(dataset_id, CT_ind, 
                                             allCT, sets, 
                                             clustering_dir, 
                                             cluster_figs_dir):
    """
    Make all single dataset plots for figure 3 (task-classification). 
    This was a bunch of cells in jupyter that are being factored
    into a function so that can run across multiple datasets in
    a loop.
    """
    do_plot_pre_lick_cells = False

    
    
    n_components = 40
    sets = [#{'method':'nmf', 'protocol':'4way', 'randseed':1, 'n_components':n_components, 'l1':0.0},
            {'method':'nmf', 'protocol':'full', 'randseed':1, 'n_components':n_components, 'l1':0.0},
            {'method':'classify', 'protocol':'mr2', 'randseed':'', 'n_components':'', 'l1':''},
        ]
    all_nmf = load_clustering_results(dataset_id, sets, clustering_dir, protocol_as_key=True)

    # clust_fig_dir = os.path.join(cluster_figs_dir, 'avg_plots_{}'.format(n_components), str(dataset_id))
    clust_fig_dir = os.path.join(cluster_figs_dir, str(dataset_id))

    os.makedirs(clust_fig_dir, exist_ok=True)
    print('Save to:')
    print(clust_fig_dir)
    
    ### Load clustered data.
    protocol = 'mr2' # '4way'

    ### The below requires having loaded the CT datastructure.
    fps = allCT[CT_ind].fps
    if protocol == '4way':
        rates = gaussian_filter1d(allCT[CT_ind].St, 1.5, axis=1, mode='constant')
        event_frames = allCT[CT_ind].event_frames
        data = concatenate_trial_type_avgs(all_nmf['4way']['trial_sets'], rates, 
                                              do_plot=False)
    elif protocol == '2way':
        rates = gaussian_filter1d(allCT[CT_ind].St, 1.5, axis=1, mode='constant')
        do_super_clust = True
        if not do_load:
            data = concatenate_trial_type_avgs(all_nmf['2way']['trial_sets'], rates, 
                                                  do_plot=True)
    elif protocol == 'oeg2way':
        pass

    elif protocol == 'full':
        rates = gaussian_filter1d(allCT[CT_ind].St, 1.5, axis=1, mode='constant')
        rates_flat = np.reshape(rates, (rates.shape[0], rates.shape[1]*rates.shape[2]), order='F')
        data = rates_flat
        do_super_clust = False

    elif protocol == 'glm':
        rates = gaussian_filter1d(allCT[CT_ind].St, 1.5, axis=1, mode='constant')
        event_frames = allCT[CT_ind].event_frames
        data = concatenate_trial_type_avgs(all_nmf['glm']['trial_sets'], rates, 
                                              do_plot=False)

    elif protocol == 'mr2':
        rates = gaussian_filter1d(allCT[CT_ind].St, 1.5, axis=1, mode='constant')
        event_frames = allCT[CT_ind].event_frames
        data = concatenate_trial_type_avgs(all_nmf['mr2']['trial_sets'], rates, 
                                              do_plot=False)
        data_front = concatenate_trial_type_avgs(all_nmf['mr2']['trial_sets'], rates, 
                                                    do_plot=False, get_first_half=True)
        data_back = concatenate_trial_type_avgs(all_nmf['mr2']['trial_sets'], rates, 
                                                    do_plot=False, get_second_half=True)
    print('Done computing rates.')
    
    if 'mr2' in all_nmf.keys():
        tm = utils.get_task_modulated(rates, all_nmf['mr2']['trial_sets']);
    
    
    ### Plot sources that are selective between odor onset and lick onset
    if do_plot_pre_lick_cells:
        task_classes = all_nmf['mr2']['ordered_super_clustering']
        trial_sets = all_nmf['mr2']['trial_sets']


        pre_lick_dir = os.path.join(cluster_figs_dir, 
                                     'pre_lick_sources', str(dataset_id))
        os.makedirs(pre_lick_dir, exist_ok=True)

        odor_onset = 65

        earliest_frame = odor_onset
        latest_frame = 74
        which_classes = [1, 2, 3]
        which_sets = [0, 1, 2]
        lick_rates = allCT[CT_ind].bd.spout_lick_rates
        spike_rates = allCT[CT_ind].St
        plot_pre_lick_sources(lick_rates, spike_rates,
                              task_classes, trial_sets, 
                              which_classes, which_sets,
                              earliest_frame, latest_frame,
                              fps, save_dir=pre_lick_dir)

    ### Generate the super-clustering ordered average trace figure.
    if 'mr2' in all_nmf.keys():
        source_coloring = assign_colors_to_sources(all_nmf['mr2']['ordered_clustering'], 
                                                      all_nmf['mr2']['ordered_super_clustering'],
                                                      cmap='jet',
                                                      same_within_super_cluster=True,
                                                      specify_discrete_colors=True)

        trial_set_inds = all_nmf['mr2']['trial_set_inds']
        trial_names = all_nmf['mr2']['trial_names']
        label_positions = trial_set_inds + np.diff(trial_set_inds)[0]/2 ## To label each trial type.
        cm = 'plasma'
        plot_clustered_sources(all_nmf['mr2']['ordered_clustering'], 
                                  all_nmf['mr2']['ordered_super_clustering'], 
                                  source_means=data_back,  ### Set to data_back for cross-validated plotting. 
                                                           ### Or set to data_front to use the same as ordering. 
                                  source_coloring=source_coloring,
                                  cmap=cm, 
                                  clim=[-2, 4],
                                  vertical_lines={'k':trial_set_inds, 
                                                  'w': np.hstack((trial_set_inds+64, 
                                                                  trial_set_inds+108))
                                                 }, 
                                  title_str='dataset {}'.format(dataset_id),
                                  labels={'labels':trial_names, 
                                          'positions':label_positions},
                                  time_labels={'labels':['0']*len(trial_names) + 
                                                        ['2']*len(trial_names),
                                               'positions': np.hstack(
                                                   (trial_set_inds+event_frames[1],
                                                    trial_set_inds+event_frames[1]+2*fps))},
                                  exclude_super_clusters=[0],
                                 )

        savename = 'id'+str(dataset_id)+'_'+protocol+'_traces_'+cm
        if not os.path.isdir(clust_fig_dir):
            os.makedirs(clust_fig_dir)
        print(clust_fig_dir)
    #     plt.gcf().set_size_inches(w=2.5, h=4) # Control size of figure in inches
        plt.savefig(os.path.join(clust_fig_dir, savename+'.pdf'), 
                    transparent=True, rasterized=True, dpi=600) 
        plt.savefig(os.path.join(clust_fig_dir, savename+'.png'), 
                    transparent=True, rasterized=True, dpi=50) 
    else:
        print('Not plotting super-clusters.')
        
    ### Now plot the average traces of each super cluster.
    if 'mr2' in all_nmf.keys():
        plot_cluster_averages(data, 
                                 all_nmf['mr2']['ordered_clustering'], 
                                 all_nmf['mr2']['ordered_super_clustering'], 
                                 trial_set_inds, by_trial_type=True,
                                 event_frames=[event_frames[1],
                                               event_frames[1]+2*fps],
                                 event_labels=[0, 2],
                                 time_labels={'labels':['0']*len(trial_names) + ['2']*len(trial_names),
                                               'positions': np.hstack((trial_set_inds+event_frames[1],
                                                                      trial_set_inds+event_frames[1]+2*fps))},
                                 vertical_lines={'k':trial_set_inds, 
                                                 'g': np.hstack((trial_set_inds+64, 
                                                                  trial_set_inds+108))
                                                 },
                                )

        savename = 'id'+str(dataset_id)+'_'+protocol+'_cluster_averages.pdf'
        plt.gcf().set_size_inches(w=2.819, h=0.4) # Control size of figure in inches
        plt.savefig(os.path.join(clust_fig_dir, savename), 
                    transparent=True, rasterized=True, dpi=600) 
        
    ### You can try only include the trial-modulated sources. 
    do_include_only_task_modulated = False
    if do_include_only_task_modulated:
        source_coloring[~tm, :] = np.array([0.8, 0.8, 0.8, 0.8])
        
    ### For each super cluster, plot the location of the included cells, over grey background dots,
    ### with the corresponding color from the above plot. 
    if 'mr2' in all_nmf.keys():
        subplot_titles = None
    #     subplot_titles= all_nmf['4way']['super_clust_info']['titles']
        plot_cluster_spatial_maps(source_coloring,
                                     all_nmf['mr2']['ordered_super_clustering'],
                                     all_nmf['mr2']['centroid_atlas_coords'],
                                     radius=3, #0.5, 
                                     do_overlay=False,
                                     background_color=np.array([0.8, 0.8, 0.8, 0.8]),
                                     subplot_titles= subplot_titles,
                                     specific_clusters=None)

        savename = 'id'+str(dataset_id)+'_'+protocol+'_spatial_maps.pdf'
        if subplot_titles is None:
            plt.gcf().set_size_inches(w=3.7, h=2) # Control size of figure in inches
        else:
            plt.gcf().set_size_inches(w=2.5, h=2) # Control size of figure in inches

        plt.savefig(os.path.join(clust_fig_dir, savename), 
                    transparent=True, rasterized=True, dpi=600) 
        
        
    ### Plot single trials for the schematic, with the trial type on the right side, and also the mean traces
    plt.figure(figsize=(15, 15))
    if dataset_id == 7:
        source_id = 1031
    #     source_id = 5
    else: 
        source_id = 0 # until further notice...

    trial_types = np.copy(allCT[CT_ind].bd.spout_positions)
    trial_types[np.where(allCT[CT_ind].bd.trial_types==4)[0]] = 0

    event_frame = allCT[CT_ind].event_frames[1]
    plot_single_trials_with_trial_type(source_id, rates, trial_types,
                                         start_f=50, end_f=150, 
                                          event_frame=event_frame, fps=fps,
                                         cmap='gray_r')

    plt.gcf().set_size_inches(w=1.5, h=3) # Control size of figure in inches
    savename = 'id'+str(dataset_id)+'_'+protocol+'_single_trial_'+str(source_id)+'.pdf'
    plt.savefig(os.path.join(clust_fig_dir, savename), 
                transparent=True, rasterized=True, dpi=600) 

    ### Now plot mean traces
    plt.figure()
    plot_trial_type_mean_traces(source_id, rates, trial_types, event_frame, fps)

    plt.gcf().set_size_inches(w=1.5, h=3) # Control size of figure in inches
    savename = 'id'+str(dataset_id)+'_'+protocol+'_single_trial_means_'+str(source_id)+'.pdf'
    plt.savefig(os.path.join(clust_fig_dir, savename), 
                transparent=True, rasterized=True, dpi=600) 

    