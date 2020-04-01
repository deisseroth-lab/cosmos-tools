import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import os
import time
import matplotlib.pyplot as plt
import cosmos.traces.trace_analysis_utils as utils


def partition_dataset_subset(rates, BD, which_subset,
                             fraction_test=0.5, rand_seed=0,
                             do_debug=False):
    """
    Split a subset of the dataset into test and train sets.
    Keep trials intact. For example, only use pre-odor time period.

    :param rates: [nsources x ntrials x ntime] The time series
                  of each source for each trial.
    :param BD: BpodDataset class instance (i.e. contains info about
               the trial types and timing within each trial, and trial
               block structure etc.)
    :param which_subset: string that indicates which subset of the data to use.
                         - 'all': All timepoints of all trials.
                         - 'pre_odor': The 2 seconds prior to odor onset
                         - 'post_odor': 2 seconds beginning 0.5s after odor onset.
                         - 'block_start': Only use trials from the first half of a block.
                         - 'block_end': Only use trials from the second half of a block.
    :param fraction_test: float. Fraction of the total trials that should be assigned
                          to the test dataset.

    :return: data_split: a dict. keys: 'test' or 'train'.
                                 vals: The associated rates matrices.
    :return trials_split: a dict. keys: 'test' or 'train'.
                                  vals: The associated trial indices.
    """

    trial_types = BD.spout_positions
    ind_within_block = BD.ind_within_block
    fps = 1.0 / 0.034  # Frames per second of neural imaging.

    if which_subset=='all':
        print('Frame range: all')
        data_split, trials_split = partition_dataset(rates, trial_types,
                                                     fraction_test=fraction_test,
                                                     rand_seed=rand_seed,
                                                     do_debug=do_debug)
    elif which_subset=='pre_odor':
        subset_rates = rates.copy()
        odor_frame = round(BD.stimulus_times[0]*fps)
        start_frame = int(np.floor(odor_frame - 2*fps - 1))
        end_frame = int(np.floor(odor_frame - 1))
        print('Frame range: {}-{} '.format(start_frame, end_frame))
        subset_rates = subset_rates[:, :, start_frame:end_frame]

        data_split, trials_split = partition_dataset(subset_rates, trial_types,
                                                     fraction_test=fraction_test,
                                                     rand_seed=rand_seed,
                                                     do_debug=do_debug)

    elif which_subset=='post_odor':
        subset_rates = rates.copy()
        odor_frame = round(BD.stimulus_times[0]*fps)
        start_frame = int(np.floor(odor_frame + 0.5*fps))
        end_frame = int(np.floor(odor_frame + 2.5*fps))
        print('Frame range: {}-{} '.format(start_frame, end_frame))
        subset_rates = subset_rates[:, :, start_frame:end_frame]

        data_split, trials_split = partition_dataset(subset_rates, trial_types,
                                                     fraction_test=fraction_test,
                                                     rand_seed=rand_seed,
                                                     do_debug=do_debug)
    elif which_subset=='block_start':
        subset_rates = rates.copy()
        which_trials = np.where(ind_within_block < 5)[0]

        odor_frame = round(BD.stimulus_times[0]*fps)
        start_frame = int(np.floor(odor_frame - 1*fps))
        end_frame = int(np.floor(odor_frame + 2.5*fps))
        print('Frame range: {}-{} '.format(start_frame, end_frame))
        subset_rates = subset_rates[:, which_trials, start_frame:end_frame]

        data_split, trials_split = partition_dataset(subset_rates,
                                                     trial_types[which_trials],
                                                     fraction_test=fraction_test,
                                                     rand_seed=rand_seed,
                                                     do_debug=do_debug)

        ### Put selected trials in the original frame of reference
        ### (i.e. with all trials).
        trials_split['test'] = which_trials[trials_split['test']]
        trials_split['train'] = which_trials[trials_split['train']]

    elif which_subset=='block_end':
        subset_rates = rates.copy()
        which_trials = np.where(np.logical_and(ind_within_block > 10,
                                               ind_within_block < 15))[0]

        odor_frame = round(BD.stimulus_times[0] * fps)
        start_frame = int(np.floor(odor_frame - 1 * fps))
        end_frame = int(np.floor(odor_frame + 2.5 * fps))
        print('Frame range: {}-{} '.format(start_frame, end_frame))
        subset_rates = subset_rates[:, which_trials, start_frame:end_frame]

        data_split, trials_split = partition_dataset(subset_rates,
                                                     trial_types[which_trials],
                                                     fraction_test=fraction_test,
                                                     rand_seed=rand_seed,
                                                     do_debug=do_debug)

        ### Put selected trials in the original frame of reference
        ### (i.e. with all trials).
        trials_split['test'] = which_trials[trials_split['test']]
        trials_split['train'] = which_trials[trials_split['train']]
        pass


    return data_split, trials_split


def partition_dataset(rates, trial_types, fraction_test=0.5, rand_seed=0,
                      do_debug=False):
    """
    Split a dataset into test and train
    subsets. Keep entire trials intact. 
    
    #### Do I need to ensure that I have a minimum number of each trial type?

    :param rates: [nsources x ntrials x ntime] The time series
                  of each source for each trial.
    :param trial_types: [ntrials]. The type of each trial. So that you
                        can ensure that test and train subsets each get an equal
                        split of all of the trial types.
    :param fraction_test: float. Fraction of the total trials that should be assigned
                          to the test dataset.
    :return: data_split: a dict. keys: 'test' or 'train'.
                                 vals: The associated rates matrices.
    :return trials_split: a dict. keys: 'test' or 'train'.
                                  vals: The associated trial indices.
    """

    np.random.seed(0)
    ncells, ntrials, ntime = rates.shape

    # Initialize output data structures.
    data_split = dict()
    trials_split = dict()
    train_trials = np.array([]).astype(int)
    test_trials = np.array([]).astype(int)

    # Randomly assign trials to test or train,
    # while ensuring that each trial type is equally
    # represented.
    for trial_type in np.unique(trial_types):
        trials = np.where(trial_types==trial_type)[0]
        trials = np.random.permutation(trials).astype(int)
        ntest = int(len(trials) * fraction_test)
        ntrain = len(trials) - ntest
        train_trials = np.append(train_trials, trials[:ntrain])
        test_trials = np.append(test_trials, trials[ntrain:])

    # Assign outputs.
    data_split['train'] = rates[:, train_trials, :]
    data_split['test'] = rates[:, test_trials, :]
    trials_split['train'] = train_trials
    trials_split['test'] = test_trials

    # Some tests to check that things are working. Could move to unit tests...
    if do_debug:
        overlap = np.intersect1d(trials_split['train'], trials_split['test'])
        print('Test/train overlap: {}'.format(overlap))

        for trial_type in np.unique(trial_types):
            for key in trials_split.keys():
                trials = trials_split[key]
                frac = (np.sum(trial_types[trials] == trial_type) /
                        np.sum(trial_types == trial_type))
                print('{}, {}, {}'.format(key, trial_type, frac))


    return data_split, trials_split


def group_sources_into_regions(CT, method):
    """
    Assign each source to a group based on the spatial location
    of its centroid.
    Can either group based on a gridding scheme, or one based
    on brain regions defined by the atlas.

    :param CT: CosmosTraces class instance. Contains information
               about the centroid location of each source as
               well as atlas alignment.
    :param method: Which strategy used for defining the groups:
                  - 'atlas': Use brain regions defined by atlas.
                  - 'grid': Grid the cortex into evenly sized
                            rectangles, ignoring the atlas.

    :return: centroid_labels: [nsources] The group label assigned
                              to each source.
    :return: group_names: dict. for each group label, an associated
                          name (i.e. the associated brain area or
                          grid coordinates).
    """
    group_names = dict()

    if method=='grid':
        grid_x = 110
        grid_y = 110

        max_x = np.max(CT.centroids[:,0])
        min_x = np.min(CT.centroids[:, 0])
        max_y = np.max(CT.centroids[:, 1])
        min_y = np.min(CT.centroids[:, 1])

        label = 0
        centroid_labels = np.zeros(np.shape(CT.centroids)[0]).astype(int)
        for row in range(np.ceil((max_y-min_y)/grid_y).astype(int)):
            for col in range(np.ceil((max_x-min_x)/grid_x).astype(int)):
                for c in range(len(centroid_labels)):
                    if (CT.centroids[c, 0] > col*grid_x + min_x and
                        CT.centroids[c,0] < (col+1)*grid_x + min_x and
                        CT.centroids[c, 1] > row*grid_y + min_y and
                        CT.centroids[c, 1] < (row+1)*grid_y + min_y):

                        centroid_labels[c] = label
                group_names[label] = 'r{}c{}'.format(row, col)
                label += 1
    elif method=='atlas':
        label = 0
        centroid_labels = np.zeros((np.shape(CT.centroids)[0], 1)).astype(int)
        for hemisphere in [0, 1]:
            for region_name in ['MO', 'PTLp', 'RSP', 'SSp', 'VIS']:
                region_id = CT.regions[region_name]
                in_hemisphere = CT.hemisphere_of_cell==hemisphere
                in_region = np.array(CT.region_of_cell)[:,np.newaxis]==region_id
                which_cells = np.logical_and(in_hemisphere, in_region)
                centroid_labels[which_cells] = label
                group_names[label] = '{}_{}'.format(region_name, hemisphere)
                label += 1
        centroid_labels = np.squeeze(centroid_labels)


    return centroid_labels, group_names


def get_distance_between_sources(centroids):
    """
    Compute the distance between the centroid of each
    source. You can use this to exclude sources within
    a certain distance of a target source.

    :param centroids: [nsources x 2]. (x, y) position of each source.
    :return: distances: [nsources, nsources]. The distance (in
                        the same units as the provided centroids)
                        between each source.
    """
    nsources = centroids.shape[0]
    distances = np.zeros((nsources, nsources))
    for source in range(nsources):
        c = centroids[source, :]
        diff = centroids - c
        distances[source, :] = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

    return distances


def precompute_regions_pca(rates, centroid_labels, ncomponents):
    """
    Compute the PCA model and dimensionality reduced form
    of the sources in each grouping (as defined by centroid_labels).
    The PCA representation is then used to predict
    the activity of individual sources in other groups.
    NOTE: If you want to predict the activity of a source using
    the other sources in the same group, then you should
    recompute the PCA representation excluding the target source.

    :param rates: [nsources x ntime] All traces (i.e. in the training dataset.)
    :param centroid_labels: [nsources]. The assignment of each
                            source to a group.
    :param ncomponents: int. How many principal components to use.

    :return: groups_pca: dict. keys: region id.
                               vals: sklearn pca model. Can be used
                                     to transform traces in the form
                                     [ntime x nsources].
    :return groups_Xtrain_reduced: dict. keys: region id.
                                         vals: [ntimepoints x ndimensions]
                                               dimensionality reduced
                                               traces corresponding to
                                               each group.
    """

    groups_pca = {}
    groups_Xtrain_reduced = {}
    for group in np.unique(centroid_labels):
        peers = centroid_labels == group
        pca = PCA(n_components=ncomponents, random_state=1)
        Xtrain = rates[peers, :].T
        Xtrain_reduced = pca.fit_transform(Xtrain)
        groups_pca[group] = pca
        groups_Xtrain_reduced[group] = Xtrain_reduced

    return groups_pca, groups_Xtrain_reduced


def predict_target_from_peers(peers, target,
                              data_split,
                              fit_opts,
                              cached_pca=None,
                              cached_Xtrain_reduced=None,
                              do_print_time=False):
    """
    Predict the trace of a target source based on
    the dimensionality reduced traces of a set
    of peer sources.

    :param peers: [npeers]. The id of each source in the
                  set of peers. (The id corresponds
                  the matrix of the traces for all sources).
    :param target: int. The id of the target source.
    :param data_split: dict. keys: 'test', and 'train'.
                             vals: [nsources, ntrials, ntimepoints]
                                   array of traces for each
                                   source.
    :param fit_opts: dict. contains:
                     - 'ridge_alpha': Regularizatino value
                     - 'ncomponents': Rank of dimensionality reduction.
    :param cached_pca: optional. An sklearn pca model fit to the
                       traces of the peers.
    :param cached_Xtrain_reduced: optional. [ntimepoints x ndimensions]
                                  The reduced dimensionality
                                  representation of the peers' traces.
    :return: results: dict. Contains results of the regression.
                     - 'peers', 'target', 'model',
                        'train_score', 'train_prediction', 'train_true'
                        'test_score', 'test_prediction', 'test_true'

    """
    peers_trace = dict()
    target_trace = dict()
    for dset in ['train', 'test']:
        nsources, ntrials, nframes = data_split[dset].shape
        rates = data_split[dset].reshape((nsources, ntrials * nframes))
        peers_trace[dset] = rates[peers, :]
        target_trace[dset] = rates[target, :]

    ### Get dimensionality reduced representation of peers' traces.
    if cached_pca is None or cached_Xtrain_reduced is None:
        print('N', end='...')
        pca = PCA(n_components=fit_opts['ncomponents'], random_state=1)
        Xtrain = peers_trace['train'].T
        Xtrain_reduced = pca.fit_transform(Xtrain)
    else:
        pca = cached_pca
        Xtrain_reduced = cached_Xtrain_reduced


    ### Fit model.
    t0 = time.time()
    model = Ridge(alpha=fit_opts['ridge_alpha'], random_state=1)
    model.fit(Xtrain_reduced, target_trace['train'])

    if do_print_time:
        print('Ridge fit time: {}'.format(time.time() - t0))

    ### Predict on training dataset.
    t1 = time.time()
    train_score = model.score(Xtrain_reduced, target_trace['train'])
    train_prediction = model.predict(Xtrain_reduced)

    if do_print_time:
        print('Ridge predict time: {}'.format(time.time() - t1))

    ### Predict on test dataset.
    Xtest = peers_trace['test'].T
    Xtest_reduced = pca.transform(Xtest)
    test_score = model.score(Xtest_reduced, target_trace['test'])
    test_prediction = model.predict(Xtest_reduced)

    ### Organize results.
    results = dict()
    results['peers'] = peers
    results['target'] = target
    results['model'] = model
    results['train_score'] = train_score
    results['train_prediction'] = train_prediction
    results['train_true'] = target_trace['train']
    results['test_score'] = test_score
    results['test_prediction'] = test_prediction
    results['test_true'] = target_trace['test']

    return results

def plot_indiv_target_results(results, centroids):
    """
    Plot peer prediction results for a single target source.

    :param results: dict containing the output of predict_target_from_peers()
    :param centroids: [nsources x 2] The (x, y) position of each source.
    :return:
    """
    peers = results['peers']
    target = results['target']

    # Look at the fit performance on the train and test data.
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(results['train_prediction'][:2000], 'r', label='Predicted')
    plt.plot(results['train_true'][:2000], 'k', alpha=0.5, label='True')
    plt.title('Train score: {:4f}'.format(results['train_score']))
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(results['test_prediction'][:2000], 'r', label='Predicted')
    plt.plot(results['test_true'][:2000], 'k', alpha=0.5, label='True')
    plt.title('Test score: {:4f}'.format(results['test_score']))
    plt.legend()
    plt.subplot(1, 3, 3)
    # Double check that you are excluding nearby neurons.
    plt.plot(centroids[peers, 0], centroids[peers, 1], 'b.')
    not_peers = np.ones(centroids.shape[0]).astype(bool)
    not_peers[peers] = 0
    plt.plot(centroids[not_peers, 0], centroids[not_peers, 1], 'r.')
    plt.plot(centroids[target, 0], centroids[target, 1], 'ko')


def get_regression_performance(results):
    """
    Compute a number of measures of the performance
    of the peer-prediction regression.

    :param: results: dict. Contains results of the regression.
                     - 'peers', 'target', 'model',
                        'train_score', 'train_prediction', 'train_true'
                        'test_score', 'test_prediction', 'test_true'
    :return: performance: dict. contains entries for each measure
                          of performance.
    """
    ytrue = results['test_true']
    ypred = results['test_prediction']

    v_exp = 1 - (np.var(ytrue - ypred) / np.var(ytrue))
    r_2 = np.corrcoef(ypred, ytrue)[0, 1] ** 2
    null_pred = ytrue.mean() * np.ones_like(ytrue)
    ed_poisson = (1 - (2 * np.sum(ytrue * np.log(np.abs(ytrue / (ypred+1e-16) + 1e-16))
                                  - ytrue + ypred))/
                      (2 * np.sum( ytrue * np.log(np.abs(ytrue / (null_pred+1e-16) + 1e-16))
                                   - ytrue + null_pred)))
    performance = {}
    performance['v_exp'] = v_exp
    performance['r_2'] = r_2
    performance['ed_poisson'] = ed_poisson

    return performance


def print_regression_performance(results, performance, fit_opts):
    """
    Print out a summary of the regression.

    :param: results: dict. Contains results of the regression.
                     - 'peers', 'target', 'model',
                        'train_score', 'train_prediction', 'train_true'
                        'test_score', 'test_prediction', 'test_true'
    :param: performance: dict. contains entries for each measure
                          of performance.
    :param fit_opts: dict. contains options for the regression.
    """
    print('Target: {}, train_group: {}, rank: {}, ' \
          'alpha: {}, target_group: {}'.format(
            results['target'],
            results['train_group'],
            fit_opts['ncomponents'],
            fit_opts['ridge_alpha'],
            results['target_group']))
    print('Variance explained: {:.4f}; R^2 {:.4f}; ED: {:.4f}'.format(
            performance['v_exp'],
            performance['r_2'],
            performance['ed_poisson']))
    print('Train score: {:.4f}'.format(results['train_score']))
    print('Test score: {:.4f}'.format(results['test_score']))



def organize_peer_prediction_results(out):
    """
    Organize the results of all peer predictions regressions
    (i.e. across all target neurons, predicted by
    all groups) into matrices indexed by target neuron id, number
    of singular values used during the regression, and which
    group was used for predicting the target.

    :param out: The loaded output file produced by peer prediction regression.
    :return: organized_out. A set of matrices using different metrics
                            for the regression performance.
                            Each matrix is of dimensions
                            [n_target_neurons x n_tested_singularvals x
                             n_groups_for_prediction]
    """

    groups = np.unique(out['centroid_labels'])
    ntargets = len(out['target_list'])
    nsv = len(out['singularval_list'])
    ngroups = len(groups)
    all_test_scores = np.zeros((ntargets, nsv, ngroups))
    all_train_scores = np.zeros((ntargets, nsv, ngroups))
    all_v_exp = np.zeros((ntargets, nsv, ngroups))
    all_r_2 = np.zeros((ntargets, nsv, ngroups))

    for i in range(len(out['test_scores'])):
        test_score = out['test_scores'][i]
        train_score = out['train_scores'][i]
        v_exp = out['v_exps'][i]
        r_2 = out['r_2s'][i]

        param = out['params'][i]
        neuron_ind = np.where(np.array(out['target_list']) == param['target'])[0]
        sv_ind = np.where(np.array(out['singularval_list']) == param['fit_opts']['ncomponents'])[0]
        train_group = np.where(np.array(groups) == param['train_group'])[0]
        all_test_scores[neuron_ind, sv_ind, train_group] = test_score
        all_train_scores[neuron_ind, sv_ind, train_group] = train_score
        all_v_exp[neuron_ind, sv_ind, train_group] = v_exp
        all_r_2[neuron_ind, sv_ind, train_group] = r_2

    organized_out = {}
    organized_out['all_test_scores'] = all_test_scores
    organized_out['all_train_scores'] = all_train_scores
    organized_out['all_v_exp'] = all_v_exp
    organized_out['all_r_2'] = all_r_2

    return organized_out


def plot_sources_by_group(centroids, centroid_labels):
    """
    Plot all sources. Color code them based on the group
    they are assigned to.

    :param centroids: [nsources x 2]. (x,y)(?) coords of
                      each source.
    :param centroid_labels: [nsources]
    :return:group_centroid: The centroid of each group of sources.
    """
    n_per_group = np.zeros((len(np.unique(centroid_labels)), 1))

    ### Compute a centroid for each group.
    group_centroid = np.zeros((len(np.unique(centroid_labels)), 2))
    for i, group_ind in enumerate(np.unique(centroid_labels).astype(int)):
        inds = np.where(centroid_labels == group_ind)[0]
        if len(inds) > 50:
            group_centroid[i, :] = (np.amax(centroids[inds, :], axis=0)
                                    + np.amin(centroids[inds, :], axis=0)) / 2

    ### Plot all centroids, color coded by group.
    for i, group_ind in enumerate(np.unique(centroid_labels).astype(int)):
        inds = np.where(centroid_labels == group_ind)[0]
        n_per_group[i] = len(inds)
        if len(inds) > 50:
            plt.plot(centroids[inds, 0], centroids[inds, 1], '.')

    plt.xlim([np.min(centroids[:, 0]), np.max(centroids[:, 0])])
    plt.ylim([np.min(centroids[:, 1]), np.max(centroids[:, 1])])
    plt.plot(group_centroid[:, 0], group_centroid[:, 1], 'o')

    return group_centroid

def plot_performance_by_target(centroid_labels, target_list,
                               group_centroid, group_names,
                               expt_num,
                               singularval_list,
                               all_scores, atlas_tform,
                               plot_vs_rank=False,
                               do_bar_chart=False):
    """
    For each target region, display the regression performance
    when predicting based on each other region.

    ### TODO: Make an option to make this a bar chart?
    :param centroid_labels: [nsources]. The group assignment of each source.
    :param target_list: [ntargets]. The ids of the target sources
                        (i.e. their traces are being predicted).
                        Indexes into the centroid_labels array.
    :param group_centroid: [ngroups x 2]. The centroid location
                          of each group of peers (i.e. each atlas region).
    :param expt_num: int. The id # of this regression experiment.
    :param singularval_list: List of all dimensionality reduction ranks used
                             in this regression experiment.
    :param all_scores: [n_target_neurons x n_tested_singularvals x
                        n_groups_for_prediction]. The regression score
                        for each target neuron in each regression
                        condition. i.e. one of the outputs
                        from organize_peer_prediction_results().
    :param atlas_tform: i.e. CT.atlas_tform (stored in the CosmosTraces
                        object for the current dataset).
    :param plot_vs_rank: bool. Whether to plot performance vs. dimensionality
                         reduction rank. Generally set this to false.
    :return:
    """

    use_groups = np.unique(centroid_labels)
    rank_ind = 0

    n_rows = int(len(use_groups) / 5) + 1
    plt.figure(figsize=(20, len(use_groups) / 3 * n_rows))

    order = np.array([0, 5, 3, 8, 2, 7, 1, 6,  4, 9]).astype(int)
    for ind, target_group in enumerate(use_groups[:]):
    # for ind, target_group in enumerate(order):

        neurons = np.intersect1d(np.where(centroid_labels == target_group)[0],
                                 target_list)
        indices = np.where(np.in1d(target_list, neurons))[0]
        target_r_2 = np.mean(all_scores[indices, :, :], axis=0)


        if do_bar_chart:
            target_score = target_r_2[rank_ind, :]
            plt.subplot(n_rows, 5, ind + 1)
            plt.bar(np.arange(len(order)), target_score[order])
            xticks = [k for k in group_names.keys()]
            xlabels = [k for k in group_names.values()]
            xlabels =  [xlabels[i][0]+xlabels[i][-1] for i in order]
            plt.xticks(xticks, xlabels, rotation=0)
            plt.ylim([0, 0.05])
            plt.title(group_names[target_group] + ' expt ' + str(expt_num))
            # import pdb; pdb.set_trace()

        else:
            if group_centroid[ind, 0] > 0:

                if plot_vs_rank:
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    target_r_2[:, target_group] = 0
                    plt.plot(singularval_list, target_r_2, '-o');
                    plt.xlabel('Rank')
                    plt.ylabel('Test score')
                    plt.title('Target group: {}'.format(target_group))
                    plt.subplot(1, 2, 2)
                else:
                    plt.subplot(n_rows, 5, ind + 1)

                target_score = target_r_2[rank_ind, :]
                target_score[ind] = 0.03  # Sets the radius
                utils.centroids_on_atlas(target_score, np.arange(len(target_score)),
                                         group_centroid,
                                         atlas_tform, max_radius=5000,
                                         highlight_inds=np.array(ind),
                                         vmin=0, vmax=0.1, fill_highlight=True)
                plt.title('Group {}, rank {}, expt {}'.format(target_group,
                                                              singularval_list[rank_ind],
                                                              expt_num))
    if not do_bar_chart:
        plt.subplot(n_rows, 5, ind + 2)
        plt.imshow(np.zeros((0, 0)), vmin=0, vmax=0.1, cmap=plt.cm.hsv)
        plt.axis('off')
        plt.colorbar()
