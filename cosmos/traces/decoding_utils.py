"""
Utility functions used for decoding lick direction from neural activity.
"""

from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import os
import pickle
import time
import scipy.stats
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import statsmodels.stats.multitest as mt
from collections import Counter

import copy
import keras

from cosmos.traces.decoding.decoders import DenseNNDecoder, LSTMDecoder
import cosmos.traces.trace_analysis_utils as utils
from cosmos.traces.cosmos_traces import CosmosTraces


# Note: 'classifier_utils' focuses on decoding the lick target from
# single-trial data. In contrast, 'decoding_utils' focuses on decoding licks
# from individual imaging frames. They are largely independent codebases
# that share a few common performance metrics. The models used are different.


default_splitopts = {'bins_current': 1,  # include the current frame
                     'bins_before': 0,  # frames before the event
                     'bins_after': 0,  # frames after the event
                     'standardize_X': True,  # u=0, s=1
                     'center_Y': True,  # u=0
                     'dense_Y': False,  # return a sliding window of Y as well
                     'train_inds': None,
                     'test_inds': None,
                     'valid_inds': None}


def to2d(x):
    return x.reshape((x.shape[0], -1, ))


def load_cosmos_traces(dataset, data_dir, base_dir, bpod_dir,
                       do_behavior_plots=False):

    dataset['data_root'] = data_dir
    dataset['fig_save_dir'] = base_dir + '/cosmos/trace_analysis/'
    dataset['behavior_dir'] = bpod_dir
    CT_load = CosmosTraces(dataset, behavior_plots=do_behavior_plots)
    return CT_load



def organize_decoding_expts_across_mice(expt_ids, data_dir, concluded_expts,
                                        which_metric, which_set,
                                        linestyle, mouse_colors,
                                        do_plot=True):
    """
    Aggregate decoding results across mice, and plot results.
    :param expt_ids:
    :param data_dir:
    :param concluded_expts:
    :param decoding_load_dir:
    :param which_metric:
    :param do_plot:
    :return:
    """

    all_accs_list = []
    all_expt_nums_organized = []
    all_collapsed_accs = [] ### Accuracies collapsed across folds (i.e. the mean or max across folds.)
    all_ordering = []
    for mind, expt_id in enumerate(expt_ids):
        expt_group = concluded_expts[expt_id]
        decoding_load_dir = os.path.join(data_dir, 'decoding_results', str(expt_group['id']))
        expt_nums = expt_group['expt_nums']
        expt_type = expt_group['expt_type']
        metric, expts_info = load_decoding_experiments(decoding_load_dir, expt_nums,
                                                       which_metric=which_metric)

        # Organize the folds of each experiment within a group
        accs_list, accs_names, expt_nums_organized = organize_decoding_experiments(metric,
                                                                                   expts_info,
                                                                                   expt_type) #'nneurons', 'neuron_set', 'hemisphere'
        accs_dict = {}
        for iter, name in enumerate(accs_names):
            accs_dict[name] = accs_list[iter]
        all_accs_list.append(accs_list)
        all_expt_nums_organized.append(expt_nums_organized)

        # Now plot the results.

        opts = dict()
        opts['which_set'] = which_set
        opts['expt_group_id'] = expt_group['id']
        opts['linestyle'] = linestyle
        opts['color'] = mouse_colors[mind]
        opts['markersize'] = 3
        mean_accs, xvals = plot_decoding_experiment_set(accs_dict, opts,
                                                        do_collapse_folds=True,
                                                        do_plot=do_plot)

        all_collapsed_accs.append(mean_accs)

    return (all_accs_list, all_expt_nums_organized, all_collapsed_accs, all_ordering, xvals)

def summarize_decoding_results(ds, model):
    """
    Provided the ground truth data and a trained model,
    summarizes the performance of the model.

    :param ds: data_split dict. Contains 'X_train', 'Y_train', 'X_valid',
                                'Y_valid', 'X_test', 'Y_test'.
                                Also contains 'sample_weights', which
                                has the weight associated for each
                                training sample (i.e. time point), for
                                train/valid/test.
    :param model: trained keras model.
    :param history: dict. the history of the training.

    :returns decode_summary: dict. Contains various metrics, each as a dict
                             containing entires for train, valid, test.
    """
    # Compute predictions on train/validation/test sets.
    y_pred = {}
    y_true = {}
    licks_pred = {}
    licks_true = {}
    weighted_score = {}
    confusion = {}

    auc_weighted = {}
    auc_micro = {}
    auc_macro = {}
    auc_samples = {}
    auc_class = {}
    for dset in ['train', 'valid', 'test']:
        y_pred[dset] = model.predict(to2d(ds['X_'+dset]))
        y_true[dset] = ds['Y_'+dset]
#         weighted_score[dset] = model.evaluate(to2d(ds['X_'+dset]),
#                                                     ds['Y_'+dset],
#                                         sample_weight=sample_weights[dset])
        if len(y_pred[dset].shape) > 1:
            licks_pred[dset] = np.argmax(y_pred[dset], 1)
        else:
            licks_pred[dset] = y_pred[dset]
            y_pred[dset] = y_pred[dset][:, np.newaxis]
        if len(ds['Y_'+dset].shape) > 1:
            licks_true[dset] = np.argmax(ds['Y_'+dset], 1)
        else:
            licks_true[dset] = ds['Y_'+dset]

        confusion[dset] = confusion_matrix(licks_true[dset],
                                           licks_pred[dset]).astype('float')
        confusion[dset] /= confusion[dset].sum(1)[:, None]

        # FIXME
        # This is the recall/sensitivity/True positive rate?
        # Confusion matrix is defined as
        # C_{i,j} equals number of
        # observations in group i
        # but predicted to be in group j.
        # This normalization gives you, along
        # the diagonal, i.e.:
        # (number correctly predicted in class i) / (total number in class i)

        # Sensitivity would then by found by normalizing
        # the columns instead of the rows?
        # You could compute the F1 score, which is 2*TP/(2*TP + FP + FN),
        # i.e. this is dividing
        # by the sum across the row and the column.

        # Should you be doing this:
        # scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        nclasses = y_true[dset].shape[1]
        for i in range(nclasses):
            fpr[i], tpr[i], _ = roc_curve(
                y_true[dset][:, i], y_pred[dset][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute AUC (with different methods for combining across classes)
        from sklearn.metrics import roc_auc_score
        auc_weighted[dset] = roc_auc_score(ds['Y_' + dset], y_pred[dset],
                                           average='weighted')
        auc_micro[dset] = roc_auc_score(ds['Y_' + dset], y_pred[dset],
                                        average='micro')
        auc_macro[dset] = roc_auc_score(ds['Y_' + dset], y_pred[dset],
                                        average='macro')
        auc_samples[dset] = roc_auc_score(ds['Y_' + dset], y_pred[dset],
                                          average='samples')
        auc_class[dset] = roc_auc_score(ds['Y_' + dset], y_pred[dset],
                                        average=None)

    decode_summary = {}
    decode_summary['y_true'] = y_true
    decode_summary['y_pred'] = y_pred
    decode_summary['licks_pred'] = licks_pred
    decode_summary['licks_true'] = licks_true
    decode_summary['confusion'] = confusion
    #  decode_summary['weighted_score'] = weighted_score

    decode_summary['auc_weighted'] = auc_weighted
    decode_summary['auc_micro'] = auc_micro
    decode_summary['auc_macro'] = auc_macro
    decode_summary['auc_samples'] = auc_samples
    decode_summary['auc_class'] = auc_class

    decode_summary['trial_labels'] = ds['trial_labels']

    return decode_summary


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


def shuffle_labels(ds, do_circ_perm=False):
    """
    Shuffle training labels (to
    compute chance level performance).

    :param ds: output of set_neurons_for_decoding().
               A dict that contains entries for
               'X_train', 'X_test', and 'X_valid' entries.
    :return: ds_shuffled: the input ds, but with 'Y_train',
             'Y_test', and 'Y_valid' randomly shuffled.
    """
    ds_shuffled = copy.deepcopy(ds)

    # Shuffles in place, along first axis.
    if do_circ_perm:
        print('Using circular permutation shuffle.')
        for key in ['Y_test', 'Y_train', 'Y_valid']:
            shift = np.random.randint(int(ds_shuffled[key].shape[0]/4),
                                      int(ds_shuffled[key].shape[0]*3/4)) # Ensure no chance for exact overlap
            ds_shuffled[key] = np.roll(ds_shuffled[key], shift, axis=0)
    else:
        np.random.shuffle(ds_shuffled['Y_test'])
        np.random.shuffle(ds_shuffled['Y_train'])
        np.random.shuffle(ds_shuffled['Y_valid'])

    return ds_shuffled


def pca_project(ds, n_components):
    """
    Extract the top principal components
    of the selected neurons, and use that for
    train/test/validate datasets.

    Computes the PCA basis using the training data.
    Then projects X_train, X_test, and X_valid
    onto that basis.

    This enables you to keep the number of parameters the same
    across different sized subsets of neurons.

    :param ds: output of set_neurons_for_decoding().
               A dict that contains entries for
               'X_train', 'X_test', and 'X_valid' entries.
    :param n_components: int. Number of PCA components.

    :return ds_pca: the input ds, but with 'X_train, 'X_test',
                    and 'X_valid' replaced by their
                    respective projections onto the PCA basis.
    """

    pca = PCA(n_components=n_components, random_state=1)
    pca.fit(ds['X_train'][:, 0, :])

    total_evr = np.sum(pca.explained_variance_ratio_)
    X_train = np.zeros(
        (ds['X_train'].shape[0], ds['X_train'].shape[1], pca.n_components))
    X_test = np.zeros(
        (ds['X_test'].shape[0], ds['X_test'].shape[1], pca.n_components))
    X_valid = np.zeros(
        (ds['X_valid'].shape[0], ds['X_valid'].shape[1], pca.n_components))

    for i in range(X_train.shape[1]):
        # Training dataset may consist of
        # multiple frames per datapoint
        # (i.e. 3 neural frames to predict one licking timepoint).
        X_train[:, i, :] = pca.transform(ds['X_train'][:, i, :])
        X_test[:, i, :] = pca.transform(ds['X_test'][:, i, :])
        X_valid[:, i, :] = pca.transform(ds['X_valid'][:, i, :])

    print('EVR: ' + str(total_evr))

    ds_pca = copy.deepcopy(ds)
    ds_pca['X_train'] = X_train
    ds_pca['X_test'] = X_test
    ds_pca['X_valid'] = X_valid

    return ds_pca


def plot_confusion_matrix(confusion_mat, cmap='Purples'):
    """
    Plot a confusion matrix with the actual
    values overlaid in each square.
    """
    _ = plt.imshow(confusion_mat, vmin=0, vmax=1, cmap=cmap)
    _ = plt.colorbar()
    for (j, i), label in np.ndenumerate(confusion_mat):
        plt.gca().text(i, j, '{:.2f}'.format(label), ha='center',
                       va='center', color='k')
    plt.xticks(np.arange(confusion_mat.shape[0]))
    plt.yticks(np.arange(confusion_mat.shape[1]))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_decoding_summary(expt_num, train_opts, decode_summary, CT,
                          neuron_opts, ds, history=None, nparams=None):
    """
    Plot:
     - confusion matrices for train, validate, test.
     - loss during training for train, validate.
     - locations of cells included in the classification.
     - ROC classification curve.
    """

    # Generate summary plots.
    plt.figure(figsize=(17, 8))
    plt.suptitle('expt {} -- {} -- # params: {}'.format(expt_num,
                                                        str(train_opts),
                                                        str(nparams)))

    # Plot confusion matrices.
    for ind, dset in enumerate(['train', 'valid', 'test']):
        ax = plt.subplot(2, 3, ind+1)
        confusion = decode_summary['confusion'][dset]
        plot_confusion_matrix(confusion)
        plt.title('{}. acc: {:.3f}'.format(dset, np.mean(np.diag(confusion))))

    # Plot training history.
    plt.subplot(2, 3, 4)
    if history is not None:
        plt.plot(history['loss'], 'b', label='train loss')
        plt.plot(history['val_loss'], 'r', label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()

    # Plot locations of included cells.
    plt.subplot(2, 3, 5)
    if CT is not None:
        CT.centroids_on_atlas(np.ones(neuron_opts['which_cells'].shape),
                              neuron_opts['which_cells'], max_radius=10)
        plt.title('{} cells'.format(len(neuron_opts['which_cells'])))

    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    plt.subplot(2, 3, 6)
    fpr_keras, tpr_keras, threshold_keras = roc_curve(
        ds['Y_valid'][:, 0], decode_summary['y_pred']['valid'][:, 0])
    plt.plot(
        fpr_keras, tpr_keras,
        label='auc {:.3f}'.format(auc(fpr_keras, tpr_keras)))
    plt.xlabel('False-positive rate')
    plt.ylabel('True-positive rate')
    plt.legend(loc='lower right')


def plot_decoded_trials(trials, ds, decode_summary,
                        expt_num=None, subset='valid'):
    """
    Plot performance on individual trials.
    :param trials: list of ints. trials within a subset.
    :param ds: data_subset dict.
    :param decode_summary: dict. output of summarize_decoding_results().
    :param expt_num: int. include this to put it in the title of the plots.
    :param subset: 'train', 'valid', or 'test'
    """
    plt.figure(figsize=(15, 3))

    if expt_num is not None:
        plt.suptitle('expt {}, {}'.format(expt_num, subset))
    for ind, trial in enumerate(trials):
        plt.subplot(1, len(trials), ind+1)

        ix = (np.where(ds['trial_labels'][subset] ==
              np.unique(ds['trial_labels'][subset])[trial])[0])

        time = np.arange(decode_summary['licks_pred'][subset].shape[0])

        plt.plot(time[ix], decode_summary['licks_true'][subset][ix], 'ro',
                 label='true', markersize=5, markerfacecolor='None',
                 markeredgewidth=0.5)
        plt.plot(time[ix], decode_summary['licks_pred'][subset][ix], 'b.',
                 label='pred', markersize=2)
        plt.legend(loc='center right')
        plt.title('Trial {}'.format(trial))


def organize_decoding_experiments(accs, expts_info, param, data_subset='test'):
    """
    Organize decoding accuracies so that they are indexed
    by the specified param (i.e. 'nneurons' or 'neuron_set').

    :param accs: dict containing lists of train/test/validate accuracies.
                 output from load_decoding_experiments.
    :param expts_info: dict containing info about each experiment,
                       corresponds to accs.
    :param param: key into expts_info to use for grouping experiments.

    :return accs_list: list of np.arrays. Each array corresponds to one
                       value of param.
    :return accs_names: the value of the param corresponding to each array
                        in accs_list. Used for plotting.
    :return which_cells_list: list of np.arrays. Each array contains the
                              indices of the neurons that have been
                              tested with the corresponding value of param.
    """
    accs_organized = defaultdict(list)
    setting = expts_info[param]
    expt_nums_organized = defaultdict(list)

    for i in range(len(setting)):
        accs_organized[str(setting[i])].append(accs[data_subset][i])
        expt_nums_organized[str(setting[i])].append(expts_info['expt_num'][i])

    accs_list = [np.array(val) for key, val in accs_organized.items()]
    accs_names = [key for key, val in accs_organized.items()]
    expt_nums = [np.array(val) for key, val in expt_nums_organized.items()]

    accs_names = np.array(accs_names)
    if param == 'nneurons':
        accs_names = accs_names.astype(int)

    return accs_list, accs_names, expt_nums


def load_decoding_experiments(decoding_save_dir, expt_nums,
                              which_metric='macro_accuracy'):
    """
    Load and organize the result accuracies of a set of decoding experiments.

    :param decoding_save_dir: string. folder containing all
                              experimental results files
    :param expt_nums: np array of int experiment id numbers.
    :param which_metric: Which metric to use for evaluating the decoding.
                         Options:
                            'macro_accuracy'
                            'macro_recall', (same as 'macro_sensitivity')
                            'macro_specificity'
                            'macro_precision',
                            'macro_f1',
                            'macro_auc'
                            'macro_informedness'

    :return metrics: dict. for train/test/valid, contains list of the specified
                     classification evaluation metric.
    :return expts_info: dict. each entry is a list of some
                        parameter associated with
                        each experiment. Parameters include
                        'neuron_set', 'nneurons',
                        'which_cells', 'hemisphere', and 'expt_num'.
    """
    expts_info = defaultdict(list)
    metrics = defaultdict(list)
    for expt_num in expt_nums:
        # if expt_num < 200:
        # Potentially remove this if no longer using those experiments.
        #     expt_file = os.path.join(decoding_save_dir,
        #                              'expt_{:04}'.format(expt_num))
        # else:
        expt_file = os.path.join(decoding_save_dir,
                                 'expt_{:06}'.format(expt_num))

        with open(expt_file + '_opts', "rb") as handle:
            opts = pickle.load(handle)
        with open(expt_file + '_decode_summary', "rb") as handle:
            decode_summary = pickle.load(handle)

        for dset in ['train', 'valid', 'test']:
            licks_true = decode_summary['licks_true'][dset]
            licks_pred = decode_summary['licks_pred'][dset]
            y_pred = decode_summary['y_pred'][dset]
            y_true = keras.utils.to_categorical(licks_true)

            cm = confusion_matrix(licks_true, licks_pred).astype('float')
            s = get_classification_summary(licks_pred, licks_true)
            mm = compute_classification_evaluations(s, do_print=False)
            fpr, tpr, roc_auc = multi_class_roc_auc(y_true, y_pred,
                                                    do_plot=False)

            if which_metric == 'macro_accuracy':
                metric = np.mean([x for x in mm['accuracy'].values()])

            elif (which_metric == 'macro_recall' or which_metric ==
                  'macro_sensitivity'):
                metric = np.mean([x for x in mm['sensitivity'].values()])

            elif which_metric == 'macro_specificity':
                metric = np.mean([x for x in mm['specificity'].values()])

            elif which_metric == 'macro_precision':
                metric = np.mean([x for x in mm['precision'].values()])

            elif which_metric == 'macro_f1':
                metric = f1_score(licks_true, licks_pred, average='macro')

            elif which_metric == 'macro_auc':
                metric = np.mean([x for x in roc_auc.values()])

            elif which_metric == 'macro_informedness':
                metric = np.mean([x for x in mm['informedness'].values()])

            metrics[dset].append(metric)

        expts_info['neuron_set'].append(opts['neuron_opts']['neuron_set'])
        expts_info['neuron_rand_seed'].append(
            opts['neuron_opts']['neuron_rand_seed'])
        expts_info['nneurons'].append(opts['neuron_opts']['nneurons'])
        expts_info['which_cells'].append(opts['neuron_opts']['which_cells'])
        expts_info['hemisphere'].append(opts['neuron_opts']['hemisphere'])
        expts_info['expt_num'].append(opts['neuron_opts']['expt_num'])
        if 'nparams' in expts_info.keys():
            expts_info['nparams'].append(opts['neuron_opts']['nparams'])

    return metrics, expts_info


def get_class_weights(ds, dsets=['train', 'valid', 'test']):
    """
    Compute the weight associated with each timepoint,
    based on the relative number of examples of each data class.
    """
    class_weights = {}
    for dset in dsets:
        Y_classes = np.argmax(ds['Y_'+dset], 1)
        weights = compute_class_weight('balanced', np.unique(Y_classes),
                                       Y_classes)

        class_weights[dset] = dict()
        for i in np.unique(Y_classes):
            class_weights[dset][i] = weights[i]
        # sample_weights[dset] = weights[np.argmax(ds['Y_'+dset], axis=1)]
    return class_weights


def get_sample_weights(ds, dsets=['train', 'valid', 'test']):
    """
    Compute the weight associated with each timepoint,
    based on the relative number of examples of each data class.
    """
    sample_weights = {}
    for dset in dsets:
        Y_classes = np.argmax(ds['Y_'+dset], 1)
        weights = compute_class_weight('balanced', np.unique(Y_classes),
                                       Y_classes)
        sample_weights[dset] = weights[np.argmax(ds['Y_'+dset], axis=1)]
    return sample_weights


def get_last_expt_in_folder(folder):
    """
    When provided a folder that contains files in the format 'expt_0001_...',
    extract the highest number experiment in the folder.
    """
    files = np.sort(np.array([int(x.split('_')[1])
                             for x in os.listdir(folder)
                             if x.split('_')[0] == 'expt']))
    if len(files) > 0:
        expt = files[-1]
    else:
        expt = 0

    return expt


def get_neuron_subsets(all_neurons, subset_size, nrepeats):
    """
    Generates individual subsets of neurons, each with
    subset_size number of neurons.
    Ensures that every neuron is in a subset at least nrepeats times.

    :param all_neurons: a list containing indices of all of the neurons
                        that are to be dealt into subsets.
                        For example, this could a list of all of the indices
                        of neurons in visual cortex. Or all of the neurons
                        on one brain hemisphere.
    :param subset_size. int. The number of neurons to be included in each
                        subset.
    :param nrepeats: int. The minimum number of times that each neuron is to be
                     included in a subset. Must be positive.

     Note: The total number of subsets returned will depend on the subset_size,
           nrepeats, and the number of neurons in all_neurons.

     :returns subsets: a list of arrays. Each array contains the neuron indices
                       corresponding to that subset.
    """
    if nrepeats <= 0:
        raise ValueError('nrepeats in get_neuron_subsets() must be >0.')

    subsets = []
    for r in range(nrepeats):
        neuron_inds = all_neurons.copy()
        neuron_inds = np.random.permutation(neuron_inds)

        if len(neuron_inds) < subset_size:
            subsets.append(neuron_inds)
        else:
            for i in range(int(np.floor(len(neuron_inds)/subset_size))):
                subsets.append(neuron_inds[i*subset_size:(i+1)*subset_size])

            if len(neuron_inds)/subset_size > np.floor(
                    len(neuron_inds)/subset_size):
                subset = neuron_inds[(i+1)*subset_size:]
                subset = np.append(
                    subset, neuron_inds[:(subset_size-len(subset))])
                subsets.append(subset)

    return subsets


def get_neurons_for_decoding(neuron_opts, CT, nrepeats, ordering=None):
    """
    Get the replicates of different neuron combinations
    for a single group of neurons.
    I.e. if you wish to get subsets of size 100 neurons
    from all of the neurons in Visual cortex. This function
    returns a list of subsets such that each neuron in visual
    cortex is in at least `nrepeats` of those subsets.

    :param neuron_opts: dict. Must contain:
                        - 'neuron_set': the label which defines
                          the group from which neurons are to be
                          drawn. (i.e. 'rand' - all neurons across cortex
                                       'VIS', 'MO', 'PTLp', 'SSp', 'RSP' -
                                               specific regions)
                        - 'hemisphere': 0 or 1. if not None, then will only
                            select neurons from one hemisphere.

    :param CT: A CosmosTraces object.
    :param nrepeats: int. The minimum number of subsets in which
                     each neuron in the overall group is to be included.

    :returns subsets: a list of arrays. Each array contains indices
                      of neurons corresponding to that subset.

    """
    random.seed(neuron_opts['neuron_rand_seed'])

    # Define all_neurons based on neuron_opts
    if neuron_opts['neuron_set'] == 'rand':  # All regions.
        all_neurons = np.arange(CT.ncells)

    elif (isinstance(neuron_opts['neuron_set'], str) and
          neuron_opts['neuron_set'] in CT.regions.keys()):  # One region.
        region = neuron_opts['neuron_set']
        all_neurons = np.array(CT.cells_in_region[CT.regions[region]])

    elif isinstance(neuron_opts['neuron_set'], list):  # Set of regions
        pass
    # TO DO if you decide it to be necessary.
    # Template for modification is commented below.
    #         nregions = len(neuron_opts['neuron_set'])
    #         which_cells = []
    #         for region in neuron_opts['neuron_set']:
    #             region_cells = np.array(
    # CT.cells_in_region[CT.regions[region]])
    #             if hemisphere is not None:
    #                 region_cells = region_cells[
    # np.where(CT.hemisphere_of_cell[region_cells]==hemisphere)[0]]
    #             if neuron_opts['nneurons'] is not None:
    #                 if len(region_cells) > neuron_opts['nneurons']/nregions:
    #                     region_cells = region_cells[
    # np.array(random.sample(range(len(region_cells)),
    # int(neuron_opts['nneurons']/nregions)))]
    #             which_cells.extend(region_cells)
    #         which_cells = np.array(which_cells)
    else:
        print('neuron_set {} has not been implemented.\n'.format(
                    neuron_opts['neuron_set']),
              'Options are \'rand\', and any key from', CT.regions.keys())

    if neuron_opts.get('hemisphere') is not None:
        all_neurons = all_neurons[
            np.where(CT.hemisphere_of_cell[all_neurons] ==
                     neuron_opts['hemisphere'])[0]]

    if ordering is not None:
        # I.e. order neurons within a region based on a
        # provided ordering of all neurons.
        ordering_index = []
        for neuron in all_neurons:
            ordering_index.append(np.where(neuron == ordering)[0])
        # import pdb
        # pdb.set_trace()
        ordering_index = np.hstack(ordering_index)
        reduced_ordering = np.argsort(ordering_index)
        subset = all_neurons[reduced_ordering[:neuron_opts['nneurons']]]
        subsets = [subset]
    else:
        if neuron_opts['nneurons'] is None:
            subsets = [all_neurons]
        else:
            subsets = get_neuron_subsets(
                all_neurons, neuron_opts['nneurons'], nrepeats)

    return subsets


def decode_trials(expt_num, decoding_save_dir, do_plot=True):
    """
    Wrapper function for decode_trials_from_decoded_frames().
    For a single decoding experiment, where individual
    frames were decoded, decode each trial by assigning
    it the label that appears most frequency within the
    decoded frames in that trial.

    :param expt_num: The id number of the decoding experiment
                     to load and process.
    :param decoding_save_dir: The path to the directory where
                             decoding experiments are saved out.
    :param do_plot: bool.

    :return confusions: The confusion matrix for the decoded trials.
                        For train/valid/test.
    :return all_trial_pred: The prediction for each trial. For train/valid/test
    :return all_trial_true: The true label for each trial. For train/valid/test
    """
    expt_file = os.path.join(decoding_save_dir,
                             'expt_{:06}'.format(expt_num))

    with open(expt_file + '_decode_summary', "rb") as handle:
        decode_summary = pickle.load(handle)

    (confusions,
     trial_pred,
     trial_true) = decode_trials_from_decoded_frames(decode_summary)

    if do_plot:
        plt.figure(figsize=(17, 8))
        plt.suptitle('expt {} decoded trials'.format(expt_num))
        for ind, dset in enumerate(['train', 'valid', 'test']):
            ax = plt.subplot(2, 3, ind + 1)
            plot_confusion_matrix(confusions[dset])
            plt.title('{}. acc: {:.3f}'.format(dset, np.mean(
                np.diag(confusions[dset]))))

        plt.figure(figsize=(17, 8))
        plt.suptitle('expt {} decoded trials'.format(expt_num))
        for ind, dset in enumerate(['train', 'valid', 'test']):
            ax = plt.subplot(2, 3, ind + 1)
            plt.plot(trial_true[dset], 'r', label='true')
            plt.plot(trial_pred[dset], 'bo', label='pred')
            plt.legend(loc='upper center', bbox_to_anchor=(0.85, 1.2))
            plt.title('{}. acc: {:.3f}'.format(dset, np.mean(
                np.diag(confusions[dset]))))

    return confusions, trial_pred, trial_true


def decode_trials_from_decoded_frames(decode_summary):
    """
    Using the results of decoding individual frames within
    each trial, decode each trial by assigning it the
    label that appears most frequently among the decoded
    frames in that trial.

    :param decode_summary: The output of summarize_decoding_results()

    :return confusions: The confusion matrix for the decoded trials.
                        For train/valid/test.
    :return all_trial_pred: The prediction for each trial. For train/valid/test
    :return all_trial_true: The true label for each trial. For train/valid/test
    """

    all_trial_pred = dict()
    all_trial_true = dict()
    confusions = dict()

    for dset in ['train', 'valid', 'test']:
        all_trial_pred[dset] = []
        all_trial_true[dset] = []

        trials = decode_summary['trial_labels'][dset]
        pred = decode_summary['licks_pred'][dset]
        true = decode_summary['licks_true'][dset]
        for trial in np.unique(trials):
            trial_inds = np.where(trials == trial)[0]
            trial_pred = pred[trial_inds]
            trial_true = true[trial_inds]

            overall_trial_pred = Counter(trial_pred).most_common(1)[0][0]
            all_trial_pred[dset].append(overall_trial_pred)

            overall_trial_true = Counter(trial_true).most_common(1)[0][0]
            all_trial_true[dset].append(overall_trial_true)

        cm = confusion_matrix(all_trial_true[dset],
                              all_trial_pred[dset]).astype('float')
        cm /= cm.sum(1)[:, None]
        confusions[dset] = cm

    return confusions, all_trial_pred, all_trial_true


def run_decoding_experiment(neuron_opts,
                            data_opts,
                            decoding_save_dir,
                            data_split,
                            CT,
                            dataset_id,
                            expt_param,
                            do_pca=True,
                            pca_components=85,
                            do_shuffle=False):
    """
    Decode licks from neural activity based on
    the provided parameters.

    :param neuron_opts: A dict containing parameters related
                        to the neural sources used for decoding
                        in the current experiment.
                        Must contain the following keys:
                        'expt_num', 'hemisphere', 'neuron_rand_seed',
                        'neuron_set', 'nneurons', 'nparams',
                        'pca_components', 'which_cells'.
    :param data_opts: A dict containing parameters related
                     to how the data was split into train/test/validate groups.
    :param decoding_save_dir: Location for saving outputs of decoding (i.e.
                              the trained model, and summary of the results).
    :param data_split: The actual data for training (i.e. output from
                       get_data_for_decoding())
    :param CT: CosmosTraces object.
    :param dataset_id: int. The id of the full dataset.
    :param expt_param: Dict containing parameters relating to this
                       specific decoding experiment (i.e. output from
                       select_experiment_group()).
    :param do_pca: bool. Project data onto PCA basis before decoding?
    :param pca_components: int. The number of pca components on which
                           to project.
    :param do_shuffle: bool. Whether to shuffle the data labels (this gives
                       you the chance level performance of decoding.


    """
    neuron_opts['do_shuffle'] = do_shuffle
    neuron_opts['do_pca'] = do_pca
    neuron_opts['pca_components'] = pca_components

    tstart = time.time()

    expt_num = neuron_opts['expt_num']

    print('Expt {}, setting {}, nneurons {}'.format(expt_num,
                                                    neuron_opts['neuron_set'],
                                                    neuron_opts['nneurons']))

    expt_file = os.path.join(decoding_save_dir, 'expt_{:06}'.format(expt_num))
    if os.path.exists(expt_file + '_model'):
        print('Change the expt number!')
    else:
        # For the specific experiment at hand, set the actual labeled data
        # to use for training.
        ds = set_neurons_for_decoding(data_split, neuron_opts,
                                      CT, do_plot=False)
        if neuron_opts['do_shuffle']:
            np.random.seed(data_opts['rand_seed'])
            ds = shuffle_labels(ds, do_circ_perm=True)

        if neuron_opts['do_pca']:
            ds = pca_project(ds, neuron_opts['pca_components'])

        # Declare model
        do_linear = True
        if do_linear:
            train_opts = {'units': 4,
                          'dropout': 0.75,
                          'num_epochs': 20,
                          'layer_activation': None,
                          'l1': None,
                          'hidden_layers': False,
                          }
        else:
            print('--->Using NONLINEAR model')
            train_opts = {'units': [10],
                          'dropout': 0.75,
                          'num_epochs': 20,
                          'layer_activation': 'relu',
                          'l1': None,
                          'hidden_layers': True,
                          }

        model_dnn = DenseNNDecoder(units=train_opts['units'],
                                   dropout=train_opts['dropout'],
                                   num_epochs=train_opts['num_epochs'],
                                   verbose=0)

        # Compute weights to account for different amounts of data per class.
        sample_weights = get_sample_weights(ds, ['train', 'valid', 'test'])

        # Setup training callbacks.
        callbacks = []
        callbacks.append(
            keras.callbacks.ModelCheckpoint(expt_file + '_model',
                                            # Overwrite at each epoch.
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            # Approx equivalent
                                            # to early stopping.
                                            save_weights_only=False,
                                            mode='auto'))

        # Fit model.
        sw = sample_weights['train']
        train_hist = model_dnn.fit(to2d(ds['X_train']),
                                   ds['Y_train'],
                                   loss='categorical_crossentropy',
                                   out_activation='softmax',
                                   layer_activation=train_opts[
                                       'layer_activation'],
                                   fit_kwargs={
                                       'callbacks': callbacks,
                                       'sample_weight': sw,
                                       'validation_data': (to2d(ds['X_valid']),
                                                           ds['Y_valid'],
                                                           sample_weights[
                                                               'valid']),
                                   },
                                   l1=train_opts['l1'],
                                   hidden_layers=train_opts['hidden_layers'])
        # import pdb; pdb.set_trace()
        neuron_opts['nparams'] = model_dnn.model.count_params()

        # Save model and related options
        opts = {'data_opts': data_opts, 'train_opts': train_opts,
                'neuron_opts': neuron_opts, 'dataset_id': dataset_id,
                'expt_param': expt_param}
        with open(expt_file + '_opts', "wb") as handle:
            pickle.dump(opts, handle)
        with open(expt_file + '_history', "wb") as handle:
            pickle.dump(train_hist.history, handle)
        #             with open(expt_file + '_ds', "wb") as handle:
        #                 pickle.dump(ds, handle)
        print('---> Saved to: ' + expt_file)

        # Now analyze results of training and save out a summary.
        do_load = True
        if do_load:
            model = keras.models.load_model(expt_file + '_model')
            with open(expt_file + '_history', "rb") as handle:
                history = pickle.load(handle)
        #                 with open(expt_file + '_ds', "rb") as handle:
        #                     ds = pickle.load(handle)
        #                     sample_weights = ds['sample_weights']
        else:
            # Note: This will not use the best model, only the most recent.
            model = model_dnn.model
            history = train_hist.history
        #                 ds = ds

        plot_dir = os.path.join(decoding_save_dir, 'plots')

        decode_summary = summarize_decoding_results(ds, model)
        with open(expt_file + '_decode_summary', "wb") as handle:
            pickle.dump(decode_summary, handle)

        plot_decoding_summary(expt_num, train_opts, decode_summary, CT,
                              neuron_opts, ds, history,
                              model.model.count_params())
        plt.savefig(
            plot_dir + '/summary_' + 'expt_{:06}'.format(expt_num) + '.png')

        # trials = [10, 15, 20]
        trials = [1, 3, 5]
        plot_decoded_trials(trials, ds, decode_summary, expt_num,
                            subset='valid')
        plt.savefig(
            plot_dir + '/trials_' + 'expt_{:06}'.format(expt_num) + '.png')

        print(time.time() - tstart)


def select_experiment_group(expt_group_id, ordering):
    """
    Select parameters corresponding to a
    group of experiments.
    See code for detail about each experiment
    group.
    Groups include: comparing the decoding accuracy
    of different numbers of includes sources,
    or sources from different regions, etc...

    :param expt_group_id: int. See below for explanation of
                          what each expt_group is.
    :param ordering: [ncells], an ordering of all of the
                       neural sources. i.e. the output
                       of get_source_discriminativity().
                       For certain expt_group_id's, this
                       is irrelevant, and can potentially
                       be set to None.
    :return expt_param: A struct containing parameters for
                        the specified expt_group_id.
                        These parameters are used by
                        get_decoding_experiment_group().
    """

    if expt_group_id == 1:
        """
        Compare regions by themselves. No ordering, just random subsets.
        """
        expt_param = {'expt_name': 'compare_regions',
                      'nrepeats': 3,  # Random subsets.
                      'neuron_rand_seed': 0,
                      'nneurons': 100}
    if expt_group_id == 2:
        """
        Compare hemispheres. No ordering, just random subsets.
        """
        expt_param = {'expt_name': 'compare_hemispheres',
                      'nrepeats': 3,
                      'neuron_rand_seed': 0}
    if expt_group_id == 3:
        """
        Compare different total numbers of sources.
        No ordering, just random subsets.
        """
        expt_param = {'expt_name': 'compare_subset_size',
                      'nrepeats': 2,  # Random subsets
                      'neuron_rand_seed': 0}
    if expt_group_id == 4:
        """
        *Compare regions by themselves. Using top n neurons according
        to discriminitivity ordering.
        """
        expt_param = {'expt_name': 'compare_regions',
                      'ordering': ordering,
                      'neuron_rand_seed': 0,
                      'nneurons': 75}
    if expt_group_id == 5:
        """
        Compare hemispheres. Using discriminitivity ordering.
        """
        expt_param = {'expt_name': 'compare_hemispheres',
                      'ordering': ordering,
                      # Ordered subsets, by anova pvalue
                      'neuron_rand_seed': 0,
                      'nneurons': 400}
    if expt_group_id == 6:
        """
        Successively append regions, i.e. starting with motor.
        Using all cells per in the region.
        """
        expt_param = {'expt_name': 'append_regions',
                      'ordering': None,
                      # Will use every cell in the window,
                      # emanating from motor cortex.
                      'neuron_rand_seed': 0,
                      'nneurons': None}
    if expt_group_id == 7:
        """
        Compare regions by themselves. Using all cells per in the region.
        """
        expt_param = {'expt_name': 'compare_regions',
                      'ordering': None,  # Will use every cell in the region.
                      'neuron_rand_seed': 0,
                      'nneurons': None}
    if expt_group_id == 8:
        """
        *Compare different total numbers of sources.
        Using discriminitivity ordering.
        """
        expt_param = {'expt_name': 'compare_subset_size',
                      'ordering': ordering,  # Ordered subsets, by anova pvalue
                      'neuron_rand_seed': 0}
    if expt_group_id == 9:
        """
        *Unique contribution of each region. Using discriminitivity ordering.
        """
        expt_param = {'expt_name': 'unique_regions',
                      'ordering': ordering,
                      'neuron_rand_seed': 0,
                      'nneurons': 75}

    if expt_group_id == 10:
        """
        *Decoding with all possible sources.
        """
        expt_param = {'expt_name': 'use_all_sources',
                      'ordering': ordering,
                      'neuron_rand_seed': 0,
                      'nneurons': None}

    return expt_param


def get_decoding_experiment_group(params, CT, decoding_save_dir):
    """
    Defines the configurations of experiments in an experiment group.
    An experiment group consists of n experiments, where each experiment
    trains a classifier using a different subset of all of the neurons.
    This function returns a list which contains a dict for each experiment,
    which contains an array of the indices of the neurons that are included
    in that experiment.

    :param expt_name: string. indexes into a switch statement that will define
                              the experiment.
    :param CT: CosmosTraces object.
    :param ordering:

    :return experiment_group: list of neuron_opts.
                              each array contains an opts dict which includes
                              - 'which_cells' (indices of cells to include)
                              - other associated parameters.
    """

    experiment_group = []
    expt_num = get_last_expt_in_folder(decoding_save_dir) + 1
    expt_name = params['expt_name']

    neuron_rand_seed = params.get('neuron_rand_seed', 0)  # Defaults to 0
    nrepeats = params.get('nrepeats', 3)  # Defaults to 3

    # Optionally specify an order an in which subsets of different
    # sizes should be included.
    ordering = params.get('ordering', None)

    # Now set up experiment group.
    if expt_name == 'compare_regions':
        if 'nneurons' not in params.keys():
            nneurons = 200
        else:
            nneurons = params['nneurons']

        for region in ['MO', 'SSp', 'PTLp', 'VIS', 'RSP']:
            neuron_opts_template = {'nneurons': nneurons,
                                    'neuron_set': region,
                                    'neuron_rand_seed': neuron_rand_seed,
                                    'hemisphere': None,
                                    'expt_num': None}

            subsets = get_neurons_for_decoding(neuron_opts_template, CT,
                                               nrepeats, ordering=ordering)
            for subset in subsets:
                neuron_opts = neuron_opts_template.copy()
                neuron_opts['which_cells'] = subset
                neuron_opts['expt_num'] = expt_num
                experiment_group.append(neuron_opts)
                expt_num += 1

    elif expt_name == 'compare_subset_size':
        for nneurons in [100, 250, 500, 750, 1000]: ### This is what you originally had.
        # for nneurons in [50, 100, 250, 500, 750, 1000]:
            neuron_opts_template = {'nneurons': nneurons,
                                    'neuron_set': 'rand',
                                    'neuron_rand_seed': neuron_rand_seed,
                                    'hemisphere': None,
                                    'expt_num': None}

            if ordering is not None:
                # subsets = [ordering[:nneurons], ordering[-nneurons:]]
                subsets = [ordering[:nneurons]]
            else:
                subsets = get_neurons_for_decoding(neuron_opts_template, CT,
                                                   nrepeats)

            for subset in subsets:
                neuron_opts = neuron_opts_template.copy()
                neuron_opts['which_cells'] = subset
                neuron_opts['expt_num'] = expt_num
                experiment_group.append(neuron_opts)
                expt_num += 1

    elif expt_name == 'use_all_sources':
        nneurons = CT.ncells

        neuron_opts_template = {'nneurons': nneurons,
                                'neuron_set': 'rand',
                                'neuron_rand_seed': neuron_rand_seed,
                                'hemisphere': None,
                                'expt_num': None}
        if ordering is not None:
            # subsets = [ordering[:nneurons], ordering[-nneurons:]]
            subsets = [ordering[:nneurons]]
        else:
            subsets = get_neurons_for_decoding(neuron_opts_template, CT,
                                               nrepeats)

        for subset in subsets:
            neuron_opts = neuron_opts_template.copy()
            neuron_opts['which_cells'] = subset
            neuron_opts['expt_num'] = expt_num
            experiment_group.append(neuron_opts)
            expt_num += 1

    elif expt_name == 'compare_hemispheres':
        for hemisphere in [0, 1]:
            # for nneurons in [100, 200, 400, 600]:
            if 'nneurons' not in params.keys():
                nneurons = 200
            else:
                nneurons = params['nneurons']
            neuron_opts_template = {'nneurons': nneurons,
                                    'neuron_set': 'rand',
                                    'neuron_rand_seed': neuron_rand_seed,
                                    'hemisphere': hemisphere,
                                    'expt_num': None}

            subsets = get_neurons_for_decoding(
                neuron_opts_template, CT, nrepeats,
                ordering=ordering)

            for subset in subsets:
                neuron_opts = neuron_opts_template.copy()
                neuron_opts['which_cells'] = subset
                neuron_opts['expt_num'] = expt_num
                experiment_group.append(neuron_opts)
                expt_num += 1

    elif expt_name == 'append_regions':
        subsets = []
        labels = ['']
        # TODO: Try swapping RSP and VIS
        for region in ['MO', 'SSp', 'PTLp', 'RSP', 'VIS']:
            for hemisphere in [0, 1]:
                neuron_opts_template = {'nneurons': None,
                                        'neuron_set': region,
                                        'neuron_rand_seed': neuron_rand_seed,
                                        'hemisphere': hemisphere,
                                        'expt_num': None}

                subsets.append(get_neurons_for_decoding(neuron_opts_template,
                                                        CT,
                                                        nrepeats,
                                                        ordering=None))
                labels.append(labels[-1]+region[0])
        for i in range(len(subsets)):
            if i > 0:
                subset = np.hstack(subsets[0:i+1])[0]
            else:
                subset = subsets[0][0]
            neuron_opts = neuron_opts_template.copy()
            neuron_opts['neuron_set'] = labels[i+1]
            neuron_opts['which_cells'] = subset
            neuron_opts['expt_num'] = expt_num
            experiment_group.append(neuron_opts)
            expt_num += 1

    elif expt_name == 'unique_regions':
        subsets = []
        labels = ['']
        for target_region in ['MO', 'SSp', 'PTLp', 'RSP', 'VIS', 'none']:
            subset = []
            for region in ['MO', 'SSp', 'PTLp', 'RSP', 'VIS']:
                if not region == target_region:
                    nneurons = params['nneurons']
                    # if target_region == 'none':
                    # nneurons = int(nneurons*4.0/5.0)
                    # # Make it the same total number of neurons
                    # when including all regions, as when leave
                    # a region out.
                    neuron_opts_template = {'nneurons': nneurons,
                                            'neuron_set': region,
                                            'neuron_rand_seed':
                                                neuron_rand_seed,
                                            'hemisphere': None,
                                            'expt_num': None}

                    subset.append(get_neurons_for_decoding(
                        neuron_opts_template, CT, nrepeats, ordering=ordering))
                # import pdb
                # pdb.set_trace()
            subsets.append(np.squeeze(np.hstack(subset)))
            labels.append('~'+target_region[0])

        for i in range(len(subsets)):
            subset = subsets[i]
            neuron_opts = neuron_opts_template.copy()
            neuron_opts['neuron_set'] = labels[i+1]
            neuron_opts['which_cells'] = subset
            neuron_opts['expt_num'] = expt_num
            experiment_group.append(neuron_opts)
            expt_num += 1

    return experiment_group


def set_neurons_for_decoding(data_split, neuron_opts, CT, do_plot=False):
    """
    Select a subset of neurons on which to train/validate/test.

    :param data_split: dict. Contains entries for X_train, Y_train.
                             X_valid, Y_valid, X_test, Y_test.
    :param neuron_opts: dict. Contains entry for:
                        - which_cells: an array with the indices of cells
                                       to include.
    :param CT: CosmosTraces object.
    """

    which_cells = neuron_opts['which_cells']

    data_subset = {}
    data_subset['X_train'] = data_split['X_train'][:, :, which_cells]
    data_subset['Y_train'] = data_split['Y_train']
    data_subset['X_valid'] = data_split['X_valid'][:, :, which_cells]
    data_subset['Y_valid'] = data_split['Y_valid']
    data_subset['X_test'] = data_split['X_test'][:, :, which_cells]
    data_subset['Y_test'] = data_split['Y_test']

    # Also include the specific trials associated with each timepoints
    # in each train/valid/test set.
    data_subset['trial_labels'] = {'train': data_split['train_trial_labels'],
                                   'valid': data_split['valid_trial_labels'],
                                   'test': data_split['test_trial_labels']}

    # Also include the indices into the original full timeseries that are
    # in each train/valid/test set.
    data_subset['idx'] = {'train': data_split['train_idx'],
                          'valid': data_split['valid_idx'],
                          'test': data_split['test_idx']
                          }

    # Plot their locations.
    if do_plot:
        CT.centroids_on_atlas(np.ones(which_cells.shape),
                              which_cells, max_radius=10)
        plt.title('{}: {} neurons'.format(neuron_opts['neuron_set'],
                                          len(neuron_opts['which_cells'])))

    return data_subset


def evenly_spaced(*iterables):
    """
    >>> evenly_spaced(range(10), list('abc'))
    [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]

    The way this works is by
    assigning a float index between 0 and 1 to each
    item (IT.count yields this index) which is
    then zipped with the item itself, where for each
    sequence, the indices span from 0 to 1.
    The zipped pairs are then concatenated, and
    then sorted according to the the indices.
    Then the indices are ignored and just the
    sorted items are returned.
    The float index serves to space out all
    members of each index.

    This is based on a stackoverflow post...
    """
    import itertools as IT
    return [item[1] for item in
            sorted(IT.chain.from_iterable(
                zip(IT.count(start=1.0 / (len(seq) + 1),
                             step=1.0 / (len(seq) + 1)), seq)
                for seq in iterables))]


def evenly_space_trial_types(trials, trial_types, do_random_permute=True):
    """
    Reorders trials such that trials of each trial type are evenly spaced
    throughout the array.

    :param trials: [ntrials] The indices of the trials.
    :param trial_types: [ntrials] The trial type corresponding to
                        each of those trials.
    :param do_random_permute: bool. Randomly permute the
                              array before reordering.
                              If false, then provided ordering
                              within each trial type
                              will remain the same.
    :return reordered_trials: [ntrials] The reordered array of trial indices
    """

    trial_types_dict = dict()
    for tt in np.unique(trial_types):
        trial_types_dict[tt] = np.where(trial_types == tt)[0]
        if do_random_permute:
            trial_types_dict[tt] = np.random.permutation(trial_types_dict[tt])
        trial_types_dict[tt] = list(trial_types_dict[tt])

    # Now distribute the trials of each trial type throughout a sequence.
    iterables = [tt for tt in trial_types_dict.values()]
    spaced_inds = evenly_spaced(*iterables)
    reordered_trials = trials[spaced_inds]

    return reordered_trials


def get_stars(p_value):
    """ Take in a p-value and return stars. """
    star_map = [('ns', np.inf), ('*', 0.05),
                ('**', 0.01), ('***', 0.001),
                ('****', 0.0001)]
    if np.isinf(p_value) or np.isnan(p_value):
        return 'ns'

    for label, thresh in star_map[::-1]:
        if p_value < thresh:
            return label

def get_pvals_for_decoding_experiment_set_vs_shuff(decoding_performances,
                                                   which_set,
                                                   shuff_decoding_performances):
    """
    Compute  statistical tests for a decoding experiment set.
    The specific tests you do depends on which experiment set.
    Compare with shuffled values for each decoding experiment.


    :param decoding_performances: [n_mice x n_expt_conditions].
                                  The decoding performance for each
                                  mouse for each condition in the
                                  experiment set.
    :param which_set: int.
        Possible experiment sets:
            - 1: performance vs. number of included neurons.
            - 2: sufficient performance vs. cortical region
            - 3: unique performance of each cortical region.
    :return: pval_str: A string containing information about the
                       relevant pvalues. You can then print this
                       or put it as the title of a plot.
    """

    # Compute statistical tests across mice.
    if which_set == 1:

        ### Repeated measures Anova before post-hoc ttests
        from statsmodels.stats.anova import AnovaRM
        import pandas
        df = pandas.DataFrame()
        mice_ids = np.arange(decoding_performances.shape[0])
        nsources = [100, 250, 500, 750, 1000]
        for mm in mice_ids:
            for nind, nn in enumerate(nsources):
                df = df.append({'nsources':nn, 'mouse_id':mm,
                                'score':decoding_performances[mm, nind]},
                               ignore_index=True)
        aovrm = AnovaRM(df, 'score', 'mouse_id', within=['nsources'])
        res = aovrm.fit()
        print(res)

        do_include_n50_sources = False
        do_compare_vs_1000 = False
        if do_compare_vs_1000:
            ind1000 = 4
            if do_include_n50_sources:
                ind1000 = 5
            t, p0 = scipy.stats.ttest_rel(decoding_performances[:, ind1000],
                                          decoding_performances[:, 0])
            t, p1 = scipy.stats.ttest_rel(decoding_performances[:, ind1000],
                                          decoding_performances[:, 1])
            t, p2 = scipy.stats.ttest_rel(decoding_performances[:, ind1000],
                                          decoding_performances[:, 2])
            t, p3 = scipy.stats.ttest_rel(decoding_performances[:, ind1000],
                                          decoding_performances[:, 3])
            if do_include_n50_sources:
                t, p4 = scipy.stats.ttest_rel(decoding_performances[:, ind1000],
                                              decoding_performances[:, 4])

        else:
            t, p0 = scipy.stats.ttest_rel(decoding_performances[:, 1],
                                          decoding_performances[:, 0])
            t, p1 = scipy.stats.ttest_rel(decoding_performances[:, 2],
                                          decoding_performances[:, 1])
            t, p2 = scipy.stats.ttest_rel(decoding_performances[:, 3],
                                          decoding_performances[:, 2])
            t, p3 = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                          decoding_performances[:, 3])
            if do_include_n50_sources:
                t, p4 = scipy.stats.ttest_rel(decoding_performances[:, 5],
                                              decoding_performances[:, 4])
        if not do_include_n50_sources:
            sig, p, _, _ = mt.multipletests([p0, p1, p2, p3],
                                            alpha=0.05, method='fdr_bh')
            ss = ' p0 {:.3}{}, p1 {:.3}{},  p2 {:.3}{}, p3 {:.3}{} vs1000={}\n'
            pval_str = ss.format(p[0], get_stars(p[0]), p[1], get_stars(p[1]),
                                 p[2], get_stars(p[2]), p[3], get_stars(p[3]),
                                 int(do_compare_vs_1000))
        else:
            sig, p, _, _ = mt.multipletests([p0, p1, p2, p3, p4],
                                            alpha=0.05, method='fdr_bh')
            ss = ' p0 {:.3}{}, p1 {:.3}{},  p2 {:.3}{}, p3 {:.3}{}, p4 {:.3}{} vs1000={}\n'
            pval_str = ss.format(p[0], get_stars(p[0]), p[1], get_stars(p[1]),
                                 p[2], get_stars(p[2]), p[3], get_stars(p[3]),
                                 p[4], get_stars(p[4]),
                                 int(do_compare_vs_1000))
    elif which_set == 2:
        t, p0 = scipy.stats.ttest_rel(decoding_performances[:, 0],
                                      shuff_decoding_performances[:, 0])
        t, p1 = scipy.stats.ttest_rel(decoding_performances[:, 1],
                                      shuff_decoding_performances[:, 1])
        t, p2 = scipy.stats.ttest_rel(decoding_performances[:, 2],
                                      shuff_decoding_performances[:, 2])
        t, p3 = scipy.stats.ttest_rel(decoding_performances[:, 3],
                                      shuff_decoding_performances[:, 3])
        t, p4 = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                      shuff_decoding_performances[:, 4])
        sig, p, _, _ = mt.multipletests([p0, p1, p2, p3, p4],
                                        alpha=0.05, method='fdr_bh')
        ss = ' p0 {:.3}{}, p1 {:.3}{},  p2 {:.3}{}, p3 {:.3}{}, p4 {:.3}{}\n'
        pval_str = ss.format(p[0], get_stars(p[0]), p[1], get_stars(p[1]),
                             p[2], get_stars(p[2]), p[3], get_stars(p[3]),
                             p[4], get_stars(p[4]))
    elif which_set == 3:
        t, p0 = scipy.stats.ttest_rel(decoding_performances[:, 0],
                                      shuff_decoding_performances[:, 0])
        t, p1 = scipy.stats.ttest_rel(decoding_performances[:, 1],
                                      shuff_decoding_performances[:, 1])
        t, p2 = scipy.stats.ttest_rel(decoding_performances[:, 2],
                                      shuff_decoding_performances[:, 2])
        t, p3 = scipy.stats.ttest_rel(decoding_performances[:, 3],
                                      shuff_decoding_performances[:, 3])
        t, p4 = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                      shuff_decoding_performances[:, 4])
        sig, p, _, _ = mt.multipletests([p0, p1, p2, p3, p4],
                                        alpha=0.05, method='fdr_bh')
        ss = ' p0 {:.3}{}, p1 {:.3}{},  p2 {:.3}{}, p3 {:.3}{}, p4 {:.3}{}\n'
        pval_str = ss.format(p[0], get_stars(p[0]), p[1], get_stars(p[1]),
                             p[2], get_stars(p[2]), p[3], get_stars(p[3]),
                             p[4], get_stars(p[4]))
    else:
        t, p = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                     decoding_performances[:, 1])
        pval_str = ' p (0 vs 4): {:.3}'.format(p)

    return pval_str


def get_pvals_for_decoding_experiment_set(decoding_performances, which_set):
    """
    Compute  statistical tests for a decoding experiment set.
    The specific tests you do depends on which experiment set


    :param decoding_performances: [n_mice x n_expt_conditions].
                                  The decoding performance for each
                                  mouse for each condition in the
                                  experiment set.
    :param which_set: int.
        Possible experiment sets:
            - 1: performance vs. number of included neurons.
            - 2: sufficient performance vs. cortical region
            - 3: unique performance of each cortical region.
    :return: pval_str: A string containing information about the
                       relevant pvalues. You can then print this
                       or put it as the title of a plot.
    """

    # Compute statistical tests across mice.
    if which_set == 1:

        ### Repeated measures Anova before post-hoc ttests
        from statsmodels.stats.anova import AnovaRM
        import pandas
        df = pandas.DataFrame()
        mice_ids = np.arange(decoding_performances.shape[0])
        nsources = [100, 250, 500, 750, 1000]
        for mm in mice_ids:
            for nind, nn in enumerate(nsources):
                df = df.append({'nsources':nn, 'mouse_id':mm,
                                'score':decoding_performances[mm, nind]},
                               ignore_index=True)
        aovrm = AnovaRM(df, 'score', 'mouse_id', within=['nsources'])
        res = aovrm.fit()
        print(res)

        t, p0 = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                      decoding_performances[:, 0])
        t, p1 = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                      decoding_performances[:, 1])
        t, p2 = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                      decoding_performances[:, 2])
        t, p3 = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                      decoding_performances[:, 3])
        sig, p, _, _ = mt.multipletests([p0, p1, p2, p3],
                                        alpha=0.05, method='fdr_bh')
        ss = ' p0 {:.3}{}, p1 {:.3}{},  p2 {:.3}{}, p3 {:.3}{}\n'
        pval_str = ss.format(p[0], get_stars(p[0]), p[1], get_stars(p[1]),
                             p[2], get_stars(p[2]), p[3], get_stars(p[3]))
    elif which_set == 2:
        t, p0 = scipy.stats.ttest_1samp(decoding_performances[:, 0], 0.5)
        t, p1 = scipy.stats.ttest_1samp(decoding_performances[:, 1], 0.5)
        t, p2 = scipy.stats.ttest_1samp(decoding_performances[:, 2], 0.5)
        t, p3 = scipy.stats.ttest_1samp(decoding_performances[:, 3], 0.5)
        t, p4 = scipy.stats.ttest_1samp(decoding_performances[:, 4], 0.5)
        sig, p, _, _ = mt.multipletests([p0, p1, p2, p3, p4],
                                        alpha=0.05, method='fdr_bh')
        ss = ' p0 {:.3}{}, p1 {:.3}{},  p2 {:.3}{}, p3 {:.3}{}, p4 {:.3}{}\n'
        pval_str = ss.format(p[0], get_stars(p[0]), p[1], get_stars(p[1]),
                             p[2], get_stars(p[2]), p[3], get_stars(p[3]),
                             p[4], get_stars(p[4]))
    elif which_set == 3:
        t, p0 = scipy.stats.ttest_1samp(decoding_performances[:, 0], 0)
        t, p1 = scipy.stats.ttest_1samp(decoding_performances[:, 1], 0)
        t, p2 = scipy.stats.ttest_1samp(decoding_performances[:, 2], 0)
        t, p3 = scipy.stats.ttest_1samp(decoding_performances[:, 3], 0)
        t, p4 = scipy.stats.ttest_1samp(decoding_performances[:, 4], 0)
        sig, p, _, _ = mt.multipletests([p0, p1, p2, p3, p4],
                                        alpha=0.05, method='fdr_bh')
        ss = ' p0 {:.3}{}, p1 {:.3}{},  p2 {:.3}{}, p3 {:.3}{}, p4 {:.3}{}\n'
        pval_str = ss.format(p[0], get_stars(p[0]), p[1], get_stars(p[1]),
                             p[2], get_stars(p[2]), p[3], get_stars(p[3]),
                             p[4], get_stars(p[4]))
    else:
        t, p = scipy.stats.ttest_rel(decoding_performances[:, 4],
                                     decoding_performances[:, 1])
        pval_str = ' p (0 vs 4): {:.3}'.format(p)

    return pval_str


def plot_decoding_experiment_set(accs_dict, opts,
                                 do_collapse_folds=True,
                                 do_plot=True):
    """
    Plot the decoding performance for each condition in an
    experiment set.
    Possible experiment sets:
        - 0: performance vs. number of included neurons.
        - 1: performance vs. number of included neurons.
        - 2: sufficient performance vs. cortical region
        - 3: unique performance of each cortical region.


    :param accs_dict: Dict. A key for each condition in the experiment set.
                             Values: A list containing the performance (i.e.
                             accuracy, or auc, or sensitivity) of each
                             fold for the corresponding condition.
    :param opts: Dict. Contains information specific to the conditions of the
                       experiment set.
    :param: do_collapse_folds: bool. If true, then plot the
                               average across folds,
                               otherwise plot each individual fold.
    :return mean_accs: list. the mean performance across folds,
                             for each condition.
    :return xvals: list. the x-value when plotting the performance
                         of each condition.
    """

    which_set = opts['which_set']
    linestyle = opts['linestyle']
    label = opts['expt_group_id']
    # label = opts['mouse_name']
    color = opts['color']
    markersize = opts['markersize']

    # Plot results for each experiment set
    if which_set == 0 or which_set == 1:
        keys = np.sort(list(accs_dict.keys()))
        mean_accs = [np.mean(accs_dict[key]) for key in keys]
        xvals = keys
        if do_plot:
            if do_collapse_folds:
                plt.plot(keys, mean_accs,
                         linestyle, label=label,
                         color=color,
                         markersize=markersize)
            else:
                nfolds = len(accs_dict[keys[0]])
                for fold in range(nfolds):
                    fold_acc = [accs_dict[key][fold] for key in keys]
                    plt.plot(keys, fold_acc,
                             linestyle, label=label,
                             color=color,
                             markersize=markersize)

            plt.xticks([0, 250, 500, 750, 1000])

    elif which_set == 2:
        keys = np.sort(list(accs_dict.keys()))

        # Hard coded ordering of regions
        # (assuming they are alphabetically sorted)
        ordering = np.array([0, 3, 1, 2, 4]).astype(int)
        keys = keys[ordering]

        xvals = np.arange(5)
        mean_accs = [np.mean(accs_dict[key]) for key in keys]
        if do_plot:
            if do_collapse_folds:
                plt.plot(xvals, mean_accs,
                         linestyle, label=label,
                         color=color,
                         markersize=markersize)
            else:
                nfolds = len(accs_dict[keys[0]])
                for fold in range(nfolds):
                    fold_acc = [accs_dict[key][fold] for key in keys]
                    plt.plot(xvals, fold_acc,
                             linestyle, label=label,
                             color=color,
                             markersize=markersize)

            labels = []
            for key in keys:
                labels.append(key[0])
            plt.gca().set_xticks(xvals)
            plt.gca().set_xticklabels(labels)

    elif which_set == 3:
        normalizer = accs_dict['~n']
        xvals = np.arange(5)

        keys = np.sort(list(accs_dict.keys()))
        keys = np.delete(keys, np.where(keys == '~n')[0][0])
        # Hard coded ordering of regions
        # (assuming they are alphabetically sorted)
        ordering = np.array([0, 3, 1, 2, 4]).astype(int)
        keys = keys[ordering]

        norm_accs = {}
        for key in keys:
            norm_accs[key] = 1 - accs_dict[key]/normalizer

        mean_accs = [np.mean(norm_accs[key]) for key in keys]

        if do_plot:
            if do_collapse_folds:
                plt.plot(xvals, mean_accs,
                         linestyle, label=label,
                         color=color,
                         markersize=markersize)
            else:
                nfolds = len(accs_dict[keys[0]])
                for fold in range(nfolds):
                    fold_acc = [norm_accs[key][fold] for key in keys]
                    plt.plot(xvals, fold_acc,
                             linestyle, label=label,
                             color=color,
                             markersize=markersize)

            labels = []
            for key in keys:
                labels.append(key[:2])
            plt.gca().set_xticks(xvals)
            plt.gca().set_xticklabels(labels)
            # plt.axhline(0.0, linestyle='--', color='k')
    # elif which_set == 4:         # deprecated?
    #     keys = np.sort(list(accs_dict.keys()))[1::2]
    #     mean_accs = [np.mean(accs_dict[key]) for key in keys]
    #     xvals = keys
    #     plt.plot(xvals, mean_accs,
    #             linestyle,
    #             label=label,
    #             color=color,
    #             markersize=markersize)
    #
    #     xlabels = []
    #     for key in keys:
    #         xlabels.append(key[0::2])
    #     plt.gca().set_xticklabels(xlabels, rotation=45)

    return mean_accs, xvals


def get_data_for_decoding(CT, opts, do_debug=False, random_split=True):
    """
    Preprocesses data for classification, according to various options.
    :param CT: A CosmosTraces object.
    :param opts: options dict.
        - decoding_set: int. ID # that refers to which set of labels to decode.
                        See code for the specifics of each ID. For example,
                        1 refers to decoding 4 classes:
                                spout1,spout2,spout3, and no-lick.
                        2 refers to decoding 2 classes:
                                lick vs. no-lick
                        3 refers to decoding 3 classes: NOT YET IMPLEMENTED.
                               for lick timepoints, spout1 vs spout2 vs spout3.
                        5 refers (I think) to latent spout position.
                        6 refers to decoding latent target spout position (i.e.
                                trial type). Decoding each frame.
                                 3 classes, [frames x spouts]
                        For each new type of experiment,
                        write a new decoding_set as opposed
                        to modifying a previous one.
        - train_feat: Input data from which to decode. 'spikes',
                                                       'spikes_binarized',
                                                       'smooth_spikes',
                                                       or 'fluor'.
        - train_frac, test_frac, valid_frac: float. Fraction of total data for
                                             training, test, validation.
        - remove_multi_licks: bool. Exclude time points at which licks on two
                                    or more spouts were detected.
        - bins_current: int. Set to 1 if you wish to include the frame of each
                            event that you are decoding.
        - bins_before: int. How many frames before each event based on which to
                            decode. Causal.
        - bins_after: int. How many frames after each event based on which to
                           decode. Acausal.
    """
    # Unpack parameter options.
    decoding_set = opts['decoding_set']
    train_feat = opts['train_feat']
    train_frac = opts['train_frac']
    test_frac = opts['test_frac']
    valid_frac = opts['valid_frac']
    remove_multi_licks = opts['remove_multi_licks']
    np.random.seed(opts['rand_seed'])

    # Extract relevant data from CT object.
    BD = CT.bd
    fps = CT.fps
    spout_lick_times = BD.spout_lick_times
    nframes = CT.C.shape[1]
    use_led_frames = True
    if use_led_frames:
        # Exclude final trial which is incomplete
        trial_onset_frames = CT.led_frames[:-1] - 1
    else:
        trial_onset_frames = CT.trial_onset_frames

    # Set input data from which to decode.
    if train_feat == 'spikes':
        neural_data = CT.S.T
        # neural_data = (neural_data > 0).astype(float)  ### Commented this on 20190814 (for resubmission), and add 'spikes_binarized' option
    if train_feat == 'spikes_binarized':
        neural_data = CT.S.T
        neural_data = (neural_data > 0).astype(float)  ### Commented this on 20190814 (for resubmission)
    if train_feat == 'smooth_spikes':
        neural_data = gaussian_filter1d(CT.S, 1.5, axis=1, mode='constant').T
    elif train_feat == 'fluor':
        # This is the denoised fluorescence, CT.F is the raw fluorescence
        neural_data = CT.C.T  # time x neuron

    # import pdb; pdb.set_trace()
    # Gather all licks.
    spout_lick_frames = defaultdict(list)
    spout_keys = [0, 2, 3]
    for spout in spout_keys:
        spout_trial_lick_times = spout_lick_times[spout]
        # for trial, lick_t in spout_trial_lick_times.items():
        for trial in range(len(trial_onset_frames)-1):
            if trial in spout_trial_lick_times.keys():
                lick_t = spout_trial_lick_times[trial]

                lick_frames = np.floor(lick_t*fps).astype(np.int)
                lick_frames += trial_onset_frames[trial].astype(np.int)
                try:
                    spout_lick_frames[spout].extend(lick_frames)
                except:
                    spout_lick_frames[spout].append(lick_frames)

    # Set target labels to decode.
    if decoding_set == 1:  # spout1 vs spout2 vs spout3 vs nolick
        Y_licks = np.zeros((nframes, len(spout_keys)))
        lick_idx = []
        for k, v in spout_lick_frames.items():
            Y_licks[v, spout_keys.index(k)] = 1
            lick_idx.extend(v)

        Y_full = np.zeros((Y_licks.shape[0], 1+Y_licks.shape[1]))
        Y_full[:, 1:] = Y_licks
        Y_full[np.where(Y_full.sum(1) == 0), 0] = 1
    elif decoding_set == 2:  # lick vs. no-lick
        Y_full = np.zeros((nframes, 2))
        for k, v in spout_lick_frames.items():  # For each spout.
            Y_full[v, 1] = 1

        Y_full[np.where(Y_full.sum(1) == 0), 0] = 1
        pass
    elif decoding_set == 3:  # spout1 vs spout2 vs spout3 on just lick times.

        # Y_licks = np.zeros((nframes, len(spout_keys)))
        # lick_idx = []
        # for k, v in spout_lick_frames.items():
        #     Y_licks[v, spout_keys.index(k)] = 1
        #     lick_idx.extend(v)
        #
        # import pdb; pdb.set_trace()

        pass
    elif decoding_set == 4:  # Trial type at different time points in trial?
        pass
    elif decoding_set == 5:
        # Latent spout position, size = trials x spouts
        Y_full = np.zeros((BD.ntrials, BD.nspouts))
        for idx, spout in enumerate(np.sort(np.unique(BD.spout_positions))):
            Y_full[:, idx] = np.array(BD.spout_positions == spout, dtype=float)
    elif decoding_set == 6:
        # Latent spout position, size  = frames x spouts
        Y_full = np.zeros((nframes, BD.nspouts))
        for spout_idx, spout in enumerate(
                np.sort(np.unique(BD.spout_positions))):
            for trial in range(len(trial_onset_frames)-1):
                interval = np.arange(trial_onset_frames[trial],
                                     trial_onset_frames[trial+1])
                if BD.spout_positions[trial] == spout:
                    Y_full[interval, spout_idx] = [1]*len(interval)
    elif decoding_set == 7:  # spout1 vs spout2 vs spout3, all timepoints.
        Y_licks = np.zeros((nframes, len(spout_keys)))
        lick_idx = []
        for k, v in spout_lick_frames.items():
            Y_licks[v, spout_keys.index(k)] = 1
            lick_idx.extend(v)
        Y_full = Y_licks
    elif decoding_set == 8:  # latent spout1 vs latent spout2 vs latent spout3
                            # just pre-odor, and just clean trials.

        # GOAL: [
        # Y_full = np.zeros((Y_licks.shape[0], 1+Y_licks.shape[1]))
        pass

    else:
        raise('Decoding_set #{} has not yet been implemented.'
              .format(decoding_set))

    # Generate assignment of each timepoint to train/test/validation datasets.
    # Randomize trials, but keep each trial intact.
    # Assign a random subset of trials to train/test/validate, according
    # to the specific train_frac/test_frac/valid_frac.

    # First, determine the trial number of to each time point.
    trial_labels = np.zeros((nframes))
    for i in range(len(trial_onset_frames)-1):
        trial_labels[trial_onset_frames[i]:trial_onset_frames[i+1]] = i
    trial_labels[trial_onset_frames[-1]:] = len(trial_onset_frames)-1
    trials = np.unique(trial_labels)

    # Also determine the frame within a trial of each time point.
    # Can use this to potentially just filter out a subset
    # of timepoints.
    # Remember to exclude timepoints where the label == 0 (the first frame).
    within_trial_labels = np.zeros((nframes))
    for i in range(len(trial_onset_frames)-1):
        inds = np.arange(trial_onset_frames[i], trial_onset_frames[i+1])
        if np.all(inds < nframes):
            within_trial_labels[inds] = inds - inds[0]

    # Optionally only include clean trials
    # (i.e. where the mouse just licks one direction)
    if opts.get('just_clean_trials') is not None:
        if opts['just_clean_trials'] > 0:
            clean_trials = BD.get_clean_trials(opts['just_clean_trials'])
            trials = clean_trials

    # Optionally only include trials where the mouse
    # did not lick before the odor.
    if opts.get('just_no_preodor_lick_trials') is not None:
        if opts['just_no_preodor_lick_trials']:
            no_preodor_lick_trials = BD.get_no_preodor_lick_trials(max_licks=0)
            trials = np.intersect1d(trials, no_preodor_lick_trials)

    if do_debug:
        BD.plot_lick_times(trials_subset=trials)

    # Set up the train/test/valid sets so that each one has at least
    # one of each trial type, in each fold.
    if opts.get('ensure_all_trial_types_present') is not None:
        if opts['ensure_all_trial_types_present']:
            trial_types = BD.spout_positions[trials]
            np.random.seed(opts['rand_seed'])
            reordered_trials = evenly_space_trial_types(trials, trial_types,
                                                        do_random_permute=True)
            trials = reordered_trials

            if do_debug:
                plt.figure()
                plt.title('Reordering of trials to evenly space trial types')
                plt.plot(BD.spout_positions[trials])
                plt.ylabel('Target spout')
    else:
        # Temporally shuffle up the trials that go into the different groups
        if random_split:
            np.random.seed(opts['rand_seed'])
            trials = np.random.permutation(trials)

    if opts.get('nfolds') is not None and opts.get('fold_number') is not None:
        shift = int(len(trials)*opts['fold_number']/opts['nfolds'])
        trials = np.roll(trials, shift)

    ind1 = int(len(trials)*train_frac)
    ind2 = int(len(trials)*test_frac)
    ind3 = int(len(trials)*valid_frac)
    train_trials = trials[:ind1]
    test_trials = trials[ind1:ind1+ind2]
    valid_trials = trials[ind1+ind2:ind1+ind2+ind3]

    if do_debug:
        plt.figure()
        plt.subplot(131)
        plt.title('Train')
        plt.plot(BD.spout_positions[train_trials.astype(int)])
        plt.ylabel('Target spout')
        plt.subplot(132)
        plt.title('Test')
        plt.plot(BD.spout_positions[test_trials.astype(int)])
        plt.subplot(133)
        plt.title('Valid')
        plt.plot(BD.spout_positions[valid_trials.astype(int)])

        BD.plot_lick_times(trials_subset=test_trials)
        plt.suptitle('Test trials')
        BD.plot_lick_times(trials_subset=train_trials)
        plt.suptitle('Train trials')
        BD.plot_lick_times(trials_subset=valid_trials)
        plt.suptitle('Valid trials')

    # Now assign each timepoint to train/test/validate based on its trial num.
    train_inds = np.isin(trial_labels, train_trials)
    test_inds = np.isin(trial_labels, test_trials)
    valid_inds = np.isin(trial_labels, valid_trials)

    if opts.get('exclude_no_licks') is not None and opts.get(
            'exclude_no_licks'):
        no_lick_frames = np.sum(Y_licks, axis=1) == 0
        train_inds = np.logical_and(train_inds, ~no_lick_frames)
        test_inds = np.logical_and(test_inds, ~no_lick_frames)
        valid_inds = np.logical_and(valid_inds, ~no_lick_frames)

    # Optionally restrict included frames to a certain range within each trial.
    if 'within_trial_frame_range' in opts.keys():
        fr = opts['within_trial_frame_range']
        if fr is not None:
            includable_frames = np.logical_and(within_trial_labels > fr[0]+1,
                                               within_trial_labels < fr[1])
            train_inds = np.logical_and(train_inds, includable_frames)
            test_inds = np.logical_and(test_inds, includable_frames)
            valid_inds = np.logical_and(valid_inds, includable_frames)

    train_trial_labels = trial_labels[train_inds]
    test_trial_labels = trial_labels[test_inds]
    valid_trial_labels = trial_labels[valid_inds]

    print('Data split into train, test, validate sets.',
          '\nFractions: Train {:.3f}, Test {:.3f} Valid {:.3f}'.format(
            np.sum(train_inds)/len(train_inds),
            np.sum(test_inds)/len(test_inds),
            np.sum(valid_inds)/len(valid_inds)))

    if do_debug:
        plt.figure()
        plt.plot(within_trial_labels[train_inds][:1000])
        plt.ylabel('Frame within trial')
        plt.title('Which frames within trials are included')

    # Split the data into train, validation, test sets.
    splitopts = default_splitopts
    splitopts['bins_before'] = opts['bins_before']
    splitopts['bins_current'] = opts['bins_current']
    splitopts['bins_after'] = opts['bins_after']
    splitopts['train_inds'] = train_inds
    splitopts['test_inds'] = test_inds
    splitopts['valid_inds'] = valid_inds
    splitopts['standardize_X'] = opts['standardize_X']
    splitopts['center_Y'] = opts['center_Y']

    if decoding_set == 5:
        trial_frames = {}
        trial_frames['train'] = np.unique(
            np.array(trial_labels[train_inds], dtype=int))
        trial_frames['test'] = np.unique(
            np.array(trial_labels[test_inds], dtype=int))
        trial_frames['valid'] = np.unique(
            np.array(trial_labels[valid_inds], dtype=int))
        trial_frames['train_frames'] = trial_onset_frames[
            trial_frames['train']]
        trial_frames['test_frames'] = trial_onset_frames[
            trial_frames['test']]
        trial_frames['valid_frames'] = trial_onset_frames[
            trial_frames['valid']]
        data_split = split_dataset_trials(neural_data, Y_full,
                                          trial_frames, opts=opts)
    else:
        data_split = split_dataset(neural_data, Y_full,
                                   splitopts, do_debug=do_debug)

    # import pdb; pdb.set_trace()
    # For simplicity, remove frames where multiple spouts were licked.
    if remove_multi_licks:
        bad_ix = np.where(data_split['Y_train'].sum(1) > 1)[0]
        data_split['Y_train'] = np.delete(data_split['Y_train'],
                                          bad_ix, axis=0)
        data_split['X_train'] = np.delete(data_split['X_train'],
                                          bad_ix, axis=0)

    data_split['train_trial_labels'] = train_trial_labels.astype(int)
    data_split['test_trial_labels'] = test_trial_labels.astype(int)
    data_split['valid_trial_labels'] = valid_trial_labels.astype(int)
    data_split['train_trials'] = train_trials
    data_split['test_trials'] = test_trials
    data_split['valid_trials'] = valid_trials

    if do_debug:
        plt.subplot(131)
        plt.imshow(data_split['Y_valid'], aspect='auto')
        plt.title('Y valid')
        plt.subplot(132)
        plt.imshow(data_split['Y_test'], aspect='auto')
        plt.title('Y test')
        plt.subplot(133)
        plt.imshow(data_split['Y_train'], aspect='auto')
        plt.title('Y train')

    return data_split


def verify_data_alignment(trial_ind, which_cell, CT, data_split, data_opts):
    """
    Generate a plot to ensure that partitioned train/test/validate
    data matches the raw neural and bpod behavioral data.

    :param trial_ind: int. which trial to plot
    :param which_cell: int. ID of the cell from which to plot neural activity.
    :return:
    """

    # TODO: Potentially include some error catching/testing as well.
    spout_lick_rates = CT.bd.spout_lick_rates

    plt.figure(figsize=(15, 15))
    trial = np.unique(data_split['train_trial_labels'])[trial_ind]
    plt.plot(spout_lick_rates[0][trial] /
             np.max(spout_lick_rates[0][trial]), 'go', label='spout 1')
    plt.plot(spout_lick_rates[1][trial] /
             np.max(spout_lick_rates[1][trial]), 'co', label='spout 2')
    plt.plot(spout_lick_rates[2][trial] /
             np.max(spout_lick_rates[2][trial]), 'mo', label='spout 3')
    plt.plot(spout_lick_rates[3][trial] /
             np.max(spout_lick_rates[3][trial]), 'ko', label='spout 4')

    ind = np.where(data_split['train_trial_labels'] == trial)[0][0]
    shift = data_opts['bins_before']

    for which_spout, color in zip([1, 2, 3], ['g', 'm', 'k']):
        plt.plot(
            data_split['Y_train'][ind - shift:ind + 300 - shift, which_spout],
            color, label='Y_train_{}'.format(which_spout))

    plt.plot(
        np.diff(data_split['train_trial_labels'][ind - 1:ind + 300]) > 0,
        'y', label='Trial_start')

    plt.plot(data_split['X_train'][ind + 1:ind + 300 + 1, 0, which_cell], 'r',
             label='X_train')
    plt.plot(CT.St[which_cell, :, trial], 'ro', label='CT.St')
    plt.legend()
    plt.title('For neuron {}, trial {}'.format(which_cell, trial_ind))


def get_source_discriminativity(data_split, CT, data_opts, decoding_save_dir,
                                do_plot=False):
    """
    Rank the neural sources based on the ability
    of their activity to discriminate features of the licking/labels.

    Specifically, compute an omnibus p-value using Kruskal-Wallis H-test
    (non-parametric version of ANOVA) that the population
    medians of neural activity over all timepoints when the mouse
    is engaging in each type of behavior (nolick, lick1, lick2, lick3)
    are different.

    Saveout the discriminativity (i.e. pvalues) to
    [decoding_save_dir, '_ordering'].

    :return: p_ordering: Ordering of sources by their discriminativity.
    """

    X = data_split['X_train']  # Neural data.
    if len(data_split['Y_train'].shape) > 1:
        y = np.argmax(data_split['Y_train'], axis=1)
    else:
        y = data_split['Y_train']
    nbins = X.shape[1]  # The number of neural timepoints surrounding each lick
    # event that are to be included in the decoding.

    pvals = np.zeros((CT.ncells, nbins))
    for bin in range(nbins):
        for cell in range(CT.ncells):
            if np.max(X[:, bin, cell] > 0.1):
                c = cell

                if data_opts['decoding_set'] == 1:  # 4-way classification
                    f, p = scipy.stats.kruskal(X[np.where(y == 0)[0], bin, c],
                                               X[np.where(y == 1)[0], bin, c],
                                               X[np.where(y == 2)[0], bin, c],
                                               X[np.where(y == 3)[0], bin, c])
                elif data_opts['decoding_set'] == 2:  # 2-way classification
                    f, p = scipy.stats.kruskal(X[np.where(y == 0)[0], bin, c],
                                               X[np.where(y == 1)[0], bin, c])

                elif data_opts['decoding_set'] == 6:  # 3-way classification
                        f, p = scipy.stats.kruskal(X[np.where(y == 0)[0], bin,
                                                     c],
                                                   X[np.where(y == 1)[0], bin,
                                                     c],
                                                   X[np.where(y == 2)[0], bin,
                                                     c])
                elif data_opts['decoding_set'] == 7:  # 3-way classification
                        f, p = scipy.stats.kruskal(X[np.where(y == 0)[0], bin,
                                                     c],
                                                   X[np.where(y == 1)[0], bin,
                                                     c],
                                                   X[np.where(y == 2)[0], bin,
                                                     c])
            else:
                p = 1
            pvals[cell, bin] = p

    minpvals = np.min(pvals, axis=1)  # Choose the best pvalue across time
    ordering_metric = minpvals
    p_ordering = np.argsort(ordering_metric)

    ordering_file = os.path.join(decoding_save_dir, '_ordering')
    ordering_dict = {'minpvals': minpvals}
    with open(ordering_file, "wb") as handle:
        pickle.dump(ordering_dict, handle)

    if do_plot:
        plt.figure, plt.plot(np.sort(np.min(pvals, axis=1)))
        plt.ylabel('pvalue')
        plt.xlabel('Order cell #')
        plt.axhline([0.05])

        # Plot the locations of the cells with different cutoffs
        ncells = 50
        plt.figure()
        CT.centroids_on_atlas(np.ones(ncells),
                              p_ordering[:ncells], max_radius=10)
        plt.title('Best {} cells'.format(ncells))

        plt.figure()
        CT.centroids_on_atlas(np.ones(ncells),
                              p_ordering[-ncells:], max_radius=10)
        plt.title('Worst {} cells'.format(ncells))

    # TODO: Write unit test checking that this makes sense.

    # return p_ordering
    return p_ordering, ordering_metric


def get_classification_summary_cm(cm):
    """
    Using confusion matrix,
    returns a dict containing, for each class,
    the number of True Positives, False Positives,
    Actual Positives, True Negatives, False Negatives,
    Actual Negatives.

    :param cm: confusion matrix. Sum across each row is the number of
               actual positives in that class.
    """
    TP = dict()
    FP = dict()
    P = dict()
    TN = dict()
    FN = dict()
    N = dict()
    nclasses = cm.shape[0]
    for c in range(nclasses):
        TP[c] = cm[c, c]
        FP[c] = np.sum(cm[:, c]) - TP[c]
        P[c] = np.sum(cm[c, :])
        TN[c] = np.sum(cm) - np.sum(cm[c, :]) - np.sum(cm[:, c]) + cm[c, c]
        FN[c] = np.sum(cm[c, :]) - TP[c]
        N[c] = np.sum(cm) - np.sum(cm[c, :])

    summary = dict()
    summary['TP'] = TP
    summary['FP'] = FP
    summary['P'] = P
    summary['TN'] = TN
    summary['FN'] = FN
    summary['N'] = N

    return summary


def get_classification_summary(licks_pred, licks_true):
    """
    Returns a dict containing, for each class,
    the number of True Positives, False Positives,
    Actual Positives, True Negatives, False Negatives,
    Actual Negatives.

    :param licks_pred: array. predictions.
    :param licks_true: array. true values.
    """

    TP = dict()
    FP = dict()
    P = dict()
    TN = dict()
    FN = dict()
    N = dict()
    classes = np.unique(licks_true)
    for c in classes:
        TP[c] = np.sum(np.logical_and(licks_pred == c, licks_true == c))
        FP[c] = np.sum(np.logical_and(licks_pred == c, licks_true != c))
        P[c] = np.sum(licks_true == c)
        TN[c] = np.sum(np.logical_and(licks_pred != c, licks_true != c))
        FN[c] = np.sum(np.logical_and(licks_pred != c, licks_true == c))
        N[c] = np.sum(licks_true != c)

    summary = dict()
    summary['TP'] = TP
    summary['FP'] = FP
    summary['P'] = P
    summary['TN'] = TN
    summary['FN'] = FN
    summary['N'] = N

    return summary


def compute_classification_evaluations(classification_summary, do_print=False):
    """
    Compute various evaluations of the classification performance based
    on the summary numbers.

    :param classification_summary:  A dict containing
                                    the number of True Positives, False Pos,
                                    Actual Positives, True Negatives,
                                    False Negatives, Actual Negatives
                                    (each a dict that contains entries
                                    for each classification class).
    :param do_print: bool. Print the results in a legible format.
    :return evaluations: A dict containing Sensitivity, Specificity,
                         Precision, Accuracy,
                         and Informedness (each a dict that contains entries
                         for each
                         classification class).
    """

    m = defaultdict(dict)
    cs = classification_summary
    TP = cs['TP']
    FP = cs['FP']
    P = cs['P']
    TN = cs['TN']
    FN = cs['FN']
    N = cs['N']
    classes = list(TP.keys())
    for c in classes:
        m['sensitivity'][c] = TP[c] / P[c]
        m['specificity'][c] = TN[c] / N[c]
        m['precision'][c] = TP[c] / (TP[c] + FP[c])
        m['accuracy'][c] = (TP[c] + TN[c]) / (P[c] + N[c])
        m['informedness'][c] = m['sensitivity'][c] + m['specificity'][c] - 1

    if do_print:
        for key in m.keys():
            print(key + ': \t' +
                  '{:.3f}'.format(np.mean([x for x in m[key].values()]), 3) +
                  ', \t ' + str([round(x, 3) for x in m[key].values()]))

    evaluations = m
    return evaluations


def multi_class_roc_auc(y_true, y_pred, do_plot=False):
    """
    Compute the ROC curve for each class
    in a multi-class classification.
    Also compute the area under that curve
    for each class. If you average those
    results together, you should obtain
    the 'macro average' (in the terminology
    of sklearn).

    This uses a one vs. all strategy for computing each
    roc curve.

    y_true: [ndatapoints x nclasses]. One-hot
            representation of the true class labels.
    y_pred: [ndatapoints x nclasses]. The prediction
            probability associated with each class for
            each datapoint.
    """
    from sklearn.metrics import roc_curve, auc

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    nclasses = y_true.shape[1]
    for c in range(nclasses):
        fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_pred[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    if do_plot:
        for i in fpr.keys():
            plt.plot(fpr[i], tpr[i], label=i)
        plt.legend()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('One vs. all ROC')
    return fpr, tpr, roc_auc



def get_ROC_curves(expts_info, expt_sets, data_dir, fig_save_dir,
                   is_shuff=False,
                   do_plot_confusion_matrices=True,
                   do_plot_individual_roc_curves=True,
                   ):
    """
    Get the true-positive rate vs. false-positive rate
    for the provided sets of decoding experiments.
    This enables you to plot the receiver-operating-curves.

    Additionally, for each set of decoding experiments,
    plot the confusion matrix.

    :param expts_info: dict. keys correspond to indices in 'expt_sets'.
                       values are the output from lick_decoder.decode_licks(),
                       i.e. {'id': 19, 'expt_nums': np.arange(1021, 1025),
                             'expt_type': 'neuron_set', 'info':10}
    :param expt_sets: a list of indices into expts_info dict.
    :param data_dir: path to directory containing data, i.e. '/Dropbox/cosmos_data/'
    :param fig_save_dir: path to directory for saving plots.
    :param is_shuff: bool. Are the decoding experiments a shuffled control?
    :param do_plot_confusion_matrices: bool. Whether to plot the mean confusion matrix
                                       for each experiment set.
    :param do_plot_individual_roc_curves: bool. Whether to plot the individual
                                          roc curves for each fold of each experiment.
    :return: all_mean_fpr - list of arrays, false-positive-rate as the classification
                            threshold is varied.
             all_mean_tpr - list of arrays. true-positive-rate as the classification
                            threshold is varied.
             all_labels - the name corresponding to each entry in all_mean_fpr and all_mean_tpr
    """

    all_mean_fpr = []
    all_mean_tpr = []
    all_labels = []
    plt.figure(expt_sets[0])
    for expt_id in expt_sets:  # Load each decoding experiment group (i.e. for each mouse)
        expt_group = expts_info[expt_id]
        decoding_load_dir = os.path.join(data_dir, 'decoding_results',
                                         str(expt_group['id']))
        expt_nums = expt_group['expt_nums']

        if not is_shuff:
            all_labels.append(expt_group['id'])
        fold_auc = []
        fold_informedness = []
        fold_fpr = defaultdict(list)
        fold_tpr = defaultdict(list)
        fold_cm = []
        for expt_num in expt_nums:  # Load each fold in a decoding experiment group
            expt_file = os.path.join(decoding_load_dir,
                                     'expt_{:06}'.format(expt_num))

            with open(expt_file + '_opts', "rb") as handle:
                opts = pickle.load(handle)
            with open(expt_file + '_decode_summary', "rb") as handle:
                decode_summary = pickle.load(handle)

            y_pred = decode_summary['y_pred']
            licks_true = decode_summary['licks_true']
            licks_pred = decode_summary['licks_pred']

            do_recompute_y_true = True  # There are two advantages to recomputing:
                                        #    -will not have any multi-label datapoints,
                                        #    -also wasn't saved for prev classification
            if do_recompute_y_true:
                y_true = dict()
                for key in licks_true.keys():
                    y_true[key] = keras.utils.to_categorical(
                        licks_true[key])
            else:
                y_true = decode_summary['y_true']

            # Compute false-positive and true-positive rates
            # using the test dataset.
            dset = 'test'
            cm = confusion_matrix(licks_true[dset], licks_pred[dset])
            s = get_classification_summary(licks_pred[dset],
                                           licks_true[dset])
            mm = compute_classification_evaluations(s, do_print=False)
            fpr, tpr, roc_auc = multi_class_roc_auc(y_true[dset],
                                                    y_pred[dset],
                                                    do_plot=False)

            fold_cm.append(cm)
            fold_informedness.append(
                np.mean([x for x in mm['informedness'].values()]))
            fold_auc.append(np.mean([x for x in roc_auc.values()]))
            for key in fpr.keys():
                fpr_new = np.linspace(0, 1, 2000)
                tpr_new = resample(fpr[key], tpr[key], fpr_new)
                fold_fpr[key].append(fpr_new)
                fold_tpr[key].append(tpr_new)

        mean_fpr = np.mean(np.vstack([fold_fpr[key] for key in fold_fpr.keys()]),
                           axis=0)
        mean_tpr = np.mean(np.vstack([fold_tpr[key] for key in fold_tpr.keys()]),
                           axis=0)

        all_mean_fpr.append(mean_fpr)
        all_mean_tpr.append(mean_tpr)

        # Plot individual ROC curves for each spout and each fold
        if do_plot_individual_roc_curves:
            plt.figure(expt_sets[0])
            spout_colors = [(.2, .2, .2, 1), 'orange', 'c', 'r']
            for key in fold_fpr.keys():
                if not is_shuff:
                    markerstyle = '-'
                    label = 'spout ' + str(key)
                else:
                    markerstyle = '--'
                    label = None
                plt.plot(np.vstack(fold_fpr[key]).T,
                         np.vstack(fold_tpr[key]).T, '-',
                         color=spout_colors[key], alpha=0.3)
                tpr_mean = np.mean(np.vstack(fold_tpr[key]), axis=0)
                fpr_mean = np.mean(np.vstack(fold_fpr[key]), axis=0)
                plt.plot(fpr_mean, tpr_mean, markerstyle,
                         color=spout_colors[key], linewidth=1,
                         label=label)
            plt.legend()
            plt.gca().legend(bbox_to_anchor=(1.5, 1.05))
            plt.annotate('shuffled', xy=[0.55, 0.45],
                         xytext=[0.75, 0.25],
                         arrowprops={'arrowstyle': '->'})
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            if not is_shuff:
                plt.title(
                    'One vs. all ROC, for mouse {}, AUC: {:.2f}'.format(
                        expt_group['id'],
                        np.mean(fold_auc)))
            print('AUC: {:.2f}'.format(np.mean(fold_auc)))

            # Control size of figure in inches
            plt.gcf().set_size_inches(w=2.5,
                                      h=1.5)

            if fig_save_dir is not None:
                plt.savefig(fig_save_dir + 'roc_curves_' + str(
                    expts_info[expt_id]['id']) + '.pdf',
                            transparent=True, rasterized=True, dpi=600)

        print('AUC: {:.2f}'.format(np.mean(fold_auc)))

        # Plot normalized confusion matrices for each mouse
        if do_plot_confusion_matrices:
            plt.figure(expt_id + 100)
            fold_avg = fold_cm[0]
            for fold in range(1, len(fold_cm)):
                fold_avg += fold_cm[fold]
            fold_avg = fold_avg / len(fold_cm)

            cm_avg = fold_avg / np.sum(fold_avg, axis=1)[:, np.newaxis]
            plt.figure()
            plot_confusion_matrix(cm_avg)
            plt.title(
                'Normalized confusion matrix, for mouse {}, shuff: {}'.format(
                    expt_group['id'], is_shuff))
            plt.gcf().set_size_inches(w=3,
                                      h=3)  # Control size of figure in inches
            if fig_save_dir is not None:
                plt.savefig(fig_save_dir + 'confusion_mat_' + str(
                            expts_info[expt_id]['id'])
                            + '_shuff{}'.format(int(is_shuff)) + '.pdf',
                            transparent=True, rasterized=True, dpi=600)

    return all_mean_fpr, all_mean_tpr, all_labels


def plot_discrimination_capacity_ordering(concluded_expts,
                                          expt_ids,
                                          data_dir,
                                          mouse_colors_dict,
                                          fig_save_dir):
    """
    For the decoding experiments, sources were ordered
    according their degree of discrimination capacity.
    Specifically, this was an anova-type test for each
    source to determine how much information it contained
    for discriminating any of the labels to be decoded.
    This function plots these discrimination capacities,
    for a specified set of decoding experiments.

    :param concluded_expts: dict. keys correspond to indices in 'expt_sets'.
                            values are the output from lick_decoder.decode_licks(),
                            i.e. {'id': 19, 'expt_nums': np.arange(1021, 1025),
                            'expt_type': 'neuron_set', 'info':10}
    :param expt_ids: keys into concluded_expts dict.
    :param data_dir: path to directory containing data, i.e. '/Dropbox/cosmos_data/'
    :param mouse_colors_dict: dict, keys are mouse experiment ids,
                                values are [#color, mouse_name]
    :param fig_save_dir: path to directory for saving plots.
    :return:
    """

    # Organize discrimination capacity ordering across experiment sets.
    all_ordering = []
    labels = []
    max_len = 0
    for expt_id in expt_ids:
        expt_group = concluded_expts[expt_id]
        decoding_load_dir = os.path.join(data_dir, 'decoding_results',
                                         str(expt_group['id']))

        use_legacy = False  # Set this to false...
        if use_legacy:
            ordering_file = os.path.join(decoding_load_dir, '_ordering')
            with open(ordering_file, "rb") as handle:
                ordering_dict = pickle.load(handle)
            ordering = ordering_dict['minpvals']
        else:
            expt_num = expt_group['expt_nums'][0]
            expt_file = os.path.join(decoding_load_dir,
                                     'expt_{:06}'.format(expt_num))

            with open(expt_file + '_opts', "rb") as handle:
                opts = pickle.load(handle)
            ordering = opts['expt_param']['minpvals']

        all_ordering.append(ordering)
        labels.append(expt_group['id'])
        if np.max(len(ordering)) > max_len:
            max_len = np.max(len(ordering))
            print(max_len)

    # Now plot all discrimination capacities together.
    # Add a dashed line to indicate a significance cutoff.
    sig_thresh = 0.05
    fraction_sig = []
    for ind, ordering in enumerate(all_ordering):

        # This is for Bonferroni
        # sig_thresh = 0.05 / max_len

        # Get FDR corrected p-values
        vals = mt.multipletests(ordering, alpha=0.05, method='fdr_bh')
        ordering = vals[1]
        fraction_sig.append(np.sum(ordering < sig_thresh) / len(ordering))
        plt.plot(np.log10(np.sort(ordering)),
                 label=mouse_colors_dict[labels[ind]][1],
                 color=mouse_colors_dict[labels[ind]][0])
        plt.ylim([-15, 0])
        plt.axhline(np.log10(sig_thresh), color=(0, 0, 0, 1), dashes=[2, 2])

    plt.xticks([0, 500, 1000])
    plt.ylabel('log p-value')
    plt.xlabel('source #')
    legend = plt.legend(frameon=False,
                        bbox_to_anchor=(1.5, 0.5), handlelength=0.5,
                        handletextpad=0.3, labelspacing=0.2)
    plt.title(
        'Fraction significant: {:.2f} +/- {:.2f}'.format(np.mean(fraction_sig),
                                                         scipy.stats.sem(
                                                             fraction_sig)))
    plt.gcf().set_size_inches(w=2, h=1.5)  # Control size of figure in inches
    if fig_save_dir is not None:
        plt.savefig(fig_save_dir + 'subset_selection_ranking.pdf',
                    transparent=True, rasterized=True, dpi=600)

    return all_ordering, labels


def plot_significant_discrimination_capacity_sources(concluded_expts,
                                                     expt_ids,
                                                     all_ordering,
                                                     allCT,
                                                     mouse_colors_dict,
                                                     fig_save_dir,
                                                     num_cells=None):
    """
      For the decoding experiments, sources were ordered
      according their degree of discrimination capacity.
      Specifically, this was an anova-type test for each
      source to determine how much information it contained
      for discriminating any of the labels to be decoded.
      This function plots these discrimination capacities,
      for a specified set of decoding experiments.

      :param concluded_expts: dict. keys correspond to indices in 'expt_sets'.
                              values are the output from lick_decoder.decode_licks(),
                              i.e. {'id': 19, 'expt_nums': np.arange(1021, 1025),
                              'expt_type': 'neuron_set', 'info':10}
      :param expt_ids: keys into concluded_expts dict.
      :param data_dir: path to directory containing data, i.e. '/Dropbox/cosmos_data/'
      :param mouse_colors_dict: dict, keys are mouse experiment ids,
                                values are [#color, mouse_name]
      :param fig_save_dir: path to directory for saving plots.
      :return:
      """
    for ind, expt_id in enumerate(expt_ids):
        ordering = all_ordering[ind]
        CT = allCT[ind]
        if num_cells is None:
            ncells = np.where(ordering < 0.05 / len(ordering))[0].shape[0]
        else:
            ncells = num_cells
        ordering_inds = np.argsort(ordering)

        atlas_coords = utils.transform_centroids_to_atlas(CT.centroids,
                                                          CT.atlas_tform)
        cell_ids = np.arange(atlas_coords.shape[0])
        # color = mouse_colors[ind]
        color = mouse_colors_dict[concluded_expts[expt_id]['id']][0]
        mouse_name = mouse_colors_dict[concluded_expts[expt_id]['id']][1]
        if isinstance(color, str):
            color = np.array(
                [int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
            color = np.append(color, 1.0)
        RGBA = np.tile(np.array(color), (cell_ids.shape[0], 1))
        background_color = np.array([0.75, 0.75, 0.75, 0.9])
        RGBA[ordering_inds[ncells:], :] = background_color

        plt.figure()
        utils.centroids_on_atlas(RGBA, cell_ids, atlas_coords, None,
                                 max_radius=2,
                                 rotate90=True)
        plt.title('{}: best {} cells'.format(mouse_name, ncells))
        plt.xticks([])
        plt.yticks([])
        plt.gcf().set_size_inches(w=1.5,
                                  h=1.5)  # Control size of figure in inches
        fname = fig_save_dir + 'significant_cells_' + str(
            concluded_expts[expt_id]['id']) + '.pdf'
        print(fname)
        plt.savefig(fname,
                    transparent=True, rasterized=True, dpi=600)


def plot_discrimination_capacity_of_sources(concluded_expts,
                                             expt_ids,
                                             all_ordering,
                                             allCT,
                                             mouse_colors_dict,
                                             fig_save_dir,
                                             num_cells=None):
    """
      For the decoding experiments, sources were ordered
      according their degree of discrimination capacity.
      Specifically, this was an anova-type test for each
      source to determine how much information it contained
      for discriminating any of the labels to be decoded.
      This function plots these discrimination capacities,
      for a specified set of decoding experiments.

      :param concluded_expts: dict. keys correspond to indices in 'expt_sets'.
                              values are the output from lick_decoder.decode_licks(),
                              i.e. {'id': 19, 'expt_nums': np.arange(1021, 1025),
                              'expt_type': 'neuron_set', 'info':10}
      :param expt_ids: keys into concluded_expts dict.
      :param data_dir: path to directory containing data, i.e. '/Dropbox/cosmos_data/'
      :param mouse_colors_dict: dict, keys are mouse experiment ids,
                                values are [#color, mouse_name]
      :param fig_save_dir: path to directory for saving plots.
      :return:
      """
    for ind, expt_id in enumerate(expt_ids):
        vals = -np.log10(all_ordering[ind])
        vals = vals / np.max(vals)
        CT = allCT[ind]

        color = mouse_colors_dict[concluded_expts[expt_id]['id']][0]
        mouse_name = mouse_colors_dict[concluded_expts[expt_id]['id']][1]

        atlas_coords = utils.transform_centroids_to_atlas(CT.centroids,
                                                          CT.atlas_tform)
        cell_ids = np.arange(atlas_coords.shape[0])
        plt.figure()
        utils.centroids_on_atlas(vals, cell_ids, atlas_coords, None,
                                 max_radius=10,
                                 rotate90=True,
                                 cmap='viridis')
        plt.title('{}'.format(mouse_name))
        plt.xticks([])
        plt.yticks([])
        plt.gcf().set_size_inches(w=1.5,
                                  h=1.5)  # Control size of figure in inches
        fname = fig_save_dir + 'discrimination_capacity_' + str(
            concluded_expts[expt_id]['id']) + '.pdf'
        print(fname)
        plt.savefig(fname,
                    transparent=True, rasterized=True, dpi=600)


def load_individual_decoding_experiment(CT_load, decoding_load_dir,
                                        expt_num, load_dataset_id,
                                        do_plot=False, fig_save_dir=None):
    """
    Load the dataset and results of a decoding experiment.
    :param CT_load: CosmosTraces object.
    :param decoding_load_dir: Path to directory containing the saved results
                             of the decoding experiment.
    :param expt_num: the id of the decoding experiment
    :param load_dataset_id: the id of the mouse/dataset
    :param do_plot: bool. Whether to plot summary of decoding.
    :param fig_save_dir: string. If None, then does not save plot.
    :return: ds - the dataset actually used for decoding
             decode_summary - the summarize results of the decoding
    """
    expt_file = os.path.join(decoding_load_dir, 'expt_{:06}'.format(expt_num))

    # Load trained decoding model and training history.
    model = keras.models.load_model(expt_file + '_model')
    with open(expt_file + '_history', "rb") as handle:
        history = pickle.load(handle)
    with open(expt_file + '_opts', "rb") as handle:
        opts = pickle.load(handle)

    # Regenerate data subsets.
    data_split = get_data_for_decoding(CT_load, opts['data_opts'],
                                       do_debug=False)
    print('X_train:', data_split['X_train'].shape,
          'X_test:', data_split['X_test'].shape)
    ds = set_neurons_for_decoding(data_split, opts['neuron_opts'], CT_load,
                                  do_plot=True)
    sample_weights = get_sample_weights(ds, ['train', 'valid', 'test'])
    if opts['neuron_opts']['do_pca']:
        ds = pca_project(ds, opts['neuron_opts']['pca_components'])

    decode_summary = summarize_decoding_results(ds, model)

    # Plot the loaded experiment
    if do_plot:
        plot_decoding_summary(expt_num, opts['train_opts'], decode_summary,
                              CT_load, opts['neuron_opts'], ds, history,
                              model.model.count_params())

        if fig_save_dir is not None:
            plt.gcf().set_size_inches(w=7, h=3)  # Control size of figure in inches
            plt.savefig(
                fig_save_dir + 'decode_summary_' + str(load_dataset_id) + '_' + str(
                    expt_num) + '.pdf',
                transparent=True, rasterized=True, dpi=600)

    return ds, decode_summary


def plot_example_decoded_licks(CT_load, ds, decode_summary,
                               start_trial, end_trial,
                               dataset_id, expt_num, fig_save_dir):
    """

    :param CT_load: CosmosTraces dataset.
    :param ds: dict containing all of the data used for decoding.
    :param decode_summary: dict containg summarized results of decoding.
    :param start_trial: int. Trial to start plotting example decoded licks.
    :param end_trial: int. Trial to end plotting example decoded licks.
    :param dataset_id: int. The id of the dataset being plotted.
    :param expt_num: int. The id of the decoding experiment being plotted.
    :param fig_save_dir: path for saving the figure.
    :return: Nothing
    """

    data_subset = 'test'  # Plot decoded results on the test dataset.
    idx = np.arange(0, 10000)
    dt = CT_load.dt
    t = dt * idx

    trial_frames = np.where(np.diff(ds['trial_labels'][data_subset][idx]) > 0)[0]
    trial_starts = dt * trial_frames

    plt.figure(figsize=(20, 20))
    plt.plot(t - trial_starts[start_trial],
             decode_summary['licks_pred'][data_subset][idx], '.',
             color='#be29ec', label='pred', markersize=1.5)
    plt.plot(t - trial_starts[start_trial],
             decode_summary['licks_true'][data_subset][idx] + 0.2, '.',
             color='#4cbb17', label='true', markersize=1.5)
    [plt.axvline(x - trial_starts[start_trial], color='k') for x in
     trial_starts]
    plt.xlim([0, trial_starts[end_trial] - trial_starts[start_trial]])
    plt.ylim([0.5, 3.5])  # Exclude the no-lick cases

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    legend = plt.legend(frameon=False, loc='center left',
                        bbox_to_anchor=(1, 0.5), handlelength=0.5,
                        handletextpad=0.3, labelspacing=0.2)
    plt.xlabel('Time [s]')
    plt.ylabel('Spout #')

    if fig_save_dir is not None:
        plt.gcf().set_size_inches(w=6, h=1.5)  # Control size of figure in inches
        plt.savefig(fig_save_dir + 'example_lick_prediction_zoomout_' +
                    str(dataset_id) + '_' + str(expt_num) + '_' +
                    str(start_trial) + '_' + str(end_trial) + '.pdf',
                    transparent=True, rasterized=True, dpi=600)


def resample(x, y, x_new):
    """
    Resample y based on x_new (using linear interpolation).

    :param x: array.
    :param y: array.
    :param x_new: array.
    :return:
    """

    f = interp1d(x, y)
    y_new = f(x_new)
    return y_new


def interp_to_size(x, n=1000):
    """
    Interpolate a uniformly spaced array
    to have n entries.

    :param x: The array to resample.
    :param n: The length of the output array.

    :return x_new: The resampled array.
    """

    f = interp1d(np.linspace(0, 1, len(x)), x)
    x_new = f(np.linspace(0, 1, n))
    return x_new


def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the
             size of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to np arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError('a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError('ws cannot be larger than a in any dimension.\
                          a.shape was %s and ws was %s' % (str(a.shape),
                         str(ws)))

    # how many slices will there be in each dimension?
    with np.errstate(divide='ignore'):
        newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dim
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size,
    # plus the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.
    # I.e., the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = list(filter(lambda i: i != 1, dim))
    return strided.reshape(dim)


# This is a modified version of the original function operating on trials
# rather than time points.
def split_dataset_trials(X, Y, trial_frames,
                         opts=default_splitopts, do_debug=True):
    """
    Split a dataset into train, test, and validate sets.
    In opts, you can provide 'train_range', etc which will
    just chunk up the data into 3 continuous chunks.
    Alternatively, you can provide 'train_inds', etc
    which are boolean arrays of length X.shape[0] (i.e. one label
    for each timepoint), indicating which timepoints are
    assigned to test/train/validate. This is more flexible
    and lets you split things up better, i.e. according to
    trials etc.

    bins_after is the most important parameter here in opts
    standardize_x and center_y also have an effect

    :param X: neural data (n_frames x n_neurons)
    :param Y: regressor (n_trials x n_spouts)

    """
    assert opts['bins_before'] == 0, 'Bins before should equal zero!'
    assert opts['bins_current'] == 0, 'Bins current should equal zero!'
    assert opts['bins_after'] > 0, 'Bins after trial start should be nonzero!'

    T, N = X.shape
    P = Y.shape[1]  # number of spouts
    window_size = 1 + opts['bins_after']

    # Chunk up the neural data
    Xt = {}
    ix = {}
    for key in ['train_frames', 'test_frames', 'valid_frames']:
        Xt[key] = np.zeros((len(trial_frames[key]), window_size, N))
        for trial_idx, trial_frame in enumerate(trial_frames[key]):
            ix[key] = np.arange(trial_frame,
                                trial_frame+opts['bins_after']+1).astype(int)
            Xt[key][trial_idx, :, :] = X[ix[key], :]
        print(np.shape(Xt[key]))
    X_train = Xt['train_frames']
    X_test = Xt['test_frames']
    X_valid = Xt['valid_frames']

    # Standardize results
    X_train_mean = np.nanmean(X_train, axis=0)
    X_train_std = np.nanstd(X_train, axis=0)
    # import pdb; pdb.set_trace()
    if opts['standardize_X']:

        # Z-score "X" inputs
        X_train = (X_train-X_train_mean)/X_train_std
        X_test = (X_test-X_train_mean)/X_train_std
        X_valid = (X_valid-X_train_mean)/X_train_std

    # Chunk up the regressors
    Y_train = Y[trial_frames['train']-1, :]
    Y_test = Y[trial_frames['test']-1, :]
    Y_valid = Y[trial_frames['valid']-1, :]

    # Zero-center outputs
    Y_train_mean = np.mean(Y_train, axis=0)
    if opts['center_Y']:
        Y_train = Y_train-Y_train_mean
        Y_test = Y_test-Y_train_mean
        Y_valid = Y_valid-Y_train_mean

    return {'X_train': X_train,
            'X_train_mean': X_train_mean,
            'X_train_std': X_train_std,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'X_valid': X_valid,
            'Y_valid': Y_valid,
            'X_all': X,
            'Y_all': Y,
            'train_idx': ix['train_frames'],
            'test_idx': ix['test_frames'],
            'valid_idx': ix['valid_frames'],
            'bins_before': opts['bins_before'],
            'bins_current': opts['bins_current'],
            'bins_after': opts['bins_after']}


# This reimplements original function from Kording lab, using stride tricks.
def split_dataset(X, Y, opts=default_splitopts, do_debug=True):
    """
    Split a dataset into train, test, and validate sets.
    In opts, you can provide 'train_range', etc which will
    just chunk up the data into 3 continuous chunks.
    Alternatively, you can provide 'train_inds', etc
    which are boolean arrays of length X.shape[0] (i.e. one label
    for each timepoint), indicating which timepoints are
    assigned to test/train/validate. This is more flexible
    and lets you split things up better, i.e. according to
    trials etc.

    :param X:
    :param Y:

    """
    T, N = X.shape
    P = Y.shape[1]
    if opts['bins_after'] > 0 and opts['bins_before'] > 0:
        assert opts['bins_current'] == 1, 'Invalid options: bins_before>0 & bins_after>0 but bins_current==0 !!'
    # get the full window size
    window_size = opts['bins_before'] + 1 + opts['bins_after']

    if opts['train_inds'] is None:
        opts['train_range'] = [0, 0.7]
        opts['test_range'] = [0.7, 0.85]
        opts['valid_range'] = [0.85, 1]
        get_ix = lambda x: np.arange(
            np.round(x[0]*T)+opts['bins_before'],
            np.round(x[1]*T)-opts['bins_after']).astype(np.int)
        train_ix = get_ix(opts['train_range'])-opts['bins_before']
        test_ix = get_ix(opts['test_range'])-opts['bins_before']
        valid_ix = get_ix(opts['valid_range'])-opts['bins_before']
    else:
        train_ix = opts['train_inds']
        test_ix = opts['test_inds']
        valid_ix = opts['valid_inds']

    uX_train = np.nanmean(X[np.where(train_ix)[0]], axis=0)
    sX_train = np.nanstd(X[np.where(train_ix)[0]], axis=0)
    # import pdb; pdb.set_trace()
    if opts['standardize_X']:
        print('Z-scoring X!!')
        np.seterr(divide='ignore', invalid='ignore')
        # get training mean and std for X, mean for Y
        X = (X - uX_train) / sX_train
    if opts['center_Y']:
        uY_train = Y[np.where(train_ix)[0]].mean(0)

    print('Max X {}'.format(np.max(X)))


    # Use stride tricks to get the windowed data.
    # In particular, sliding_window with these parameters
    # returns a matrix [ntime x nshifts x ncells].
    # If you index into shift = opts['bins_before'], then
    # That is aligned with the original time series that has
    # been truncated at the front by opts['bins_before'].
    # (For example, if opts['bins_before']=2, then
    # X[2, :] = X_win[0, 2, :].
    # Thus, X_win[0, 1, :] = X[1, :], i.e. it is one step
    # back into the past. So, if you want to get causal timepoints
    # then you will take X_win[0, :2, :], which should correspond to
    # Y_win[0, 2, :].
    # Moreover, if opts['bins_after'] = 1, then X[3, :] = X_win[0, 3, :],
    # which is one time point into the future relative to Y_win[0, 2, :].
    X_win = sliding_window(X, ws=(window_size, N), ss=(1, 0),
                           flatten=True)
    Y_win = sliding_window(Y, ws=(window_size, P), ss=(1, 0),
                           flatten=True)

    # import pdb; pdb.set_trace()

    if len(X_win.shape) == 2:
        X_win = X_win[:, np.newaxis, :]
        Y_win = Y_win[:, np.newaxis, :]

    if do_debug:
        import matplotlib.pyplot as plt
        # Make a plot to double check the sliding window shifted
        # things in the correct direction.
        plt.figure()
        plt.plot(X_win[100:150, :, 1], 'g')
        plt.plot(X[100+opts['bins_before']:150+opts['bins_before'], 1])
        plt.plot(Y_win[100:150, opts['bins_before']])
        plt.plot(Y[100:150, 1])

    if opts['bins_current'] == 0:
        if opts['bins_before'] > 0:  # purely causal
            X_win = X_win[:, :opts['bins_before'], :]
        elif opts['bins_after'] > 0:  # purely acausal
            X_win = X_win[:, -opts['bins_after']:, :]
    if not opts['dense_Y']:
        Y_win = Y_win[:, opts['bins_before']]  # This extracts the current_bin

    do_original = False
    if do_original:
        # Pre-11/6/18, seems like it has a potential bug in it.
        # Specifically, it does not return the specific trials
        # that were requested, but instead a shifted set of those trials.
        X_train = X_win[train_ix[:X_win.shape[0]]]  # Original pre-11/6/18
        X_test = X_win[test_ix[:X_win.shape[0]]]  # Original pre-11/6/18
        X_valid = X_win[valid_ix[:X_win.shape[0]]]  # Original pre-11/6/18

        Y_train = Y_win[train_ix[:Y_win.shape[0]]]  # Original pre-11/6/18
        Y_test = Y_win[test_ix[:Y_win.shape[0]]]  # Original pre-11/6/18
        Y_valid = Y_win[valid_ix[:Y_win.shape[0]]]  # Original pre-11/6/18
    else:
        X_train = X_win[train_ix[-X_win.shape[0]:]]
        # This syntax indexes first dimension and keeps higher dimensions.
        X_test = X_win[test_ix[-X_win.shape[0]:]]
        X_valid = X_win[valid_ix[-X_win.shape[0]:]]

        Y_train = Y_win[train_ix[-Y_win.shape[0]:]]
        Y_test = Y_win[test_ix[-Y_win.shape[0]:]]
        Y_valid = Y_win[valid_ix[-Y_win.shape[0]:]]

    return {'X_train': X_train,
            'X_train_mean': uX_train,
            'X_train_std': sX_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'X_valid': X_valid,
            'Y_valid': Y_valid,
            'X_all': X,
            'Y_all': Y,
            'train_idx': train_ix,
            'test_idx': test_ix,
            'valid_idx': valid_ix,
            'bins_before': opts['bins_before'],
            'bins_current': opts['bins_current'],
            'bins_after': opts['bins_after']}


# Here we randomize train, validation, and test sets in time
# rather than splitting contiguous blocks
# N.B. probably worth re-thinking this and
# being more sophisticated about trial within block, etc
def split_dataset_randtime(X, Y, opts=default_splitopts):
    opts['train_range'] = [0, 0.7]
    opts['test_range'] = [0.7, 0.85]
    opts['valid_range'] = [0.85, 1]

    T, N = X.shape
    P = Y.shape[1]
    if opts['bins_after'] > 0 and opts['bins_before'] > 0:
        assert opts['bins_current'] == 1, 'Invalid options: bins_before>0 & bins_after>0 but bins_current==0 !!'
    # get the full window size
    window_size = opts['bins_before'] + 1 + opts['bins_after']
    get_N = lambda x: np.round((x[1]-x[0])*T).astype(int)
    remaining_ix = np.arange(opts['bins_before'],
                             X.shape[0]-opts['bins_after']-2)
    train_ix = np.random.choice(remaining_ix,
                                size=get_N(opts['train_range']),
                                replace=False)
    remaining_ix = np.delete(remaining_ix,
                             np.where([x in train_ix for x in remaining_ix]))
    test_ix = np.random.choice(remaining_ix,
                               size=get_N(opts['test_range']),
                               replace=False)
    remaining_ix = np.delete(remaining_ix,
                             np.where([x in test_ix for x in remaining_ix]))
    valid_ix = remaining_ix
    uX_train = np.nanmean(X[train_ix], axis=0)
    sX_train = np.nanstd(X[train_ix], axis=0)
    if opts['standardize_X']:
        # get training mean and std for X, mean for Y
        np.seterr(divide='ignore', invalid='ignore')
        X = (X - uX_train) / sX_train
    if opts['center_Y']:
        uY_train = Y[train_ix].mean(0)
    # use stride tricks to get the windowed data
    X_win = sliding_window(X, ws=(window_size, N), ss=(1, 0), flatten=True)
    Y_win = sliding_window(Y, ws=(window_size, P), ss=(1, 0), flatten=True)
    if opts['bins_current'] == 0:
        if opts['bins_before'] > 0:  # purely causal
            X_win = X_win[:, :opts['bins_before'], :]
        elif opts['bins_after'] > 0:  # purely acausal
            X_win = X_win[:, -opts['bins_after']:, :]
    X_train = X_win[train_ix]
    X_test = X_win[test_ix]
    X_valid = X_win[valid_ix]
    if not opts['dense_Y']:  # TODO: check this
        Y_win = Y_win[:, opts['bins_before']]
    Y_train = Y_win[train_ix]
    Y_test = Y_win[test_ix]
    Y_valid = Y_win[valid_ix]
    return {'X_train': X_train,
            'X_train_mean': uX_train,
            'X_train_std': sX_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'X_valid': X_valid,
            'Y_valid': Y_valid,
            'train_idx': train_ix,
            'test_idx': test_ix,
            'valid_idx': valid_ix,
            'bins_before': opts['bins_before'],
            'bins_current': opts['bins_current'],
            'bins_after': opts['bins_after']}


# This is just Kording stuff repackaged
def split_dataset_orig(neural_data,
                       y, bins_current=1,
                       bins_before=0,
                       bins_after=0,
                       standardize_X=True,
                       standardize_Y=True,
                       training_range=[0, 0.7],
                       testing_range=[0.7, 0.85],
                       valid_range=[0.85, 1]):

    X = get_spikes_with_history(neural_data, bins_before,
                                bins_after, bins_current)
    # Format for Wiener Filter, Wiener Cascade, XGBoost,
    # and Dense Neural Network
    # Put in "flat" format, so each "neuron / time" is a single feature
    X_flat = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))

    num_examples = X.shape[0]

    # Note that each range has a buffer of"bins_before" bins at the beginning,
    # and "bins_after" bins at the end
    # This makes it so that the different sets don't include
    # overlapping neural data

    training_set = np.arange(
        np.int(np.round(training_range[0]*num_examples))+bins_before,
        np.int(np.round(training_range[1]*num_examples))-bins_after)
    testing_set = np.arange(
        np.int(np.round(testing_range[0]*num_examples))+bins_before,
        np.int(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set = np.arange(
        np.int(np.round(valid_range[0]*num_examples))+bins_before,
        np.int(np.round(valid_range[1]*num_examples))-bins_after)

    # Get training data
    X_train = X[training_set, :, :]
    X_flat_train = X_flat[training_set, :]
    y_train = y[training_set, :]

    # Get testing data
    X_test = X[testing_set, :, :]
    X_flat_test = X_flat[testing_set, :]
    y_test = y[testing_set, :]

    # Get validation data
    X_valid = X[valid_set, :, :]
    X_flat_valid = X_flat[valid_set, :]
    y_valid = y[valid_set, :]

    X_train_mean = np.nanmean(X_train, axis=0)
    X_train_std = np.nanstd(X_train, axis=0)

    if standardize_X:
        # Z-score "X" inputs.
        X_train = (X_train-X_train_mean)/X_train_std
        X_test = (X_test-X_train_mean)/X_train_std
        X_valid = (X_valid-X_train_mean)/X_train_std

        # Z-score "X_flat" inputs.
        X_flat_train_mean = np.nanmean(X_flat_train,
                                       axis=0-opts['bins_before'])
        X_flat_train_std = np.nanstd(X_flat_train, axis=0)
        X_flat_train = (X_flat_train-X_flat_train_mean)/X_flat_train_std
        X_flat_test = (X_flat_test-X_flat_train_mean)/X_flat_train_std
        X_flat_valid = (X_flat_valid-X_flat_train_mean)/X_flat_train_std

    y_train_mean = np.mean(y_train, axis=0)
    if standardize_Y:  # Zero-center outputs
        y_train = y_train-y_train_mean
        y_test = y_test-y_train_mean
        y_valid = y_valid-y_train_mean

    return {'X_train': X_train,
            'X_train_mean': X_train_mean,
            'X_train_std': X_train_std,
            'y_train_mean': y_train_mean,
            'X_flat_train': X_flat_train,
            'y_train': y_train,
            'X_test': X_test,
            'X_flat_test': X_flat_test,
            'y_test': y_test,
            'X_valid': X_valid,
            'X_flat_valid': X_flat_valid,
            'y_valid': y_valid}
