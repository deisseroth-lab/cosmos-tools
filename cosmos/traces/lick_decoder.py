
import numpy as np
import scipy
import time
import os
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed

import cosmos.traces.decoding_utils as du


class LickDecoder:
    """
    Decode behavioral licks from neural data
    for a single behavioral session.
    """
    def __init__(self, dataset_id, CT, decoding_save_dir,
                 just_decode_licks=False):
        """
        Constructor for LickDecoder class.

        :param dataset_id: int. Reference to dataset in trace_analyze_params.
        :param CT: CosmosTraces instance.
        :param decoding_save_dir: str. Path to directory for saving results.
        :param just_decode_licks: bool. If False, then 4-way decoding
            (no lick, lick to spout 1, to spout 2, to spout 3).
            If True, then 3-way decoding just on timepoints
            when the animal is licking (to spout 1, 2, 3).
        """

        self.dataset_id = dataset_id
        self.CT = CT
        self.BD = CT.bd

        self.decoding_save_dir = decoding_save_dir
        # self.plot_dir = os.path.join(decoding_save_dir, 'plots')

        self.data_opts = {
                          # ID number of which data labeling to decode.
                          # See get_data_for_decoding for details.
                          'decoding_set': 1,
                          # 'spikes', 'smooth_spikes',
                          # or 'fluor', 'spikes_binarized'
                          'train_feat': 'spikes',
                          'train_frac': 0.5,
                          'test_frac': 0.25,
                          'valid_frac': 0.25,
                          'remove_multi_licks': False,
                          'rand_seed': 0,
                          'bins_current': 1,  # include frame of each event
                          'bins_before': 0,  # 1 # frames before the event
                          'bins_after': 0,  # 1 # frames after the event
                          'standardize_X': True,  # u=0, s=1
                          'center_Y': True}  # u=0

        print(self.data_opts)
        if just_decode_licks:
            self.data_opts['decoding_set'] = 7
            self.data_opts['exclude_no_licks'] = True

    def verify_data_alignment(self, trial_ind=1, which_source=60):
        """
        Generate a plot to ensure that partitioned train/test/validate
        data matches the raw neural and bpod behavioral data.

        :param trial_ind: int. which trial to plot
        :param which_source: int. ID of the source from which to plot activity.
        """
        du.verify_data_alignment(trial_ind, which_source, self.CT,
                                 self.data_split, self.data_opts)

    def get_source_discriminativity(self):
        """
        Return an ordering of neural sources based on the ability
        of their activity to discriminate features of the licking.

        Specifically, compute a p-value using Kruskal-Wallis H-test
        (non-parametric version of ANOVA) that the population
        medians of neural activity over all timepoints when the mouse
        is engaging in each type of behavior (nolick, lick1, lick2, lick3)
        are different.

        Save the ordering out to decoding_save_dir, '_ordering'


        :return:
        """
        du.get_source_discriminativity(
            self.data_split, self.CT, self.data_opts, self.decoding_save_dir)

    def decode_licks(self, expt_group_id, nfolds,
                     do_pca=True,
                     n_pca_components=85,
                     do_shuffle_data=False):
        """
        :param expt_group_id: int. These are defined in
                              decoding_utils.select_experiment_group(),

        :param nfolds: int. Number of folds of train/test/validation datasets
                            to run.
        :param do_pca: bool. Whether to use top principal components
                             of the neural data when decoding, as opposed
                             to just using the neural data.
        :param n_pca_components: int.
              If do_pca, then how many components to use.
        :param do_shuffle_data: bool.
              If true, then shuffles the labels
              before decoding to get a measure of chance-level
              decoding performance.
        :return:
        """

        expt_nums = []
        expts_info = {}

        for fold_number in range(nfolds):
            # Split data up for this fold.
            # self.data_opts['rand_seed'] = fold
            self.data_opts['rand_seed'] = 0
            self.data_opts['fold_number'] = fold_number
            self.data_opts['nfolds'] = nfolds

            data_split = du.get_data_for_decoding(self.CT, self.data_opts,
                                                  do_debug=False)
            self.data_split = data_split
            print('X_train:', data_split['X_train'].shape,
                  'X_test:', data_split['X_test'].shape)

            # Double check that licks are aligned correctly by plotting against
            # direct-from-bpod lick rates.
            do_verify = False
            if do_verify:
                du.verify_data_alignment(trial_ind=1, which_cell=60,
                                         CT=self.CT, data_split=data_split,
                                         data_opts=self.data_opts)

            # import pdb; pdb.set_trace()
            #  Rank the neural sources based on the ability
            #  of their activity to discriminate features of the licking.
            p_ordering, minpvals = du.get_source_discriminativity(
                data_split, self.CT, self.data_opts, self.decoding_save_dir,
                do_plot=False)

            # Now train the model
            t0 = time.time()
            seed(1), set_random_seed(2)

            do_single_experiment = (expt_group_id == 0)
            if do_single_experiment:
                expt_param = None

                neuron_opts = {'nneurons': self.CT.ncells,
                               'neuron_set': 'rand',
                               'neuron_rand_seed': 0,
                               'hemisphere': None,
                               'expt_num': du.get_last_expt_in_folder(
                                                self.decoding_save_dir) + 1
                               }
                subsets = du.get_neurons_for_decoding(
                    neuron_opts, self.CT, nrepeats=1)
                neuron_opts['which_cells'] = subsets[0]
                experiment_group = [neuron_opts]
            else:
                expt_param = du.select_experiment_group(
                    expt_group_id=expt_group_id, ordering=p_ordering)
                expt_param['minpvals'] = minpvals
                experiment_group = du.get_decoding_experiment_group(
                    expt_param, self.CT, self.decoding_save_dir)

            print('# experiments: {}'.format(len(experiment_group)))
            print('Expt range: {} to {}'.format(
                experiment_group[0]['expt_num'],
                experiment_group[-1]['expt_num']))

            expt_nums.append(experiment_group[0]['expt_num'])
            expt_nums.append(experiment_group[-1]['expt_num'])

            for neuron_opts in experiment_group:
                du.run_decoding_experiment(neuron_opts,
                                           self.data_opts,
                                           self.decoding_save_dir,
                                           data_split,
                                           self.CT,
                                           self.dataset_id,
                                           expt_param,
                                           do_pca=do_pca,
                                           pca_components=n_pca_components,
                                           do_shuffle=do_shuffle_data)

            print('Total time: {:.5f}'.format(time.time() - t0))

        expts_info['expt_nums'] = expt_nums
        expts_info['id'] = self.dataset_id
        expts_info['expt_group_id'] = expt_group_id

        expts_str = self.get_expts_str(expts_info)

        return expts_info, expts_str

    def get_expts_str(self, expts_info):
        """
        Return a string the describes the experiment group,
        and that can be directly copy and pasted into
        a parameter dict for loading the results.
        :return: expts_str.
        """

        expt_type = 'neuron_set'
        if expts_info['expt_group_id'] in [3, 8]:
            expt_type = 'nneurons'
        expts_str = (('\'id\': {}, '
                      '\'expt_nums\': np.arange({}, {}),'
                      '\'expt_type\': \'{}\', '
                      '\'info\':{}').format(
                          expts_info['id'],
                          np.min(expts_info['expt_nums']),
                          np.max(expts_info['expt_nums'])+1,
                          expt_type,
                          expts_info['expt_group_id']))

        return expts_str
