
import numpy as np
import scipy
import time
import os
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed

import cosmos.traces.decoding_utils as du


class LatentLickDecoder:
    """
    Decode behavioral licks from neural data
    for a single behavioral session.
    """
    def __init__(self, dataset_id, CT, decoding_save_dir):

        self.dataset_id = dataset_id
        self.CT = CT
        self.BD = CT.bd

        self.decoding_save_dir = decoding_save_dir
        # self.plot_dir = os.path.join(decoding_save_dir, 'plots')

        self.data_opts = {'decoding_set': 6, # ID number of which data labeling to decode.
                                             # See get_data_for_decoding for details.
                         'train_feat': 'spikes', # 'spikes', 'smooth_spikes', or 'fluor'
                         'train_frac': 0.5,
                         'test_frac': 0.25,
                         'valid_frac': 0.25,
                         'remove_multi_licks': False,
                         'rand_seed': 0,
                         'bins_current': 1,  # include frame of each event
                         'bins_before': 3,  # frames before the event
                         'bins_after': 1,  # frames after the event
                         'standardize_X': False,  # u=0, s=1
                         'center_Y': True}  # u=0

        self.data_opts['just_clean_trials'] = 0.8
        self.data_opts['ensure_all_trial_types_present'] = True
        odor_frame = int(np.round(self.BD.stimulus_times[0] * CT.fps))
        self.data_opts['within_trial_frame_range'] = [1, odor_frame - 1]
        self.data_opts['just_no_preodor_lick_trials'] = True

    def verify_data_alignment(self, trial_ind=1, which_source=60):
        """
        Generate a plot to ensure that partitioned train/test/validate
        data matches the raw neural and bpod behavioral data.

        :param trial_ind: int. which trial to plot
        :param which_source: int. ID of the source from which to plot neural activity.
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
        return du.get_source_discriminativity(self.data_split,
                                              self.CT,
                                              self.data_opts,
                                              self.decoding_save_dir)



    def decode_licks(self, expt_group_id, nfolds,
                     do_pca=True,
                     n_pca_components=85,
                     do_shuffle_data=False,
                     do_debug=False):
        """
        :param expt_group_id:
        :param nfolds:
        :param do_pca:
        :param n_pca_components:
        :param do_shuffle_data:
        :return:
        """

        expt_nums = []
        expts_info = {}
        for fold_number in range(nfolds):
            self.data_opts['rand_seed'] = 0
            self.data_opts['fold_number'] = fold_number
            self.data_opts['nfolds'] = nfolds

            data_split = du.get_data_for_decoding(self.CT, self.data_opts,
                                                  do_debug=do_debug)
            self.data_split = data_split
            print('X_train:', data_split['X_train'].shape,
                  'X_test:', data_split['X_test'].shape)

            p_ordering = du.get_source_discriminativity(data_split,
                                                        self.CT,
                                                        self.data_opts,
                                                        self.decoding_save_dir,
                                                        do_plot=False)

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
                subsets = du.get_neurons_for_decoding(neuron_opts, self.CT,
                                                      nrepeats=1)
                neuron_opts['which_cells'] = subsets[0]
                experiment_group = [neuron_opts]
            else:
                expt_param = du.select_experiment_group(
                                            expt_group_id=expt_group_id,
                                            ordering=p_ordering)
                experiment_group = du.get_decoding_experiment_group(expt_param,
                                                                    self.CT,
                                                                    self.decoding_save_dir)

            print('# experiments: {}'.format(len(experiment_group)))
            print('Expt range: {} to {}'.format(experiment_group[0]['expt_num'],
                                                experiment_group[-1][
                                                    'expt_num']))

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

                du.decode_trials(neuron_opts['expt_num'], self.decoding_save_dir)

            print('Total time: {:.5f}'.format(time.time() - t0))

        expts_info['expt_nums'] = expt_nums
        expts_info['id'] = self.dataset_id
        expts_info['expt_group_id'] = expt_group_id

        expts_str = self.get_expts_str(expts_info)

        return expts_info, expts_str
        # pass





    #     expt_nums = []
    #     expts_info = {}
    #
    #     for fold_number in range(nfolds):
    #         ### Split data up for this fold.
    #         # self.data_opts['rand_seed'] = fold
    #         self.data_opts['rand_seed'] = 0
    #         self.data_opts['fold_number'] = fold_number
    #         self.data_opts['nfolds'] = nfolds
    #
    #
    #         data_split = du.get_data_for_decoding(self.CT, self.data_opts,
    #                                               do_debug=False)
    #         self.data_split = data_split
    #         print('X_train:', data_split['X_train'].shape,
    #               'X_test:', data_split['X_test'].shape)
    #
    #         ### Double check that licks are aligned correctly by plotting against
    #         ### direct-from-bpod lick rates.
    #         do_verify = False
    #         if do_verify:
    #             du.verify_data_alignment(trial_ind=1, which_cell=60,
    #                                     CT=self.CT, data_split=data_split,
    #                                     data_opts=self.data_opts)
    #
    #         ###  Rank the neural sources based on the ability
    #         ###  of their activity to discriminate features of the licking/labels.
    #         p_ordering = du.get_source_discriminativity(data_split,
    #                                                     self.CT,
    #                                                     self.data_opts,
    #                                                     self.decoding_save_dir,
    #                                                     do_plot=False)
    #         ### Now train the model
    #         t0 = time.time()
    #         seed(1), set_random_seed(2)
    #
    #         do_single_experiment = (expt_group_id == 0)
    #         if do_single_experiment:
    #             expt_param = None
    #
    #             neuron_opts = {'nneurons': self.CT.ncells,
    #                            'neuron_set': 'rand',
    #                            'neuron_rand_seed': 0,
    #                            'hemisphere': None,
    #                            'expt_num': du.get_last_expt_in_folder(
    #                                             self.decoding_save_dir) + 1
    #                            }
    #             subsets = du.get_neurons_for_decoding(neuron_opts, self.CT, nrepeats=1)
    #             neuron_opts['which_cells'] = subsets[0]
    #             experiment_group = [neuron_opts]
    #         else:
    #             expt_param = du.select_experiment_group(expt_group_id=expt_group_id,
    #                                                  ordering=p_ordering)
    #             experiment_group = du.get_decoding_experiment_group(expt_param,
    #                                                                 self.CT,
    #                                                                 self.decoding_save_dir)
    #
    #         print('# experiments: {}'.format(len(experiment_group)))
    #         print('Expt range: {} to {}'.format(experiment_group[0]['expt_num'],
    #                                             experiment_group[-1][
    #                                                 'expt_num']))
    #
    #         expt_nums.append(experiment_group[0]['expt_num'])
    #         expt_nums.append(experiment_group[-1]['expt_num'])
    #
    #         for neuron_opts in experiment_group:
    #             du.run_decoding_experiment(neuron_opts,
    #                                        self.data_opts,
    #                                        self.decoding_save_dir,
    #                                        data_split,
    #                                        self.CT,
    #                                        self.dataset_id,
    #                                        expt_param,
    #                                        do_pca=do_pca,
    #                                        pca_components=n_pca_components,
    #                                        do_shuffle=do_shuffle_data)
    #
    #         print('Total time: {:.5f}'.format(time.time() - t0))
    #
    #
    #     expts_info['expt_nums'] = expt_nums
    #     expts_info['id'] = self.dataset_id
    #     expts_info['expt_group_id'] = expt_group_id
    #
    #     expts_str = self.get_expts_str(expts_info)
    #
    #     return expts_info, expts_str
    #
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
                      '\'info\':{}').format(expts_info['id'],
                                     np.min(expts_info['expt_nums']),
                                     np.max(expts_info['expt_nums'])+1,
                                     expt_type,
                                     expts_info['expt_group_id']))

        return expts_str
