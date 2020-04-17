from matplotlib import pyplot as plt
import scipy.io as sio
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re

import argparse

from cosmos.behavior.bpod_dataset import BpodDataset


def summarize_performance_of_stim_trials(self, fig_save_dir=None, min_trial=15,
                                         max_trial=190):
    """
    Compare stim vs non-stim trial blocks.

    :return:
    """

    # You need:
    #  self.stim_types: which trials were stimmed
    #  self.success: success on those trials
    #  self.spout_positions: break this down by spout direction,
    #  - make rose plots (see bd.plot_spout_selectivity() )

    stim = self.stim_types.astype(bool)
    nostim = ~self.stim_types.astype(bool)

    include_trials = np.ones(stim.shape).astype(bool)
    include_trials[:min_trial] = 0
    include_trials[max_trial:] = 0

    stim = stim*include_trials
    nostim = nostim*include_trials

    stim_success = np.sum(np.logical_and.reduce((stim,
                                                 self.success,
                                                 include_trials)))
    nostim_success = np.sum(np.logical_and.reduce((nostim,
                                                   self.success,
                                                   include_trials)))

    self.plot_spout_selectivity(trial_subset=self.stim_types.astype(bool),
                                alt_colors=True, do_save=False,
                                min_trial=min_trial)
    plt.suptitle('Stim trials: {}/{}={:.2f}'.format(stim_success,
                                                    np.sum(stim),
                                                    stim_success/np.sum(stim)))

    if fig_save_dir is not None:
        print('Saving to: ', os.path.join(fig_save_dir, 'polar_stim.png'))
        plt.savefig(os.path.join(fig_save_dir, 'polar_stim.png'))

    self.plot_spout_selectivity(trial_subset=~self.stim_types.astype(bool),
                                alt_colors=True, do_save=False,
                                min_trial=min_trial)
    plt.suptitle(
        'Non-stim trials: {}/{}={:.2f}'.format(
            nostim_success, np.sum(nostim), nostim_success/np.sum(nostim)))
    if fig_save_dir is not None:
        print('Saving to: ', os.path.join(fig_save_dir, 'polar_nostim.png'))
        plt.savefig(os.path.join(fig_save_dir, 'polar_nostim.png'))

    # Get fraction correct by spout direction.
    scores = pd.DataFrame()
    for spout in np.unique(self.spout_positions):
        spout_stim = np.logical_and.reduce((stim,
                                            self.spout_positions == spout,
                                            include_trials))
        spout_stim_success = np.sum(np.logical_and(spout_stim, self.success))
        spout_stim_success /= np.sum(spout_stim)

        d = {'Spout': spout, 'Stim': 1, 'Success': spout_stim_success}
        scores = scores.append(d, ignore_index=True)

        spout_nostim = np.logical_and.reduce((nostim,
                                              self.spout_positions == spout,
                                              include_trials))
        spout_nostim_success = np.sum(
            np.logical_and(spout_nostim, self.success))
        spout_nostim_success /= np.sum(spout_nostim)

        d = {'Spout': spout, 'Stim': 0, 'Success': spout_nostim_success}
        scores = scores.append(d, ignore_index=True)

    plt.figure()
    sns.barplot(data=scores, x='Spout',
                y='Success', hue='Stim')
    plt.legend()
    if fig_save_dir is not None:
        print('Saving to: ',
              os.path.join(fig_save_dir, 'spout_stim_nostim_success.png'))
        plt.savefig(
            os.path.join(fig_save_dir, 'spout_stim_nostim_success.png'))

    print(scores)

    # See BpodDataset._get_trial_success()
    # success = np.logical_or.reduce((
    #     ~np.isnan(self.reward_times),
    #     np.logical_and(trial_types == 4, np.isnan(self.punish_times))
    # ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify behavior data path.')
    parser.add_argument('--behav_path',
                        type=str, help='full path to bpod file.')
    parser.add_argument('--save_path',
                        type=str, help='full path to bpod file.')
    args = parser.parse_args()

    if args.behav_path:
        behavior_path = args.behav_path
    else:
        behavior_path = (
            '/home/izkula/Dropbox/cosmos_data/' +
            'Behavior_bpod_vgat/vGatm15/' +
            'StimHDMI_ODOR_COSMOSTrainMultiBlockGNG/Session Data/' +
            'vGatm15_StimHDMI_preodor_' +
            'COSMOSTrainMultiBlockGNG_20190112_173258.mat')

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = ('/home/izkula/Dropbox/cosmos/' +
                     'trace_analysis/inhibition_plots/')

    print(save_path)
    name = behavior_path.replace('\\', '/').split('/')[-1].split('.')[0]

    fig_save_path = os.path.join(save_path, name)
    os.makedirs(fig_save_path, exist_ok=True)

    bd = BpodDataset(behavior_path, fig_save_path)
    bd.suffix = '.png'

    bd.plot_lick_times(alt_colors=True, underlay_stim_trials=True)

    if bd.stim_interval is not None:
        stim_interval = bd.stim_interval - bd.stimulus_times[0]
    else:
        stim_interval = [-2, 3]
    bd.plot_stim_licks(fig_save_dir=fig_save_path, stim_interval=stim_interval)

    # Summarize licks stim/nostim during each time period.
    intervals = {
        'Pre-odor': [0.1, 2.2], 'odor': [2.2, 3.7], 'Post-odor': [3.7, 5.7]}
    scores = bd.summarize_licks_during_stim(intervals, do_plot=True,
                                            fig_save_dir=fig_save_path)

    # Summarize task performance during stim/nostim
    summarize_performance_of_stim_trials(bd, fig_save_dir=fig_save_path)

    # plt.show()
    print('done')
