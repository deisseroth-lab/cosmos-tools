from collections import defaultdict
from copy import deepcopy
import itertools
import warnings
import os

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import mixture
import pandas as pd
import keras
from scipy.stats import ttest_rel, ttest_ind
import scipy.signal

import statsmodels.stats.multitest as mt
from scipy.spatial.distance import mahalanobis, cosine
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import patches
import scipy.stats as stats
import seaborn as sns
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import accuracy_score

from cosmos.traces.cosmos_traces import CosmosTraces
import cosmos.params.trace_analyze_params as params
import cosmos.traces.decoding_utils as decoder


# Default FPS in the absence of additional information
FPS = 29.41

# Default threshold for declaring a trial is clean
# (this is the fraction of pre-reward licks to the active spout)
DEFAULT_CLEAN_THRESHOLD = 0.8

# Default number of components to use for training all PLS models.
DEFAULT_K_COMPONENTS = 4

# Default number of max cells to use (when specific cells aren't specified)
DEFAULT_MAX_CELLS_TO_USE = 500

# Note: 'classifier_utils' focuses on decoding the lick target from
# single-trial data. In contrast, 'decoding_utils' focuses on decoding licks
# from individual imaging frames. They are largely independent codebases
# that share a few common performance metrics. The models used are different.


class ClassifierData:
    """
    Load data from a dataset_id, this will be used to train PLS models.
    """

    def __init__(self, dataset_id, data_dir, fig_save_dir, bpod_dir,
                 pre_reward_interval=[1.0, 1.5], spike_smoothing_sigma=1.5,
                 use_spikes=False):
        print('Loading dataset ', dataset_id)
        self.metadata = params.DATASETS[dataset_id]
        self.metadata['data_root'] = data_dir
        self.metadata['fig_save_dir'] = fig_save_dir
        self.metadata['bpod_dir'] = bpod_dir

        self.cosmos_traces = CosmosTraces(self.metadata)
        self.behavior = self.cosmos_traces.bd
        self.use_spikes = use_spikes
        self.spike_smoothing_sigma = spike_smoothing_sigma

        BD = self.behavior
        CT = self.cosmos_traces

        # Get true spout positions
        true_pos = BD.spout_positions.copy()
        true_pos[np.where(true_pos == 1)[0]] = 2
        true_pos_ = true_pos - 2
        true_pos = keras.utils.to_categorical(true_pos_)
        self.spout_position = true_pos

        # Unthresholded number of licks pre-reward to each
        # spout on each trial.
        # The favored/preferred spout is defined as the one
        # with the most licks in the interval of time between
        # odor onset and reward onset (odor onset + 1.5).
        t_range = [BD.stimulus_times[0], BD.stimulus_times[0] + 1.5]
        inds = np.where((BD.lick_mat_t > t_range[0]) &
                        (BD.lick_mat_t < t_range[1]))[0]
        favored_spout = [[np.sum(BD.spout_lick_mat[spout][trial, inds])
                          for trial in range(BD.ntrials)]
                         for spout in range(4)]
        favored_spout_ = np.array(favored_spout)[[0, 2, 3], :].T

        # On trials where there are no licks, the favored spout is
        # the active spout if we don't do this,
        # all the nogo trials will become the first spout
        for nolicks in np.where(np.sum(favored_spout_, 1) == 0)[0]:
            favored_spout_[nolicks, :] = true_pos[nolicks, :]
        favored_spout = keras.utils.to_categorical(
            np.argmax(favored_spout_, 1))
        self.favored_spout = favored_spout

        # Compute normalized version of each trace
        Ct_n = CT.Ct.copy()

        # Subtract off the across-condition mean from each trial.
        mtr = np.mean(Ct_n, 2)
        Ct_n = [Ct_n[:, :, trial] - mtr for trial in range(np.shape(Ct_n)[2])]
        Ct_n = np.swapaxes(np.swapaxes(Ct_n, 0, 2), 0, 1)
        # Normalize each neuron by its variance.
        for neuron in range(np.shape(Ct_n)[0]):
            Ct_n[neuron, :, :] /= np.std(Ct_n[neuron, :, :])
        self.Ct_n = Ct_n

        # Compute deconvolved and smoothed spikes.
        St_n = CT.St.copy()
        St_n = gaussian_filter1d(St_n, spike_smoothing_sigma,
                                 axis=1, mode='constant')
        mtr = np.mean(St_n, 2)
        St_n = [St_n[:, :, trial] - mtr for trial in range(np.shape(St_n)[2])]
        St_n = np.swapaxes(np.swapaxes(St_n, 0, 2), 0, 1)
        # Normalize each neuron by its variance.
        for neuron in range(np.shape(St_n)[0]):
            St_n[neuron, :, :] /= np.std(St_n[neuron, :, :])
        self.St_n = St_n

        # If we are using spikes instead of calcium, replace the Ct_n variable
        if use_spikes:
            self.Ct_n = St_n

        # Rank all sources based on their discriminativity
        self.get_source_discriminativity()

    def get_source_discriminativity(self, signal_threshold=0.1,
                                    use_position=False):
        """
        Do a kruskal-wallis test on each neuron/timestep to rank how well
        each neuron can discriminate between the most-licked PREFERRED spout

        We're only going to rank using the PRE-ODOR IMAGING FRAMES.
        """

        # Get traces + ground truth active spout
        fps = self.cosmos_traces.fps
        odor_on = int(self.behavior.stimulus_times[0] * fps)
        traces = self.Ct_n[:, :odor_on, :]
        if use_position:
            spout = np.argmax(self.spout_position, 1)
        else:
            spout = np.argmax(self.favored_spout, 1)

        # We are doing 3-way classification
        assert(len(np.unique(spout)) == 3)

        # Run the H-test on each source
        ncells = np.shape(traces)[0]
        ntimes = np.shape(traces)[1]
        pvals = np.zeros((ncells, ntimes))
        for time in range(ntimes):
            for cell in range(ncells):
                # Only process sources with some signal
                if np.max(traces[cell, time, :] > signal_threshold):
                            f, p = stats.kruskal(
                                traces[cell, time, np.where(spout == 0)[0]],
                                traces[cell, time, np.where(spout == 1)[0]],
                                traces[cell, time, np.where(spout == 2)[0]])
                else:
                    p = 1
                pvals[cell, time] = p

        # Choose the best pvalue across the included timebins
        minpvals = np.min(pvals, axis=1)
        self.p_ordering = np.argsort(minpvals)
        self.p_vals = minpvals


class ClassifierModel:
    """
    Given a classifier dataset, generate PLSRegression fits.

    This class is just a simple wrapper around the normal fitting functions.
    """

    def __init__(self, dataset, test_epoch,
                 cells_to_use=None, epoch='all', k_comps=DEFAULT_K_COMPONENTS,
                 shuffle=False, clean_threshold=DEFAULT_CLEAN_THRESHOLD,
                 cull_pre_odor_licks=True, seed=22222,
                 max_cells_to_use=DEFAULT_MAX_CELLS_TO_USE):

        # Reset the random seed to ensure reproducibility
        np.random.seed(seed)

        # Note that the actual training/test split happens at the level
        # of trials, so we choose a constant set of sources
        self.fps = dataset.cosmos_traces.fps
        self.odor_on = int(dataset.behavior.stimulus_times[0] * self.fps)
        self.reward_on = int(
            (dataset.behavior.stimulus_times[0] + 1.5)*self.fps)
        self.behavior = deepcopy(dataset.behavior)
        self.spout_position = dataset.spout_position.copy()
        self.favored_spout = dataset.favored_spout.copy()
        self.p_ordering = dataset.p_ordering
        self.predictions = None
        self.k_comps = k_comps
        self.colors = ['orange', 'k', 'c', 'r']
        self.clean_threshold = clean_threshold
        self.cull_pre_odor_licks = cull_pre_odor_licks

        # Save time epochs for getting lick behavior
        self.t_pre_odor = [0, self.behavior.stimulus_times[0]]

        # Pre-reward
        self.t_pre_reward = [self.behavior.stimulus_times[0],
                             self.behavior.stimulus_times[0] + 1.5]

        # Post-reward
        self.t_post_reward = [self.behavior.stimulus_times[0] +
                              1.5, self.behavior.stimulus_times[0] + 3.5]

        # Which time samples to use for testing
        if test_epoch in ['all', 'pre odor all', 'post odor all']:
            self.test_epoch = test_epoch
        else:
            raise ValueError('Invalid test epoch argument!')

        # Get the number of pre-odor licks on each trial
        self.trial_pre_odor_licks = ClassifierModel.get_pre_odor_licks(
            self.behavior)

        # Get list of trials where mouse licked almost
        # exclusively in the chosen direction before shuffling.
        # Training trials will chosen from this subset of data.
        self.clean_trials = self.behavior.get_clean_trials(
            self.clean_threshold, verbose=False)

        # Only evaluate on trials where there <= k pre-odor licks
        if cull_pre_odor_licks:
            k = 0
            self.pre_lick_trials = np.where(self.trial_pre_odor_licks > k)[0]

        # Shuffle labels if desired
        self.shuffle = shuffle
        if shuffle not in [None, True, False, 'circ']:
            raise ValueError('Invalid shuffle argument: True, False, \'circ\'')
        if shuffle == 'circ':
            # Circularly permute the correct spout labels (conservative)
            offset = np.random.randint(0, self.behavior.ntrials)
            self.spout_position = np.roll(self.spout_position, offset, axis=0)
            self.favored_spout = np.roll(self.favored_spout, offset, axis=0)
            self.behavior.spout_positions = np.roll(
                self.behavior.spout_positions, offset)
        elif shuffle:
            # Randomly shuffle the correct spout labels (less conservative)
            perm = np.random.permutation(self.behavior.ntrials)
            self.spout_position = self.spout_position[perm]
            self.favored_spout = self.favored_spout[perm]
            self.behavior.spout_positions = self.behavior.spout_positions[perm]

        # Which sources to use for training/testing
        ncells = np.shape(dataset.Ct_n)[0]
        if cells_to_use is None:
            if ncells > max_cells_to_use:
                warnings.warn(str(ncells) +
                              ' sources found. ' +
                              'Using only top ' + str(max_cells_to_use))
                cells_to_use = dataset.p_ordering[:max_cells_to_use]
            else:
                # Use all cells
                cells_to_use = np.arange(ncells)
        else:
            if max(cells_to_use) >= ncells:
                raise ValueError('Invalid cells_to_use argument!')
        self.cells_to_use = cells_to_use

        # Save all traces for training (neurons x time x trials)
        self.traces = dataset.Ct_n[cells_to_use, :, :]

        # Train the classifiers
        self.train(seed=seed)

    @staticmethod
    def get_pre_odor_licks(behavior):
        """ Return the number of pre-odor licks to any spout on each trial. """
        trial_pre_odor_licks = np.zeros(behavior.ntrials)
        t_pre_odor = [0, behavior.stimulus_times[0]]
        inds = np.where((behavior.lick_mat_t >
                         t_pre_odor[0]) &
                        (behavior.lick_mat_t <
                         t_pre_odor[1]))[0]
        for spout in range(4):
            # Skip the disconnected spout
            if spout != 1:
                trial_pre_odor_licks += np.sum(
                    behavior.spout_lick_mat[spout][:, inds], 1)
        return trial_pre_odor_licks

    def train(self, n_trials=10, seed=22222, exclude_pre_odor_licks=False):
        """ Train/re-train the current model using only n clean_trials. """

        # Remove pre lick trials (don't by default)
        # this doesn't matter because we train the models
        # using all timepoints vs. just the pre-odor timepoints.
        # But it is CRITICAL to remove these trials from evaluation,
        # which we do.
        if exclude_pre_odor_licks:
            clean_trials = np.setdiff1d(self.clean_trials,
                                        self.pre_lick_trials)
        else:
            clean_trials = self.clean_trials

        # Fit the PLSRegression model to ALL TIMEPOINTS on training trials
        basis, training_trials = discriminate_spout_position(
            self.traces, self.behavior, clean_trials,
            use_pca=False, scale=False, n_trials=n_trials,
            k_comps=self.k_comps, seed=seed)

        # Check if any bad trials are in the training trials
        # bad trials are those where the active spout != the most-licked spout
        all_good = np.all(np.where(self.favored_spout[training_trials])[1] ==
                          np.where(self.spout_position[training_trials])[1])

        # We do this sequentially instead of just always getting rid of the
        # bad trials first to improve reproducibility vs. versions where we
        # didn't do this check (i.e. we have to change fewer figure panels
        # because only one dataset seems to suffer from this issue)
        if not all_good:
            # Only look at clean trials were the active spout
            # matches the most-licked-towards spout
            good = np.where(np.where(self.favored_spout[clean_trials])[1] ==
                            np.where(self.spout_position[clean_trials])[1])[0]
            clean_target_trials = clean_trials[good]
            basis, training_trials = discriminate_spout_position(
                self.traces, self.behavior, clean_target_trials, use_pca=False,
                scale=False, n_trials=n_trials, k_comps=self.k_comps,
                seed=seed)

        self.basis = basis
        self.training_trials = training_trials

        # Verify that all chosen training trials were selected from those
        # where the active spout was also the most-licked spout
        assert np.all(np.where(self.favored_spout[training_trials])[1] ==
                      np.where(self.spout_position[training_trials])[1])

        # Optimal classifier setpoint is chosen from the training trials
        # Note that argmax=False when we call predict_spout_position here
        # We pass in all timepoints (self.traces) and select the subset
        # of timepoints to use for inference with
        # the self.test_epoch parameter.
        pred_s = predict_spout_position(self.traces, self.basis,
                                        self.behavior,
                                        which_frames=self.test_epoch,
                                        argmax=False,
                                        fps=self.fps)
        setpoint = get_classifier_setpoint(pred_s, self.behavior,
                                           plot=False, trials=training_trials)
        self.setpoint = setpoint

    def predict(self, trials_to_use, epoch=None, cull_pre_odor=True,
                use_spout_position=False, exclude_no_go=False):
        """ Use the current model to predict spout positions. """
        self.used_spout_position = use_spout_position

        # Use the specified set of trials to evaluate the model
        if type(trials_to_use) == str and trials_to_use == 'all':
            # Use all trials
            trials_to_use = np.arange(self.behavior.ntrials)
        elif type(trials_to_use) == str and trials_to_use == 'test':
            # Use only non-training trials
            trials_to_use = np.setdiff1d(np.arange(self.behavior.ntrials),
                                         self.training_trials)
            if cull_pre_odor:
                trials_to_use = np.setdiff1d(trials_to_use,
                                             self.pre_lick_trials)
        elif type(trials_to_use) != np.ndarray and type(trials_to_use) != list:
            # Use a specified list
            raise ValueError('Must pass in indices or \'all\' or \'test\'')
        if exclude_no_go:
            no_go = np.where(self.behavior.trial_types == 4)[0]
            trials_to_use = np.setdiff1d(trials_to_use, no_go)
        self.predicted_trials = trials_to_use

        if epoch is None:
            epoch = self.test_epoch
        else:
            print('predict(): Overriding test_epoch with', epoch)
        tr = self.traces[:, :, trials_to_use]

        if not use_spout_position:
            warnings.warn('Quantifying prediction accuracy using action!')
            truth = self.favored_spout[trials_to_use]
        else:
            truth = self.spout_position[trials_to_use]

        # Get predictions at the optimal threshold/setpoint
        self.predictions = predict_spout_position(tr, self.basis,
                                                  self.behavior,
                                                  which_frames=epoch,
                                                  fps=self.fps,
                                                  setpoint=self.setpoint)

        # To get the AUC use the unthresholded predictions
        cat_pred = predict_spout_position(tr, self.basis, self.behavior,
                                          which_frames=epoch,
                                          fps=self.fps, argmax=False)

        # Compute macro AUC (the average across the 3 spout AUC values)
        fpr, tpr, roc_auc = decoder.multi_class_roc_auc(
            truth, cat_pred, do_plot=False)
        auc = np.mean([v for v in roc_auc.values()])

        return self.predictions, auc

    def plot_prediction_raster(self):
        """
        Plot predicted spout activations vs true behavior data.
        """
        if self.predictions is None:
            raise ValueError('Call model.predict() first!')

        # Scale factor for dot size
        plot_scale = 7e3

        plt.figure(figsize=(6*2, 1.5*2))
        licks_df = pd.DataFrame()
        make_labels = True
        lick_label = True

        for color, offset, rng_val, epoch in zip('rgb', [.7, .8, .9],
                                                 [self.t_pre_odor,
                                                  self.t_pre_reward,
                                                  self.t_post_reward],
                                                 ['Pre-odor', 'Pre-reward',
                                                 'Post-reward']):
            inds = np.where((self.behavior.lick_mat_t > rng_val[0]) &
                            (self.behavior.lick_mat_t < rng_val[1]))[0]

            trials_to_use = np.setdiff1d(np.arange(self.behavior.ntrials),
                                         self.training_trials)
            trials_to_use = np.setdiff1d(trials_to_use, self.pre_lick_trials)

            for spout in range(4):
                o = 1 if spout == 0 else 0
                lick_mat = self.behavior.spout_lick_mat[spout][:, inds]
                fraction = np.sum(lick_mat, axis=1) / len(inds)
                licks = np.where(fraction > 0)[0]
                total_licks = np.sum(lick_mat)

                latent_pos = np.where(
                    self.behavior.spout_positions == (spout+1))[0]
                latent_pos = np.intersect1d(trials_to_use, latent_pos)

                most_licked = np.where(
                    np.argmax(self.favored_spout, 1) == (spout))[0]
                most_licked = np.intersect1d(trials_to_use, most_licked)

                predictions = np.where(self.predictions == (spout))[0]
                predictions = np.intersect1d(trials_to_use, predictions)

                ll = 'Active spout' if make_labels else ''
                plt.plot(latent_pos, [spout+o+.2]*len(latent_pos), '.',
                         color='#4CBB17', label=ll, mew=0)

                ll = 'Most-licked spout (pre-reward)' if lick_label else ''
                plt.plot(most_licked, [spout+1]*len(most_licked), '.',
                         color='#00BFBF', label=ll, mew=0)
                lick_label = False

                if self.used_spout_position:
                    ll = 'Predicted active spout' if make_labels else ''
                else:
                    ll = 'Predicted preferred spout' if make_labels else ''
                plt.plot(predictions, [spout+0.8]*len(predictions), '.',
                         color='#BE29EC', lw=.2,
                         label=ll, mew=0)

                if spout != 1:
                    # Get licks during each epoch
                    d = {'Licks': total_licks,
                         'Epoch': epoch, 'Spout': spout+o}
                    licks_df = licks_df.append(d, ignore_index=True)

                make_labels = False
            plt.xlabel('Trial')
            plt.ylabel('Spout')
        sns.despine()
        plt.yticks([1, 2, 3])
        plt.legend()
        return licks_df

    def plot_trajectory(self, proj, ax, spout=0, plot_means=True):
        """ Helper function for plotting individual trajectories. """
        zorder = 4
        dim = np.arange(self.k_comps)

        if plot_means:
            # Mean trajectory
            for spout in proj.keys():
                proj_dir = np.mean(proj[spout], 0)
                x = np.squeeze(proj_dir[:, dim[0]])
                y = np.squeeze(proj_dir[:, dim[1]])
                color = self.colors[spout]
                if color == 'c':
                    zorder = 4
                elif color == 'orange':
                    zorder = 5
                elif color == 'r':
                    zorder = 6

                ax.plot(x, y, color, zorder=zorder)
                ax.plot(x[0], y[0], 'k.', zorder=10, markersize=10)
                if np.shape(proj_dir)[0] > self.odor_on:
                    ax.plot(x[self.odor_on], y[self.odor_on],
                            'g.', zorder=10, markersize=10)
                if np.shape(proj_dir)[0] > self.reward_on:
                    ax.plot(x[self.reward_on], y[self.reward_on],
                            'm.', zorder=10, markersize=10)
                self.odor_on
        else:
            # Single trial
            x = np.squeeze(proj[:, dim[0]])
            y = np.squeeze(proj[:, dim[1]])
            color = self.colors[spout]
            if color == 'c':
                zorder = 4
            elif color == 'orange':
                zorder = 5
            elif color == 'r':
                zorder = 6

            ax.plot(x, y, color, zorder=zorder)
            ax.plot(x[0], y[0], 'k.', zorder=1, markersize=10)
            if np.shape(proj)[0] > self.odor_on:
                ax.plot(x[self.odor_on], y[self.odor_on],
                        'g.', zorder=1, markersize=10)
            if np.shape(proj)[0] > self.reward_on:
                ax.plot(x[self.reward_on], y[self.reward_on],
                        'm.', zorder=1, markersize=10)
            ax.plot(proj[:, dim[0]], proj[:, dim[1]],
                    color, zorder=zorder)

    def check_selectivity(self, trial, trial_mode,
                          good_cutoff=0.7, bad_cutoff=0.3):
        """
        Check if a trial is "good" or "bad" depending on
        the specified good and bad cutoffs.
        """
        selectivity = self.behavior.get_on_target_licks(trial)
        if trial_mode is 'good':
            # Skip trials with lick selectivity below threshold
            if selectivity < good_cutoff:
                return False
        elif trial_mode is 'bad':
            # Skip trials if lick selectivity is too high
            if selectivity > bad_cutoff:
                return False
        elif trial_mode is 'all':
            # Use all trials independent of lick selectivity
            pass
        else:
            raise ValueError('Invalid trial-choosing mode')
        return True

    def plot_trajectories(self, plot_means=True, stop_frame=None,
                          keep_idx=None, chosen_trials=None,
                          axes_lims=None, clean_only=True,
                          min_block_trials=0, save_path=None,
                          good_cutoff=0.7, bad_cutoff=0.3, close_fig=False):
        """
        Plot neural trajectories in learned basis.

        :param stop_at_odor_on: bool. Plot full trajectory or stop at odor on?
        :param stop_frame: int or none. Stop plotting trajectory at this frame.
        :param keep_idx: list. Neurons to retain when generating trajectories.
                        None keeps all indices.
        :param axes_lims: list of length == dim. Axis limits.
        :param chosen_trials: list. Trials to plot when generating traj.
                        None keeps all indices.
        :param clean_only: bool. Only plot clean go trials. Default to no.
        :param return_reward_pts: bool. Return all 9 coords at the reward time.
        :param min_block_trials: int.
               Only plot trials that are > min_block_trials into a block.
        :param good_cutoff: 0.7, use if clean_only is true
        :param bad_cutoff: 0.3, use if clean_only is true

        :returns trials: Trials plotted (or averaged across) in each group.
        """
        type_codes = ['', '', 'explore', 'go', 'no go']

        print('Excluding training trials + those earlier than ',
              min_block_trials, ' within a block')

        traces = self.traces.copy()

        fig, axes = plt.subplots(figsize=(12, 3), ncols=3,
                                 sharex=True, sharey=True)
        if keep_idx is None:
            keep_idx = np.arange(np.shape(traces)[0])
        if chosen_trials is None:
            chosen_trials = np.arange(np.shape(traces)[2])

        zero_idx = np.setdiff1d(np.arange(np.shape(traces)[0]), keep_idx)
        traces[zero_idx, :, :] = 0

        spout_positions = self.behavior.spout_positions - 1
        neurons = np.shape(traces)[0]

        type_name = ['go', 'no go', 'go']
        trials_to_use = ['good', 'all', 'bad']
        titles = ['Go trials', 'No go trials', 'Incorrect go trials']
        trials = defaultdict(list)

        for ax, type_, trial_mode, title in zip(
                axes, type_name, trials_to_use, titles):
            all_traj = defaultdict(list)
            type_code = np.where(np.array(type_codes) == type_)[0]
            trial_idx = np.where(self.behavior.trial_types == type_code)[0]
            trial_idx = np.intersect1d(trial_idx, chosen_trials)
            used_trials = 0
            for trial in trial_idx:
                idx = spout_positions[trial]
                traj = [self.basis.transform(
                        np.reshape(traces[:, time, trial], (-1, neurons)))
                        for time in range(traces.shape[1])]
                traj = np.squeeze(traj)

                # Only show desired chunk of trajectory
                if stop_frame is not None:
                    traj = traj[:stop_frame, :]

                # Skip training trials
                if trial in self.training_trials:
                    continue

                # Keep clean trials only
                if clean_only:
                    selective = self.check_selectivity(trial, trial_mode,
                                                       good_cutoff, bad_cutoff)
                    if not selective:
                        continue

                # Exclude trials at the beginning of the block
                if (self.behavior.ind_within_block[trial] <
                        min_block_trials):
                    continue

                # Get rid of trials with licks during the pre-odor period
                if trial in self.pre_lick_trials:
                    continue

                # Plot the trajectory
                if not plot_means:
                    self.plot_trajectory(traj, ax, spout=idx,
                                         plot_means=plot_means)
                else:
                    all_traj[idx].append(traj)
                used_trials += 1
                trials[trial_mode].append(trial)
            print('n = ', used_trials, title)

            if plot_means:
                self.plot_trajectory(all_traj, ax)
            ax.set_title(title)
            sns.despine()

        axes[0].set_aspect(1)
        # If axis limits are provided, dress up the plot for movie
        if axes_lims is not None:
            # Plot a scale bar
            for axis in axes:
                axis.axis('off')
                l0 = axes_lims[0] + 2
                axis.plot([l0, l0+1], [l0, l0], 'k')
                axis.plot([l0, l0], [l0, l0+1], 'k')

            # Update axis limits
            axes[0].set_xlim(axes_lims)
            axes[0].set_ylim(axes_lims)

        if save_path is not None:
            plt.savefig(save_path)
        if not close_fig:
            plt.show()

        return trials

    def plot_lick_movie(self, cosmos_traces, save_path=None):

        plot_scale = 100
        traj_len = 205
        titles = ['Go trials', 'No go trials', 'Incorrect go trials']

        if self.predictions is None:
            print('Generating model predictions')
            _ = self.predict('test')
        licks = self.plot_prediction_raster()
        trials = self.plot_trajectories()
        traces = cosmos_traces

        spout_lick_times = self.behavior.spout_lick_times
        nframes = traces.C.shape[1]
        use_led_frames = True
        if use_led_frames:
            # Exclude final trial which is incomplete
            trial_onset_frames = traces.led_frames[:-1] - 1
        else:
            trial_onset_frames = traces.trial_onset_frames

        # Gather all licks.
        spout_lick_mat = defaultdict(list)
        spout_keys = [0, 2, 3]
        for spout in spout_keys:
            spout_lick_mat[spout] = np.zeros((np.shape(traces.Ct)[2],
                                             np.shape(traces.Ct)[1]))
            spout_trial_lick_times = spout_lick_times[spout]
            for trial in range(len(trial_onset_frames)-1):
                if trial in spout_trial_lick_times.keys():
                    lick_t = spout_trial_lick_times[trial]

                    lick_frames = np.floor(lick_t*traces.fps).astype(np.int)
                    spout_lick_mat[spout][trial, :] = np.histogram(
                        lick_frames, np.arange(np.shape(traces.Ct)[1] + 1))[0]

        # Draw lick plot

        # Iterate over rows to plot (trials towards different spouts)
        for stop_frame in np.arange(2, traj_len):
            fig, axes = plt.subplots(figsize=(12, 3.5), ncols=3, nrows=3,
                                     sharex=True, sharey=True)
            print(stop_frame)
            for ax_row, spout, spout_name in zip(axes,  [4, 3, 1], [3, 2, 1]):
                # Iterate over columns to plot (different trial types)
                for ax, title, trial_type in zip(
                        ax_row, titles, trials.keys()):
                    sx = 0
                    spout_trials_all = np.array(trials[trial_type])
                    for spout_curr in [0, 2, 3]:
                        tr = self.behavior.spout_positions[spout_trials_all]
                        trials_to_use = np.where(tr == spout)[0]
                        spout_trials = spout_trials_all[trials_to_use]
                        lick_mat = spout_lick_mat[spout_curr][spout_trials, :]
                        if len(spout_trials) == 0:
                            continue
                        summed_licks = np.mean(lick_mat, 0)
                        licks = np.where(summed_licks > 0)[0]
                        if len(licks) > 0:
                            licks_ = licks[np.where(licks < stop_frame)[0]]
                            ax.scatter(licks_, [sx+1.8]*len(licks_),
                                       s=summed_licks[licks_]*plot_scale,
                                       color=self.colors[spout_curr])
                            sx += 1
                    # Only plot titles in first row
                    if spout == 4:
                        ax.set_title(title)
                    ax.set_ylabel('Spout ' + str(spout_name))
                    ax.axvline(self.odor_on, color='g', zorder=0)
                    ax.axvline(self.reward_on, color='m', zorder=0)
                    ax.axvline(5, color=self.colors[spout-1],
                               zorder=0, linewidth=20)
                    ax.axvline(stop_frame, color='k', zorder=-1)
                    ax.set_yticks([])

                    if spout == 1:
                        ax.set_xlabel('Time (s)')
                        ax.set_xticks(np.linspace(0, 250, 6))
                        labels = np.round(ax.get_xticks() / self.fps, 2)
                        ax.set_xticklabels(labels)

            plt.xlim([0, traj_len])
            plt.ylim([1, 5])
            sns.despine()
            save_path_ = save_path + str(stop_frame) + '.png'
            plt.savefig(save_path_)
            plt.close()

    def get_second_trials(self, trial_idx, old_spout_positions):
        """ Get index of second trials """

        second_trials = np.where(
            self.behavior.ind_within_block == 1)[0]
        trial_idx = np.intersect1d(trial_idx, second_trials)
        print(len(trial_idx), 'all second trials')

        # Get frames used for computing 'after' trajectories
        block_len = int(self.reward_on - self.odor_on)
        rng = [self.reward_on, self.reward_on+block_len]
        assert np.diff(rng) == 44, 'Constants are changing!'

        pre_rng = [self.odor_on, self.reward_on]
        assert np.diff(pre_rng) == 44, 'Constants are changing!'

        # Choose how we select which second trials to use
        # a) Use all of them --> both options are FALSE
        # b) Use trials where most (> 50%) after licks
        #    on FIRST trial are to reward --> option 1 is TRUE
        # c) Use trials where after licks on SECOND trial
        #    are selective (> 80%) to reward --> option 2 is TRUE
        if False:
            print('Using second trials with clean first trials')
            # Only keep SECOND TRIALS where REWARD licks
            # ON THE FIRST TRIAL were towards the active spout
            first_trial = np.where(
                self.behavior.ind_within_block == 0)[0]
            v = [self.behavior.get_on_target_licks(trial,
                 frame_range=rng) for trial in first_trial]
            good_trials = first_trial[
                np.where(np.array(v) > 0.5)[0]] + 1
            trial_idx = np.intersect1d(trial_idx, good_trials)

        if False:
            print('Using selective second trials')
            # ALSO only keep second trials where REWARD licks on the
            # SECOND trial were towards the active spout
            v = [self.behavior.get_on_target_licks(trial,
                 frame_range=rng) for trial in second_trials]
            good_trials = second_trials[
                np.where(np.array(v) > DEFAULT_CLEAN_THRESHOLD)[0]]
            trial_idx = np.intersect1d(trial_idx, good_trials)

        if False:
            print('Using selective for old spout second trials')
            # Test only second trials where most PRE-ODOR licks were
            # selectively made towards the previous active spout.
            trial = second_trials[0]
            v = [self.behavior.get_on_target_licks(trial,
                 frame_range=pre_rng,
                 target_spout=old_spout_positions[trial]+1)
                 for trial in second_trials]
            good_trials = second_trials[
                np.where(np.array(v) > DEFAULT_CLEAN_THRESHOLD)[0]]
            trial_idx = np.intersect1d(trial_idx, good_trials)

        if False:
            print('Using selective second trials')
            # ALSO only keep second trials where most PRE-ODOR licks on the
            # SECOND trial were towards the active spout
            v = [self.behavior.get_on_target_licks(trial,
                 frame_range=pre_rng) for trial in second_trials]
            good_trials = second_trials[
                np.where(np.array(v) > DEFAULT_CLEAN_THRESHOLD)[0]]
            trial_idx = np.intersect1d(trial_idx, good_trials)

        # Print out info about after/reward period lick selectivity
        # of the second trials we're using (and previous trials)
        trial_copy = deepcopy(trial_idx)
        trial_copy = np.setdiff1d(trial_copy, self.pre_lick_trials)
        for trials, x in zip(
                [trial_copy, trial_copy-1], ['', 'TRIAL BEFORE']):
            v = [self.behavior.get_on_target_licks(trial,
                 frame_range=rng) for trial in trials]
            print('After period reward selectivity on', x,
                  '2nd trials =', np.round(np.min(v), 2),
                  '-', np.round(np.max(v), 2), ', mean =',
                  np.round(np.mean(v), 2), ', n =', len(v))
            print(x, '\n', v, '\n')

        # Also print total trial selectivity
        for trials, x in zip(
                [trial_copy, trial_copy-1], ['', 'TRIAL BEFORE']):
            v = [self.behavior.get_on_target_licks(trial,
                 frame_range=[0, self.reward_on]) for trial in trials]
            print('>>> BEFORE reward selectivity on', x,
                  '2nd trials =', np.round(np.min(v), 2),
                  '-', np.round(np.max(v), 2), ', mean =',
                  np.round(np.mean(v), 2), ', n =', len(v))
            print(x, '\n', v, '\n')

        return trial_idx

    def measure_centroids(self, cluster_time, traj_time, use_old=False):
        """
        Take each training trial to each spout, and compute its mean
        and covariance in the full-dimensional space. Also save the mean
        value of all other trials over a defined window of time so that
        we can compare those positions to the training clusters we found.

        :param cluster_time: len 2 list
               [start stop] for training time interval to define a cluster
        :param traj_time: len 2 list
               [start stop] for test data time interval to eval trajectories
        :param use_old: default False
               Use the previous spout label instead of the current one

        :returns saved_clusters: Dict containing the means and covariances of
        each cluster, as well as saved metrics from the test data
        :returns old_spout_positions: list of previous spout position at
        each trial.
        """

        # Use all of the data and trials
        traces = self.traces.copy()
        chosen_trials = np.arange(np.shape(traces)[2])

        dim = np.arange(self.k_comps)
        print('Using', len(dim), 'dimensions to compute centroids.')
        type_names = ['go', 'explore', 'nogo']
        type_codes = [3, 2, 4]

        # Get ground truth for defining clusters.
        spout_positions = self.behavior.spout_positions - 1

        # For defining what the "same" cluster is in the distance metric
        # use either the true spout position, or the previous spout position
        same_labels = deepcopy(spout_positions)
        old_spout_positions = deepcopy(spout_positions)
        for pos in range(len(spout_positions)):
            rolled_pos = deepcopy(spout_positions)
            while rolled_pos[pos] == spout_positions[pos]:
                rolled_pos = np.roll(rolled_pos, 1)
            old_spout_positions[pos] = rolled_pos[pos]
        if use_old:
            print('WARNING: USING OLD instead of current spout label')
            same_labels = old_spout_positions

        neurons = np.shape(traces)[0]
        saved_clusters = defaultdict(lambda: defaultdict(list))
        for type_, type_code in zip(type_names, type_codes):
            trial_idx = np.where(self.behavior.trial_types == type_code)[0]
            trial_idx = np.intersect1d(trial_idx, chosen_trials)

            # For explore trials, only keep trial #2 of the block
            if type_ == 'explore':
                trial_idx = self.get_second_trials(
                    trial_idx, old_spout_positions)

            # Process the trajectory on each chosen trial
            points = defaultdict(list)
            training_points = defaultdict(list)
            on_target_licks = defaultdict(list)
            trial_lag = defaultdict(list)
            eval_trials_used = defaultdict(list)
            old_spouts = defaultdict(list)

            # Ensure we have at least 3 trials of this type
            # on this mouse to process
            if len(trial_idx) >= 3:
                for trial in trial_idx:
                    traj = [self.basis.transform(
                            np.reshape(traces[:, time, trial], (-1, neurons)))
                            for time in range(traces.shape[1])]
                    traj = np.squeeze(traj)

                    if trial in self.training_trials:
                        # Save training data for computing clusters
                        idx = spout_positions[trial]
                        ci = slice(cluster_time[0], cluster_time[1])
                        training_points[idx].append(
                            np.mean(traj[ci, dim], 0))
                    else:
                        # Separately deal with the rest of the trials
                        # these are the trajectories to plot
                        idx = same_labels[trial]

                        # Get rid of trials with licks
                        # during the pre-odor period
                        if trial in self.pre_lick_trials:
                            continue

                        # Save average over trajectory (stopping at odor on)
                        ci = slice(traj_time[0], traj_time[1])
                        points[idx].append(
                            np.mean(traj[ci, dim], 0))
                        on_target_licks[idx].append(
                            self.behavior.get_on_target_licks(
                                trial, target_spout=same_labels[trial]+1))
                        trial_lag[idx].append(
                            self.behavior.ind_within_block[trial])
                        eval_trials_used[idx].append(trial)
                        old_spouts[idx].append(old_spout_positions[trial])

            if len(points) > 0 and type_ == 'go':

                # This iterates over the data for each spout
                for n in training_points.keys():

                    # Fit a gaussian to each set of TRAINING points
                    mean = np.mean(training_points[n], 0)
                    cov_ = np.cov((np.array(training_points[n]) - mean).T)

                    # For each active spout, save the mean trajectory
                    # location and the covariance of
                    # trajectory locations.
                    saved_clusters[type_]['mu'].append(mean)
                    saved_clusters[type_]['cov'].append(cov_)
                    saved_clusters[type_]['spout'].append(n)
            else:
                saved_clusters[type_]['mu'] = saved_clusters['go']['mu']
                saved_clusters[type_]['cov'] = saved_clusters['go']['cov']
                saved_clusters[type_]['spout'] = saved_clusters['go']['spout']

            # Also save the fraction of on-target licks on that trial,
            # and the index of the trial within its block
            saved_clusters[type_]['training points'] = \
                training_points
            saved_clusters[type_]['points'] = points
            saved_clusters[type_]['selectivity'] = \
                on_target_licks
            saved_clusters[type_]['block index'] = trial_lag
            saved_clusters[type_]['trial idx'] = eval_trials_used
            saved_clusters[type_]['old spout idx'] = old_spouts

        return saved_clusters, old_spout_positions

    def plot_confusion_matrix(self, normalize=True, use_position=True,
                              new_figure=True):
        """
        This function computes and plots the confusion matrix.

        :param training_trials: list. Optional list of trials to exclude
                           because they were used for learning the basis.
        :param normalize: bool. Normalization the confusion matrix.
        :param use_position: bool. Compute confusion matrix vs. actual
                            spout position (True)
                            or favored spout position (False)
        :param new_figure: bool. Make a new figure or use active axes.
        :return cm: ndarray. Confusion matrix
        """

        if new_figure:
            plt.figure()
        cmap = plt.cm.Purples

        # "Truth" can be behavior our actual spout position
        if use_position:
            truth = self.spout_position
            title = 'Spout Position'
        else:
            truth = self.favored_spout
            title = 'Favored lick direction'

        # Remove training trials + trials with pre-odor licks
        trials_to_use = np.setdiff1d(np.arange(self.behavior.ntrials),
                                     self.training_trials)
        trials_to_use = np.setdiff1d(trials_to_use, self.pre_lick_trials)
        predictions = self.predict('test')[0]
        truth = np.argmax(truth[trials_to_use], 1)

        # Generate the confusion matrix and normalize if necessary
        cm = confusion_matrix(truth, predictions)
        classes = np.unique(truth)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]),
                                      range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.xticks([0, 1, 2], ['spout 1', 'spout 2', 'spout 3'])
        plt.yticks([0, 1, 2], ['spout 1', 'spout 2', 'spout 3'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('')
        plt.tight_layout()

        return cm


def generate_lag_plot(datasets, fig_save_dir, mouse_colors_dict,
                      test_epoch='pre odor all',
                      num_perm=100, seed=11111,
                      alpha=0.1, max_avg_bins=15, point_size=20.7):
    """
    Look at behavioral + neural decoding performance as a function of
    the position of trials within a block.

    :param datasets: Dict containing ClassifierData objects for each mouse.
    :param fig_save_dir: Where to save plots?
    :param epoch: What data to use for model training? ('pre odor all')
    :param num_perm: Number of random permutations for each mouse (100)
    :param seed: Random number seed for reproducibility (11111)
    :param alpha: Transparency setting for data from individual mice (0.1)
    :param max_avg_bins: Use the same number of lag bins for each mouse (15)

    :return lag_scores, dataframe with r^2 values vs. shuffled controls
    :return b_auc, single mouse auc curves from classifier data
    :return c_auc, single mouse auc curves from neural data
    """
    # Set up the plot
    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1b = ax1.twinx()
    sns.set_context(rc={"lines.linewidth": 2})

    # Set random seed
    np.random.seed(seed)

    # Remove data with pre-odor licks
    # Note that doing this gets rid of a fair amount of data
    # and that leaves gaps in the data that we have to interpolate over
    # by averaging the two points around the missing point
    remove_pre_lick_data = True

    b_auc = []
    c_auc = []
    lag_scores = pd.DataFrame()
    for dx, dataset_id in enumerate(datasets.keys()):
        behavior = datasets[dataset_id].behavior
        x_range = range(max(behavior.ind_within_block))
        behavioral_auc = []
        classifier_auc = []
        pre_lick_selectivity = []

        # Train a model
        model = ClassifierModel(datasets[dataset_id], test_epoch)

        # Define pre-odor interval
        rng = [model.odor_on, model.reward_on]

        for block in x_range:
            if block > max_avg_bins:
                break
            # Get trials that we're processing
            # simultaneously--those sharing a lag+bin
            block_trials = np.where(behavior.ind_within_block == block)[0]

            # Exclude training trials
            clean_block_trials = np.setdiff1d(
                block_trials, model.training_trials)

            # Exclude trials with pre-odor licks---
            # doing this leaves gaps (lags with no data)
            # The analysis works well either way but excluding
            # this data is more conservative.
            if remove_pre_lick_data:
                clean_block_trials = np.setdiff1d(
                    clean_block_trials, model.pre_lick_trials)

            # Compute the AUC between the active spout
            # and the spout most licked pre-reward
            true = datasets[dataset_id].spout_position[
                clean_block_trials, :]
            block_favored = datasets[dataset_id].favored_spout[
                clean_block_trials, :]

            # If we do not have licks in all directions at this lag,
            # for this mouse, the multi AUC is invalid so we'll
            # fill in a NaN there.
            # This only happen if we exclude trials with pre-odor licks
            if np.min(np.sum(true, 0)) > 0:

                # Look at the behaviorally predicted active spout
                fpr, tpr, roc_auc = decoder.multi_class_roc_auc(
                    true, block_favored, do_plot=False)
                b_auc_curr = np.mean([v for v in roc_auc.values()])

                # Predict the active spout from the pre-odor neural data
                _, c_auc_curr = model.predict(clean_block_trials)

                # Get the fraction of pre-reward licks to the ACTIVE spout
                selectivity = [behavior.get_on_target_licks(tr, rng)
                               for tr in clean_block_trials]
                selectivity = np.mean(selectivity)
            else:
                print('Skipping lag', block, 'on dataset', dx,
                      'due to 0 licks in some direction')
                b_auc_curr = np.nan
                c_auc_curr = np.nan
                selectivity = np.nan

            behavioral_auc.append(b_auc_curr)
            classifier_auc.append(c_auc_curr)
            pre_lick_selectivity.append(selectivity)

        # Plot pre-lick-selectivity instead of behavioral AUC
        pre_lick_selectivity = np.array(pre_lick_selectivity)

        # Interpolate around nans caused by missing data
        # note that there is no missing data
        if remove_pre_lick_data:
            nan_idx = np.where(np.isnan(pre_lick_selectivity))[0]

            # Do assignments from copies of the data so we don't inadvertently
            # smooth over adjacent interpolated points, empirically there
            # aren't any such adjacent points, but it's better to do this
            # and be safe
            classifier_auc_ = classifier_auc.copy()
            selectivity_ = pre_lick_selectivity.copy()

            for idx in nan_idx:
                if idx+1 >= len(pre_lick_selectivity):
                    pre_lick_selectivity[idx] = selectivity_[idx-1]
                else:
                    pre_lick_selectivity[idx] = np.nanmean(
                        [selectivity_[idx-1], selectivity_[idx+1]])

            nan_idx = np.where(np.isnan(classifier_auc))[0]
            for idx in nan_idx:
                if idx+1 >= len(classifier_auc):
                    classifier_auc[idx] = classifier_auc_[idx-1]
                else:
                    classifier_auc[idx] = np.nanmean(
                        [classifier_auc_[idx-1], classifier_auc_[idx+1]])

        if dx == 0:
            ax1b.plot(pre_lick_selectivity, c=[27/255, 188/255, 189/255],
                      alpha=alpha, label='Licks (during odor, pre-reward)')
            ax1.plot(classifier_auc, c=[149/255, 81/255, 160/255],
                     alpha=alpha, label='Neural decoding (pre-odor)')
        else:
            ax1b.plot(pre_lick_selectivity,
                      c=[27/255, 188/255, 189/255], alpha=alpha)
            ax1.plot(classifier_auc, c=[149/255, 81/255, 160/255], alpha=alpha)

        # Save the auc values computed from the data
        b_auc.append(np.array(pre_lick_selectivity)[:max_avg_bins])
        c_auc.append(np.array(classifier_auc)[:max_avg_bins])

    # Lines
    acolor = [149/255, 81/255, 160/255]
    ax1.plot(np.mean(c_auc, 0), c=acolor, lw=1.485)
    bcolor = [27/255, 188/255, 189/255]
    ax1b.plot(np.mean(b_auc, 0), c=bcolor, lw=1.485)

    # Superimposed points
    ax1.plot(np.mean(c_auc, 0), '.', c=acolor, ms=10)
    ax1b.plot(np.mean(b_auc, 0), '.', c=bcolor, ms=10)

    ax1.set_xlabel('Block position (trials)')

    ax1.set_ylabel('Neural decoding of most-licked spout (AUC)', color=acolor)
    ax1.set_ylim([0.3, 1])
    ax1.tick_params(axis='y', labelcolor=acolor)

    ax1b.set_ylabel('Fraction of pre-reward licks to active spout',
                    color=bcolor)
    ax1b.set_ylim([0, .7])
    ax1b.tick_params(axis='y', labelcolor=bcolor)
    sns.despine()

    return (lag_scores, b_auc, c_auc)


def get_classifier_setpoint(predictions, bpod_data,
                            plot=False, trials=None):
    """
    Given a set of unthresholded spout positions, get the optimal
    classifier threshold/setpoint for each spout.

    :param predictions: ndarray. Size trials x spouts.
    :param bpod_data: BpodDataset. Behavior data corresponding to traces.
    :param plot: bool. Plot the computed ROC curve or not.
    :params trials: list. Trials to use for computing setpoint. By default
                    use all trials.

    :return setpoint: list. Optimal setpoint for each spout classifier.
    """

    true_pos = bpod_data.spout_positions.copy()
    true_pos[np.where(true_pos == 1)[0]] = 2
    true_pos_ = true_pos - 2
    true_pos = keras.utils.to_categorical(true_pos_)
    current_setpoint = []
    if trials is None:
        trials = range(np.shape(true_pos)[0])
    if plot:
        plt.figure()
    for x in range(np.shape(true_pos)[1]):
        fpr, tpr, thresh = roc_curve(true_pos[trials, x],
                                     predictions[trials, x],
                                     drop_intermediate=False)
        sp = thresh[np.argmax(tpr - fpr)]
        if plot:
            plt.plot(tpr, fpr, label=x)
        current_setpoint.append(sp)
    if plot:
        plt.legend()
    return current_setpoint


def discriminate_spout_position(traces, bpod_data, clean_trials, use_pca=False,
                                n_trials=10, k_comps=DEFAULT_K_COMPONENTS,
                                seed=22222, mode='all',
                                verbose=False, fps=None, scale=True,
                                return_data=False, trial_pre_odor_licks=None):
    """
    Get PCA or PLS model for discriminating spout positions from traces.

    :param traces: ndarray. Either CT.Ct or CT.St (fluorescence data).
    :param bpod_data: BpodDataset. Behavior data corresponding to traces.
    :param clean_trials: List of clean trials to choose training trials from.
    :param use_pca: bool. Use either PCA or Partial Least Squares Regression.
    :param n_trials: int. Number of clean trials in a given
                          lick direction to use for training.
    :param k_comps: int. Number of components to retain in dim reduction.
    :param seed: int. For controlling random sampling of trials to use.
    :param mode: string. What timepoints in the data should we train on.
                         Valid options are: 'all', 'pre odor'
    :param verbose: print debugging text
    :param fps: imaging frame rate (otherwise assume a default value)
    :param scale: normalize/center the data inputted to PLS? (default True)
    :param return_data: return the exact data inputted to PCA/PLS?
                        (default False)
    :param trial_pre_odor_licks: number of licks during pre-odor period on
                each trial if this is provided, we will exclude
                those trials with > 0 pre odor licks from training

    :return model: sklearn model. Either a PCA or PLSRegression model.
    :return training_trials: list. Trials used for learning basis for data.
    :return proj_data: Return the exact data inputted to PCA/PLS
                       (if return_data is True)
    """

    # To ensure we don't mess up the behavior data, make a copy of it
    bpod_data = deepcopy(bpod_data)

    if fps is None:
        if verbose:
            print('Setting fps to default!')
        fps = FPS
    odor_on = int(bpod_data.stimulus_times[0]*fps)

    # Set random seed
    np.random.seed(seed)

    spout_positions = bpod_data.spout_positions

    data = []
    labels = []
    training_trials = []
    pos_trials_all = []
    for idx, spout_position in enumerate(np.unique(spout_positions)):

        # Ensure they are "go" trials
        trial_idx = np.where(bpod_data.go_trials)[0]

        # Get trials where we licked to the given spout
        pos_trials = np.where(spout_positions == spout_position)[0]

        # Merge these criteria together
        pos_trials = np.intersect1d(pos_trials, trial_idx)
        pos_trials = np.intersect1d(clean_trials, pos_trials)

        # This line has no effect except to effectively change the RNG seed
        # for reproducibility leave it here
        np.random.permutation(pos_trials)

        # Use only k trials of each type for training
        if verbose:
            print(len(pos_trials))
        if len(pos_trials) >= n_trials:
            pos_trials = np.random.choice(pos_trials, n_trials, replace=False)
        elif verbose:
            print('Warning: fewer than', n_trials,
                  'exist for ', spout_position)
        pos_trials_all.append(pos_trials)

    # Equalize the number of trials of each type
    pos_length = [len(trials) for trials in pos_trials_all]
    pos_trials_all = [np.random.permutation(trials)[:np.min(pos_length)]
                      for trials in pos_trials_all]

    for idx, spout_position in enumerate(np.unique(spout_positions)):
        training_trials.append(pos_trials_all[idx])
        if verbose:
            print('Spout ', spout_position, 'trials =', pos_trials_all[idx])

        # Treat all timepoints as independent, each sample is of size 1 x cells
        if mode == 'all':
            data_block = np.reshape(traces[:, :,
                                    pos_trials_all[idx]].T,
                                    (-1, traces.shape[0]))
        elif mode == 'pre odor':
            data_block = np.reshape(traces[:, :odor_on,
                                    pos_trials_all[idx]].T,
                                    (-1, traces.shape[0]))
        else:
            raise ValueError('Invalid timepoint mode specified!!')
        data.append(data_block)
        label_block = np.zeros((data_block.shape[0], 3))
        label_block[:, idx] = 1
        labels.append(label_block)

    # Concatenate data
    proj_data = np.vstack(data)
    proj_labels = np.vstack(labels)

    # Fit basis
    pca = PCA(k_comps)
    pls = PLSRegression(k_comps, scale=scale)
    if use_pca:
        pca.fit(proj_data)
    else:
        pls.fit(proj_data, proj_labels)

    # Return results
    model = pca if use_pca else pls
    if return_data:
        return model, np.concatenate(training_trials), proj_data
    else:
        return model, np.concatenate(training_trials)


def predict_spout_position(traces, basis, bpod_data, which_frames='all',
                           frames=30, chosen_frame=0, argmax=True,
                           fps=None, keep_idx=None, setpoint=None):
    """
    Given a PCA or PLS basis, predict the spout position.

    :param traces: ndarray. Either CT.Ct or CT.St (fluorescence data).
    :param basis: sklearn model. Either a PCA or PLSRegression model.
    :param bpod_data: BpodDataset. Behavior data corresponding to traces.
    :param which_frames: string. 'all', 'pre odor', 'post odor'.
    :param frames: int. If which_frames is 'pre odor', use this many frames
                        before odor onset for prediction.
    :param keep_idx: list. Indices to retain when generating trajectories.
                     None keeps all indices.
    :param fps: float. Frame rate of imaging in Hz, defaults to 29.40950

    :return predictions: nparray. Predicted spout position for each trial.
    """
    traces = traces.copy()
    if keep_idx is None:
        keep_idx = np.arange(np.shape(traces)[0])
    zero_idx = np.setdiff1d(np.arange(np.shape(traces)[0]), keep_idx)
    traces[zero_idx, :, :] = 0

    if fps is None:
        print('Setting fps to default!')
        fps = FPS
    ntrials = np.shape(traces)[2]
    if argmax:
        predictions = np.zeros(ntrials)
    else:
        predictions = np.zeros((ntrials, bpod_data.nspouts))
    odor_on = int(bpod_data.stimulus_times[0]*fps)
    reward_on = int((bpod_data.stimulus_times[0] + 1.5)*fps)
    for trial in range(ntrials):
        if which_frames == 'all':
            data = np.reshape(np.mean(traces[:, :, trial], 1), (1, -1))
        elif which_frames == 'pre odor all':
            data = np.reshape(np.mean(traces[:, :odor_on, trial], 1), (1, -1))
        elif which_frames == 'pre odor':
            data = np.reshape(
                np.mean(traces[:, odor_on-frames:odor_on, trial], 1), (1, -1))
        elif which_frames == 'post odor':
            data = np.reshape(
                np.mean(traces[:, odor_on:odor_on+frames, trial], 1), (1, -1))
        elif which_frames == 'post odor all':
            data = np.reshape(np.mean(traces[:, odor_on:, trial], 1), (1, -1))
        elif which_frames == 'odor to reward':
            data = np.reshape(
                np.mean(traces[:, odor_on:reward_on, trial], 1), (1, -1))
        elif which_frames == 'reward all':
            data = np.reshape(
                np.mean(traces[:, reward_on:, trial], 1), (1, -1))
        elif which_frames == 'odor onset':
            data = np.reshape(traces[:, odor_on, trial], (1, -1))
        elif which_frames == 'reward onset':
            data = np.reshape(traces[:, reward_on, trial], (1, -1))
        elif which_frames == 'chosen frame':
            data = np.reshape(traces[:, chosen_frame, trial], (1, -1))

        if argmax:
            pred = basis.predict(data)
            if setpoint is None:
                print('No setpoint defined!' +
                      ' Using argmax on raw classifier output')
            else:
                for dim in range(len(pred)):
                    pred *= (1-np.array(setpoint))
            predictions[trial] = np.argmax(pred)

        else:
            predictions[trial, :] = basis.predict(data)

    return predictions


def get_region_neurons(cosmos_traces, p_ordering,
                       area, min_neurons):
    """
    Given cosmos_traces, a ranking of each source in discrimination ability,
    and an area (or 'all'). return the best sources (up to min_neurons number)

    -1 uses all of the neurons
    """

    # Get the sources in each region
    if area == 'all':
        region_neurons = np.arange(cosmos_traces.ncells)
    else:
        region_neurons = np.array(cosmos_traces.cells_in_region[
            cosmos_traces.regions[area]])

    # Order the region neurons by discriminibility
    region_neurons = region_neurons[
        np.argsort(p_ordering[region_neurons])]

    # -1 denotes all neurons
    if min_neurons == -1:
        min_neurons = len(region_neurons)

    # Return the indices we found
    return region_neurons[:min_neurons]


def compare_initial_conditions(datasets, cluster_time, traj_time,
                               mouse_colors_dict,
                               min_neurons, good_cutoff=0.70,
                               bad_cutoff=0.30, use_old=False, mode='Closest'):
    """
    Measure the average position of single-trial neural trajectories during
    the pre-odor period. Compute the distance from each of those points to the
    distributions generated by the single-trials taken from training data.

    :param datasets: a list of ClassifierData objects
    :param cluster_time: len 2 list
           [start stop] for training time interval to define a cluster
    :param traj_time: len 2 list
           [start stop] for test data time interval to eval trajectories
    :param min_neurons: number of sources to use from each area, -1 = use all
    :param good_cutoff: lick selectivity cutoff for good trials. default = 0.70
    :param bad_cutoff: lick selectivity cutoff for bad trials. default = 0.30
    :param use_old: default False
           Use the previous spout label instead of the current one
    :param mode: default Closest
           Compute Current/Old - MODE, MODE = {'Closest', 'Old', 'Third'}
    """

    # We never call predict so this argument doesn't matter
    # but normally we train on whole trials and predict using just pre-odor
    test_epoch = 'pre odor all'

    # Parameters for setting up 3 subplots
    trial_type = ['go', 'nogo', 'go', 'explore']
    trials_to_use = ['good', 'all', 'bad', 'all']
    titles = ['Go trials', 'No go trials', 'Incorrect go trials', '2nd trials']
    spout_colors = ['orange', 'c', 'r']
    areas = ['all', 'MO', 'SSp', 'VIS']

    distances = pd.DataFrame()
    area_dist = defaultdict(list)
    models = defaultdict(list)
    old_spouts_mice = defaultdict(list)
    for area in areas:
        print('Processing', area)
        # Part 1: Get each set of trajectory points + clusters
        saved_clusters_mice = defaultdict(list)
        for dataset_id in datasets:
            # Use the same number of neurons from all areas
            # including (all), note that -1 means use all sources available
            mn = min_neurons
            region_neurons = get_region_neurons(
                datasets[dataset_id].cosmos_traces,
                datasets[dataset_id].p_ordering, area, min_neurons)

            if len(region_neurons) == 0:
                print('skipped', area, cdx)
                continue

            model = ClassifierModel(datasets[dataset_id],
                                    test_epoch,
                                    cells_to_use=region_neurons)
            saved_clusters, old_spout_positions = model.measure_centroids(
                cluster_time, traj_time, use_old=use_old)
            saved_clusters_mice[dataset_id] = saved_clusters
            old_spouts_mice[dataset_id] = old_spout_positions

            clusters = saved_clusters['go']
            dd = np.zeros((3, 3))
            for ii, (m0, c0) in enumerate(
                    zip(clusters['mu'], clusters['cov'])):
                for jj, (m1, c1) in enumerate(zip(
                        clusters['mu'], clusters['cov'])):
                    dd[ii, jj] = mahalanobis(m0, m1, np.linalg.inv(c1))
            area_dist[area].append(np.sum(dd))
            models[dataset_id] = model

        # Part 2: get data for three figure subpanels:
        # 1. Use all GO trials.
        # 2. Use all NOGO trials.
        # 3. Use all GO trials with poor performance.
        for type_, trial_mode, title in zip(
                trial_type, trials_to_use, titles):

            # Iterate over each mouse
            for mouse in saved_clusters_mice.keys():

                clusters = saved_clusters_mice[mouse][type_]

                if clusters == []:
                    continue

                # Iterate over each active spout
                for point_name, point_spout in enumerate(
                        clusters['points']):

                    # Iterate over each trial where this spout is active
                    for point_idx, point in enumerate(
                            clusters['points'][point_spout]):

                        trial = clusters['trial idx'][point_spout][point_idx]
                        pos = models[mouse].behavior.spout_positions - 1
                        assert point_spout == pos[trial]

                        # Only save the closest distance
                        # to a heteronymous spout
                        other_dist = []
                        other_target = []
                        other_old_target = []
                        other_curr_target = []

                        # Iterate over each target_spout cluster on this trial
                        broke = False
                        for target_spout, (mc, vc) in enumerate(
                                zip(clusters['mu'], clusters['cov'])):

                            if trial_mode is 'good':
                                # Skip trials with low lick selectivity
                                if (clusters['selectivity'][point_spout]
                                        [point_idx] < good_cutoff):
                                    continue
                            elif trial_mode is 'bad':
                                # Skip trials if lick selectivity is too high
                                if (clusters['selectivity'][point_spout]
                                        [point_idx] > bad_cutoff):
                                    continue
                            elif trial_mode is 'all':
                                # Use all trials
                                pass
                            else:
                                raise ValueError(
                                    'Invalid source-choosing mode')

                            # Compute distance from this point to a cluster
                            distance = mahalanobis(point,
                                                   mc, np.linalg.inv(vc))

                            # This is the name of this cluster
                            keys = list(clusters['points'].keys())

                            # Skip empty conditions
                            if target_spout >= len(keys):
                                broke = True
                                break

                            if point_name == target_spout:
                                # Save the homonymous distance
                                spout_type = 'Same'
                                target_dist = distance
                                d_same = {'Point identity': point_name + 1,
                                          'Target identity': target_spout + 1,
                                          'Spout comparison': spout_type,
                                          'Point index': point_idx,
                                          'Trial index': trial,
                                          'Distance': distance,
                                          'Area': area,
                                          'Trial group': title,
                                          'Mouse': mouse_colors_dict[mouse][1]}
                            else:
                                # Save the other heteronymous distances
                                # This is the name of the last active spout
                                old_spout = clusters['old spout idx'][
                                    point_spout][point_idx]
                                other_old_target.append(old_spout)
                                if not use_old:
                                    assert old_spout != point_spout
                                other_curr_target.append(keys[target_spout])
                                other_dist.append(distance)
                                other_target.append(target_spout)

                        # Save the appropriate other distance
                        if len(other_dist) > 0 and not broke:

                            # Append the same distance
                            distances = distances.append(
                                d_same, ignore_index=True)

                            # Process the other distance
                            spout_type = 'Different'

                            # Compute Same - MODE
                            # MODE = {'Closest', 'Old', 'Third'}
                            if mode == 'Closest':
                                # Get the closest heteronymous distance
                                o_idx = np.argmin(other_dist)
                            elif (mode == 'Old') or (mode == 'Third'):
                                # Or get the distance to the previous spout
                                o_idx = np.where((np.array(other_curr_target) - np.array(other_old_target)) == 0)[0][0]
                                if mode == 'Third':
                                    o_idx = not o_idx
                            else:
                                raise ValueError('Invalid mode!')

                            d = {'Point identity': point_name + 1,
                                 'Target identity': other_target[o_idx] + 1,
                                 'Spout comparison': spout_type,
                                 'Point index': point_idx,
                                 'Trial index': trial,
                                 'Distance': other_dist[o_idx],
                                 'Area': area,
                                 'Trial group': title,
                                 'Mouse': mouse_colors_dict[mouse][1]}
                            distances = distances.append(
                                d, ignore_index=True)

    return distances, area_dist


def quantify_variance_explained(datasets, min_neurons,
                                clean_threshold=DEFAULT_CLEAN_THRESHOLD,
                                cull_pre_odor_licks=False):
    """
    For both PCA and PLS compare the amount of variance explained
    as a function of area
    :param datasets: list of ClassifierData objects
    :param min_neurons: number of sources to use from each area
    """
    areas = ['MO', 'SSp', 'VIS', 'all']
    var_explained_sum = pd.DataFrame()
    for cdx in datasets.keys():
        for area in areas:

            # Get traces from the current area
            cd = datasets[cdx]
            region_idx = get_region_neurons(cd.cosmos_traces, cd.p_ordering,
                                            area, min_neurons)

            if len(region_idx) == 0:
                print('skipped', area, cdx)
                continue
            traces = cd.Ct_n[region_idx, :, :]

            # Get clean trials
            clean_trials = cd.behavior.get_clean_trials(
                clean_threshold, verbose=False)

            # Only use train + evaluate on trials where
            # there are no pre-odor licks
            if cull_pre_odor_licks:
                trial_pre_odor_licks = ClassifierModel.get_pre_odor_licks(
                    cd.behavior)
                zero_pre_lick_trials = np.where(trial_pre_odor_licks <= 0)[0]
                clean_trials = np.intersect1d(clean_trials,
                                              zero_pre_lick_trials)

            good = np.where(np.where(cd.favored_spout[clean_trials])[1] ==
                            np.where(cd.spout_position[clean_trials])[1])[0]
            clean_trials = clean_trials[good]

            # Compute variance explained for PCA
            pca, _, data1 = discriminate_spout_position(traces,
                                                        cd.behavior,
                                                        clean_trials,
                                                        scale=False,
                                                        use_pca=True,
                                                        return_data=True)

            # Zero-center the data
            X = data1 - np.mean(data1, 0)

            # Compute the variance of the data projected against each component
            component_variance_pca = np.var(np.dot(pca.components_, X.T),
                                            axis=1)

            # Compute variance explained for PLS
            pls, _, data2 = discriminate_spout_position(traces,
                                                        cd.behavior,
                                                        clean_trials,
                                                        scale=False,
                                                        use_pca=False,
                                                        return_data=True)
            assert np.all(data1 == data2)

            # Compute the PLS variance explained
            component_variance_pls = np.var(np.dot(X, pls.x_rotations_),
                                            axis=0)

            # Compute the total variance of the data
            tv = np.sum(np.var(X, axis=0))
            component_variance_ratio_pca = component_variance_pca / tv
            component_variance_ratio_pls = component_variance_pls / tv

            d = {'Dataset': cdx, 'Basis': 'PCA',
                 'Total variance explained':
                 np.sum(component_variance_ratio_pca),
                 'Area': area}
            var_explained_sum = var_explained_sum.append(d, ignore_index=True)

            d = {'Dataset': cdx, 'Basis': 'PLS',
                 'Total variance explained':
                 np.sum(component_variance_ratio_pls),
                 'Area': area}
            var_explained_sum = var_explained_sum.append(d, ignore_index=True)

    # Make the figure
    plt.figure(figsize=(3, 3))
    ax = plt.subplot(111)
    pca_var = var_explained_sum.loc[var_explained_sum['Basis'] == 'PCA']
    pls_var = var_explained_sum.loc[var_explained_sum['Basis'] == 'PLS']
    lpca = sns.pointplot(data=pca_var, y='Total variance explained',
                         x='Area', color='k', order=areas, ci=99)
    plt.setp(ax.collections, sizes=[50])
    lpls = sns.pointplot(data=pls_var, y='Total variance explained',
                         x='Area', color='#bcbec0', order=areas, ci=99)
    plt.setp(ax.collections, sizes=[50])
    plt.legend((lpca.lines[0], lpls.lines[-1]), ('PCA', 'PLS'))
    sns.despine()
    k = str(DEFAULT_K_COMPONENTS)
    plt.ylabel('Total variance explained\n (k=' + k + ' components)')

    # Compute stat for each area individually (pooling across mice)
    print('1) var_explained(PCA) > var_explained(PLS), for each area')
    pls = []
    pca = []
    for dataset in np.unique(pls_var['Dataset']):
            for area in np.unique(pls_var['Area']):
                pt = pls_var.loc[(pls_var['Dataset'] == dataset) &
                                 (pls_var['Area'] == area)][
                                     'Total variance explained']
                if len(pt) > 1:
                    raise ValueError('Error parsing point')
                if len(pt) == 0:
                    print('Skipping dataset', dataset, area)
                    continue
                pls.append(np.array(pt)[0])
                pt = pca_var.loc[(pca_var['Dataset'] == dataset) &
                                 (pca_var['Area'] == area)][
                                     'Total variance explained']
                if len(pt) > 1:
                    raise ValueError('Error parsing point')
                pca.append(np.array(pt)[0])
    p = stats.ttest_rel(pls, pca).pvalue
    print('paired t-test pls vs pca', ' p =', p)

    # Compute stat to compare motor variance explaiend to all other areas
    print('2) var_explained(PLS_MO) > var_explained(PLS_OTHERS)')
    for area in np.unique(pls_var['Area']):
        pls_mo = np.array(pls_var.loc[
            pls_var['Area'] == 'MO']['Total variance explained'])
        pls_area = np.array(pls_var.loc[
            pls_var['Area'] == area]['Total variance explained'])
        p = stats.ttest_rel(pls_mo, pls_area).pvalue
        print(area, 'v MO; n mice =',
              len(pls_area), ' p =', np.round(p, 6))


def compute_bases_cosine_distance(time_basis, comps):
    """
    Compute and plot the cosine distance between
    the basis vectors computed at each timepoint.
    :param time_basis:
    :param comps:
    :return:
    """
    # Compute cosine distance between the basis vectors
    bins = len(time_basis)
    plt.figure(figsize=(15, 5))
    for dim in range(comps):
        plt.subplot(1, comps, dim+1)
        basis_distance = np.zeros((bins, bins))
        for idx in range(bins):
            for jdx in range(bins):
                d = cosine(time_basis[idx].x_weights_[:, dim],
                           time_basis[jdx].x_weights_[:, dim])
                basis_distance[idx, jdx] = d

        plt.imshow(basis_distance, aspect='equal', cmap='afmhot')
        plt.colorbar(label='cosine distance')
        plt.title('basis dim = ' + str(dim))


def plot_mean_windowed_prediction_matrix(auc_mats,
                                         ax_lims,
                                         lick_timing,
                                         do_normalize,
                                         fig_save_dir=None,
                                         region=None):
    """
    Run plot_windowed_prediction_matrix() on the mean
    prediction matrix across datasets.
    :param auc_mats: output from test_all_classifiers_on_windowed_time_bins()
    :param ax_lims: [start_t, end_t] in seconds. Used for making xlim and ylim.
    :param lick_timing: [lick_onsets, lick_offsets] where lick_onsets contains
                        the time of lick onset for each dataset. Used
                        for overlaying dashed lines on the plot. Should be
                        in the same units etc. as ax_lims.
    :param do_normalize: Where the AUC scores were normalized.
    :param fig_save_dir: If not None, saves plot to this directory.
    :param region: Specifies which brain region the neurons were used.
    :return: Nothing
    """
    plt.figure()
    s = np.zeros(auc_mats[0].shape)
    for m in auc_mats:
        if do_normalize:
            # Normalize for a given trained basis
            m = m / np.max(m, axis=1)[:, np.newaxis]

            # Normalize for a given testing dataset
            # m = m / np.max(m, axis=0)[np.newaxis, :]

        s = s + m

    s = s / len(auc_mats)

    mouse_lick_onsets = lick_timing[0]
    mouse_lick_offsets = lick_timing[1]

    lick_onset = np.mean(mouse_lick_onsets)
    lick_offset = np.mean(mouse_lick_offsets)

    plot_windowed_prediction_matrix(s,
                                    do_normalize,
                                    ax_lims,
                                    lick_onset,
                                    lick_offset)

    plt.ylabel('training timepoint [s]')
    plt.xlabel('predicted timepoint [s]')
    # plt.colorbar(fraction=0.02, pad=0.14)

    if fig_save_dir is not None:
        suffix = '{}-block_auc_averaged_norm{}.pdf'.format(region,
                                                           int(do_normalize))

        save_path = os.path.join(fig_save_dir, suffix)
        plt.tight_layout()
        plt.gcf().set_size_inches(w=2, h=2)  # Control size of figure in inches
        plt.savefig(save_path)
        print('Saving to: {}'.format(save_path))


def plot_windowed_prediction_matrix(a, do_normalize, t_range, lick_onset,
                                    lick_offset):
    """
    Plot matrix representing how well a model trained on one time bin
    of the trial can predict other time bins.

    :param a: [ntimebins x ntimebins] Trained-on x Tested-on.
    :param do_normalize: bool. If true, then normalize each row (i.e.
                         normalize scores across tests for one training
                         set).
    :param t_range: [start_t, end_t] in seconds. Sets the scale of the image.
                    Can center this such that 0 corresponds to odor onset.
    :param lick_onset: Average lick onset for this dataset, relative to the
                       origin prescribed by a.
    :param lick_offset: Average lick offset for this dataset, relative to the
                       origin prescribed by a.
    :return: Nothing.
    """
    if do_normalize:
        # Normalize for a given trained basis.
        auc_mat = a / np.max(a, axis=1)[:, np.newaxis]
        # Normalize for a given testing dataset.
        # auc_mat = a / np.max(a, axis=0)[np.newaxis, :]
        clim = [0.5, 1]
    else:
        auc_mat = a
        clim = [0.5, 0.9]
    start_t = t_range[0]
    end_t = t_range[1]
    plt.imshow(auc_mat, cmap=plt.cm.plasma,
               extent=[start_t, end_t, end_t, start_t], clim=clim)
    plt.ylabel('training timepoint [s]')
    plt.xlabel('prediction timepoint [s]')
    plt.colorbar(pad=0.15)
    plt.xticks([-2, 0, 2, 4])
    plt.yticks([-2, 0, 2, 4])

    plt.axvline(lick_onset, color='k', linestyle='--', linewidth=1)
    plt.axhline(lick_onset, color='k', linestyle='--', linewidth=1)

    plt.axvline(lick_offset, color='k', linestyle='--', linewidth=1)
    plt.axhline(lick_offset, color='k', linestyle='--', linewidth=1)

    t = lick_offset
    ann = plt.annotate('', (t, start_t), (t, start_t - 1), va='center',
                       ha='center',
                       arrowprops=dict(arrowstyle='->', color='k'))
    ann = plt.annotate('', (end_t, t), (end_t + 1, t), va='center',
                       ha='center',
                       arrowprops=dict(arrowstyle='->', color='k'))
    t = lick_onset
    ann = plt.annotate('', (t, start_t), (t, start_t - 1), va='center',
                       ha='center',
                       arrowprops=dict(arrowstyle='->', color='r'))
    ann = plt.annotate('', (end_t, t), (end_t + 1, t), va='center',
                       ha='center',
                       arrowprops=dict(arrowstyle='->', color='r'))
    ann.set_annotation_clip(False)


def get_lick_onsets(datasets, dataset_ids, do_plot=False):
    """
    Estimate the average time of licking onset and offset
    within a trial, for each mouse.
    :param datasets: dict of classifier_data class instances.
    :param dataset_ids: the keys into the datasets dict.
    :param do_plot: bool.
    :return:
    """
    mouse_lick_rates = []
    mouse_lick_onsets = []
    mouse_lick_offsets = []

    for d_ind in range(len(dataset_ids)):
        BD = datasets[dataset_ids[d_ind]].behavior
        all_licks = np.copy(BD.spout_lick_rates[0])
        for i in range(1, 3):
            all_licks += BD.spout_lick_rates[i]
        lick_rate = np.sum(all_licks, axis=0)
        lick_rate = scipy.signal.savgol_filter(lick_rate, 3, 1)

        mouse_lick_rates.append(lick_rate)
        onset = np.min(np.where(lick_rate > np.max(lick_rate) / 5))
        offset = np.max(np.where(lick_rate > np.max(lick_rate) / 5))

        mouse_lick_onsets.append(onset)
        mouse_lick_offsets.append(offset)
        if do_plot:
            plt.plot(lick_rate)
            plt.plot(onset, lick_rate[onset], 'o')
            plt.plot(offset, lick_rate[offset], 'o')

    return mouse_lick_onsets, mouse_lick_offsets


def train_classifier_on_windowed_time_bins(dataset, region='all',
                                           bin_size=5,
                                           use_spikes=True):
    """
    Train multiple PLS bases that discriminate active spout,
    trained based on successive time windows of neural data.

    :param dataset: classifier_data object instance.
    :param region: string. 'all', 'MO', 'SSp', 'VIS', 'PTLp', 'RSP'
    :param bin_size: int. size (in neural data frames) of window for training.
    :param use_spikes: bool. If false, then use
                       smoothed-but-not-deconvolved traces.
    :return trained_output: dict containing
                bases: List of trained sklearn PLS objects for each time.
                training_trials: The trial indices used during training.
                time_bins: The time bins used for training each model.
    """

    # These parameters match the rest of the analyses
    comps = DEFAULT_K_COMPONENTS
    thresh = DEFAULT_CLEAN_THRESHOLD
    scale = False

    # Versus other analyses, we MUST remove pre odor licks from training here
    # elsewhere we don't do this because we train over all timepoints
    # in each trial anyway (so therefore include licks) and need more data
    # to make many of the analyses work.
    cull_pre_odor_licks = True

    # Set behavior and neural data.
    BD = dataset.behavior
    if use_spikes:
        tr_mc = dataset.St_n
    else:
        tr_mc = dataset.Ct_n

    # Define the sources to use for training.
    CT = dataset.cosmos_traces
    if region is 'all':
        which_cells = np.arange(CT.ncells)
    else:
        which_cells = np.array(CT.cells_in_region[CT.regions[region]])
    print(which_cells.shape)

    # Set up binning.
    k = np.shape(tr_mc)[1] / bin_size
    time_bins = np.array(np.linspace(0, np.shape(tr_mc)[1], k), dtype=int)

    # Get clean trials
    clean_trials = dataset.behavior.get_clean_trials(thresh, verbose=False)
    # Only use train + evaluate on trials where
    # there are no pre-odor licks
    trial_pre_odor_licks = ClassifierModel.get_pre_odor_licks(
        dataset.behavior)
    pre_lick_trials = np.where(trial_pre_odor_licks > 0)[0]
    zero_pre_lick_trials = np.where(trial_pre_odor_licks <= 0)[0]
    if cull_pre_odor_licks:
        clean_trials = np.intersect1d(clean_trials,
                                      zero_pre_lick_trials)

    # Get rid of training trials where action and latent spout don't match
    good = np.where(np.where(dataset.favored_spout[clean_trials])[1] ==
                    np.where(dataset.spout_position[clean_trials])[1])[0]
    clean_trials = clean_trials[good]

    # Train a unique basis for each timepoint bin.
    bases = {}
    np.random.seed(11111)
    for t_idx in range(len(time_bins) - 1):
        tr_t = tr_mc[which_cells, time_bins[t_idx]:time_bins[t_idx + 1], :]

        # Fit the classifier/basis to each grouping of traces.
        # Note we are doing scaling previously so scale=False.
        (bases[t_idx],
         training_trials) = discriminate_spout_position(
             tr_t, BD, clean_trials, k_comps=comps, scale=False)
        # try scale true, false
        # try thresh 0.7, 0.8
        # try cull_pre_odor_licks false true
        # try comps = 3, comps = 5

    trained_output = {'bases': bases, 'training_trials': training_trials,
                      'pre_lick_trials': pre_lick_trials,
                      'time_bins': time_bins}
    return trained_output


def test_classifiers_on_windowed_time_bins(dataset, time_basis,
                                           training_trials,
                                           pre_odor_trials,
                                           region, use_spikes=True,
                                           remove_pre_lick_data=True):
    """
    Test the prediction performance of a classifier trained on each time window
    of neural data (see train_classifier_on_windowed_time_bins()) on the
    neural data of all of the other time bins.
    :param dataset: classifier_data object instance.
    :param time_basis: trained sklearn PLS objects for the specified time.
    :param training_trials: the trial indices used during training.
    :param pre_odor_trials: the trials that contain pre-odor licks.
    :param region: string. 'all', 'MO', 'SSp', 'VIS', 'PTLp', 'RSP'
    :param use_spikes: bool. If false, use smoothed-but-not-deconvolved traces.
    :param remove_pre_lick_data: bool. If true, don't eval pre lick trials
    :return: [ntrained_bases x ntested_bases]. AUC (area under ROC curve)
             performance for each trained_basis vs. test_window comparison.

    """
    if use_spikes:
        tr_mc = dataset.St_n
    else:
        tr_mc = dataset.Ct_n

    BD = dataset.behavior
    CT = dataset.cosmos_traces

    if region is 'all':
        which_cells = np.arange(CT.ncells)
    else:
        which_cells = np.array(CT.cells_in_region[CT.regions[region]])

    # Define test trials and corresponding labels
    test_trials = np.ones(tr_mc.shape[2]).astype(bool)
    test_trials[training_trials] = 0

    # Get rid of pre-lick-trials
    if remove_pre_lick_data:
        test_trials[pre_odor_trials] = 0

    # Get rid of training trials where action and latent spout don't match
    bad = np.where(np.where(dataset.favored_spout[test_trials])[1] !=
                   np.where(dataset.spout_position[test_trials])[1])[0]
    test_trials[bad] = 0

    # This gets rid of almost all of the trials, but the stats still work out.
    # It's good the stats work out this way because it would make sense that
    # the effect would be strongest on trials with stereotyped lick behavior.
    # The block picture looks much noisier though, since we have so little data
    # we have like ~15 trials per mouse instead of > 100
    """
    good_cutoff=0.7
    for trial in np.arange(len(test_trials)):
        # Only choose selective trials
        if dataset.behavior.get_on_target_licks(trial) >= good_cutoff:
            test_trials[trial] = 0

        # Only choose go trials (3 is go)
        if dataset.behavior.trial_types[trial] != 3:
            test_trials[trial] = 0
    """

    test_labels = BD.spout_positions[test_trials]
    test_labels[test_labels == 1] = 0
    test_labels[test_labels == 3] = 1
    test_labels[test_labels == 4] = 2
    y_true = keras.utils.to_categorical(test_labels)

    # Now compute test score at each timepoint.
    which_t = np.arange(len(time_basis)) * 5  # Timepoints at which to test.
    auc_mat = np.zeros((len(time_basis), len(which_t)))
    for t_idx in range(len(time_basis)):
        tt_auc = np.zeros(len(which_t))
        for ind, tt in enumerate(which_t):
            tr_t = tr_mc[:, tt, test_trials][which_cells, :]
            data = tr_t.T
            prediction = time_basis[t_idx].predict(data)

            fpr, tpr, roc_auc = decoder.multi_class_roc_auc(y_true,
                                                            prediction,
                                                            do_plot=False)
            auc = np.mean([v for v in roc_auc.values()])
            tt_auc[ind] = auc
        auc_mat[t_idx, :] = tt_auc

    return auc_mat


def compare_time_window_blocks_predictions(auc_mats,
                                           pre_bins,
                                           peri_bins,
                                           post_bins,
                                           do_normalize):
    """

    :param auc_mats: List of auc_mats (for each dataset), as returned from
                     test_classifiers_on_windowed_time_bins().
    :param pre_bins: nparray. Time windows considered to be 'pre-task'.
    :param peri_bins: nparray. Time windows considered to be 'pre-task'.
    :param post_bins: nparray. Time windows considered to be 'post-task'.
    :param do_normalize: bool. If true, then normalize each row of auc_mat
                        (i.e.
                         normalize scores across tests for one training
                         set).
    :return: [3 x 3 x num_datasets]. [trained_on x tested_on x dataset]. Avg
             prediction performance between each of the blocks of time windows,
             for each dataset.
    """
    block_comparison = np.zeros((3, 3, 4))
    for d_ind, a in enumerate(auc_mats):
        if do_normalize:
            a = a / np.max(a, axis=1)[:, np.newaxis]

        block_comparison[0, 0, d_ind] = np.mean(a[pre_bins, :][:, pre_bins])
        block_comparison[0, 1, d_ind] = np.mean(a[pre_bins, :][:, peri_bins])
        block_comparison[0, 2, d_ind] = np.mean(a[pre_bins, :][:, post_bins])

        block_comparison[1, 0, d_ind] = np.mean(a[peri_bins, :][:, pre_bins])
        block_comparison[1, 1, d_ind] = np.mean(a[peri_bins, :][:, peri_bins])
        block_comparison[1, 2, d_ind] = np.mean(a[peri_bins, :][:, post_bins])

        block_comparison[2, 0, d_ind] = np.mean(a[post_bins, :][:, pre_bins])
        block_comparison[2, 1, d_ind] = np.mean(a[post_bins, :][:, peri_bins])
        block_comparison[2, 2, d_ind] = np.mean(a[post_bins, :][:, post_bins])

    return block_comparison


def plot_block_comparison_compare_vs_chance(block_comparison,
                                            dataset_ids,
                                            mouse_colors_dict,
                                            plot_labels,
                                            fig_save_dir=None,
                                            region=None,
                                            do_normalize=False):
    """
    Plot the output of compare_time_window_blocks_predictions().
    Compute statistics against chance performance AUC of 0.5
    :param block_comparison: [3 x 3 x num_datasets].
                             [trained_on x tested_on x dataset]. Average
                             prediction performance between each of the
                             blocks of time windows, for each dataset.
    :param dataset_ids: list of ints.
    :param mouse_colors_dict: Keys are dataset_ids.
                        Values are tuples with (names, colors) for plotting.
    :param plot_labels: The name of each of the blocks in block_comparison.
    :param fig_save_dir: If not None, save the plot to this directory.
    :param region: The name of the brain region used for this classification.
    :param do_normalize: bool. If true, then normalize each row of auc_mat (ie
                         normalize scores across tests for one training
                         set).
    :return: pvals: result of ttest_rel (multiple comparison corrected)
                    for the performance of a given trained basis on
                    the test time blocks.
    """

    # Acceptable FDR for benjamini post-hoc test
    alpha = 0.05

    plt.figure(figsize=(9, 4))
    plot_labels = ['pre', 'peri', 'post']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        for d, d_id in enumerate(dataset_ids):
            plt.plot([1, 2, 3], block_comparison[i, :, d], 'o',
                     label=mouse_colors_dict[d_id][1],
                     color=mouse_colors_dict[d_id][0])
        plt.plot([1, 2, 3], np.median(block_comparison[i, :, :], axis=1), '_r',
                 markersize=10)
        if not do_normalize:
            plt.ylim([0.40, 0.85])
        plt.xlim([0.5, 3.5])

        plt.xticks([1, 2, 3], ['pre', 'peri', 'post'])
        plt.xlabel('Test timepoints')
        if i == 0:
            if do_normalize:
                plt.ylabel('Normalized AUC')
            else:
                plt.ylabel('AUC')
        if i == 2:
            plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.axhline(0.5, linestyle='--', color='k')

    # Compare decodability auc against chance performance (0.5)
    pvals = np.zeros((3, 3))
    for test_ind in range(3):
        for train_ind in range(3):
            h, p = scipy.stats.ttest_1samp(
                block_comparison[train_ind, test_ind, :], 0.5)
            pvals[train_ind, test_ind] = p

    # Correct P-values (3 tests run simultaneously)
    for row in range(3):
        r, pc, _, _ = mt.multipletests(pvals[row, :],
                                       alpha=alpha, method='fdr_bh')
        pvals[row, :] = pc

    # Add pvalues to the subplot titles.
    for train_ind in range(3):
        plt.subplot(1, 3, train_ind + 1)
        plt.title(
            'Train on {}\n ttest1s p: \n 0:{:.3f}, 1:{:.3f}, 2:{:.3f}'.format(
                plot_labels[train_ind],
                pvals[train_ind, 0],
                pvals[train_ind, 1],
                pvals[train_ind, 2]))

    if fig_save_dir is not None:
        save_path = os.path.join(
            fig_save_dir,
            '{}-block_boxplot_comparison_vs_CHANCE_norm{}.pdf'.format(
                region, int(do_normalize)))
        plt.tight_layout()
        plt.gcf().set_size_inches(w=4, h=2)  # Control size of figure in inches

        plt.savefig(save_path)

    plt.figure()

    return pvals


def plot_block_comparison_group_by_train(block_comparison,
                                         dataset_ids,
                                         mouse_colors_dict,
                                         plot_labels,
                                         fig_save_dir=None,
                                         region=None,
                                         do_normalize=False):
    """
    Plot the output of compare_time_window_blocks_predictions().
    Within each subplot, compare the performance of bases
    Tested on each block but all Trained on the same block.

    :param block_comparison: [3 x 3 x num_datasets].
                             [trained_on x tested_on x dataset]. Average
                             prediction performance between each of the
                             blocks of time windows, for each dataset.
    :param dataset_ids: list of ints.
    :param mouse_colors_dict: Keys are dataset_ids.
                        Values are tuples with (names, colors) for plotting.
    :param plot_labels: The name of each of the blocks in block_comparison.
    :param fig_save_dir: If not None, save the plot to this directory.
    :param region: The name of the brain region used for this classification.
    :param do_normalize: bool. If true, then normalize each row of auc_mat (ie
                         normalize scores across tests for one training
                         set).
    :return: pvals: result of ttest_rel (multiple comparison corrected)
                    for the performance of a given trained basis on
                    the test time blocks.
    """

    plt.figure(figsize=(9, 4))
    plot_labels = ['pre', 'peri', 'post']
    for trained_on in range(3):
        ax = plt.subplot(1, 3, trained_on + 1)
        #     plt.boxplot(block_comparison[i, :, :].T)
        for d, d_id in enumerate(dataset_ids):
            plt.plot([1, 2, 3],
                     block_comparison[trained_on, :, d], 'o',
                     label=mouse_colors_dict[d_id][1],
                     color=mouse_colors_dict[d_id][0])
        plt.plot([1, 2, 3], np.median(
                 block_comparison[trained_on, :, :], axis=1), '_r',
                 markersize=10)
        if not do_normalize:
            plt.ylim([0.40, 0.85])
        plt.xlim([0.5, 3.5])

        plt.xticks([1, 2, 3], ['pre', 'peri', 'post'])
        plt.xlabel('Test timepoints')
        if trained_on == 0:
            if do_normalize:
                plt.ylabel('Normalized AUC')
            else:
                plt.ylabel('AUC')
        else:
            ax.set_yticklabels([])
        if trained_on == 2:
            plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.axhline(0.5, linestyle='--', color='k')

    # For each train dataset (i.e. pre-, peri-, or post-task time window)
    # Compare the prediction performance of predictors test on
    # pre-, peri-, or post-task data.
    # This is one group of subjects taking multiple tests (i.e.
    # male students taking chemistry and biology tests and comparing
    # performance. Need to use ttest_rel).

    pvals = np.zeros((3, 2))

    train_on_pre = block_comparison[0, :, :]
    train_on_peri = block_comparison[1, :, :]
    train_on_post = block_comparison[2, :, :]

    _, pvals[0, 0] = ttest_rel(train_on_pre[0, :], train_on_pre[1, :], )
    _, pvals[0, 1] = ttest_rel(train_on_pre[0, :], train_on_pre[2, :], )

    _, pvals[1, 0] = ttest_rel(train_on_peri[1, :], train_on_peri[0, :], )
    _, pvals[1, 1] = ttest_rel(train_on_peri[1, :], train_on_peri[2, :], )

    _, pvals[2, 0] = ttest_rel(train_on_post[2, :], train_on_post[0, :], )
    _, pvals[2, 1] = ttest_rel(train_on_post[2, :], train_on_post[1, :], )

    pvals /= 2  # Adjust for one-sided test, since you know that
    # testing on the data you train on should always have greater
    # performance that is greater than or equal to when using
    # a predictor trained on other data.

    # mult_comparison = 2  # Two comparisons for each statistical test
    # (i.e. comparing the two other blocks with
    # the one block trained on the data you are
    # testing with).
    # pvals *= mult_comparison  # Bonferonni adjustment (most stringent).
    alpha = 0.05

    # Add pvalues to the subplot titles.
    plt.subplot(1, 3, 1)
    pp = [pvals[0, 0], pvals[0, 1]]
    reject, pcorr, _, _ = mt.multipletests(pp, alpha=alpha, method='fdr_bh')
    plt.title('Trained on {}\n ttest-rel p: \n 0/1:{:.3f}, 0/2:{:.3f}'.format(
                'pre', pcorr[0], pcorr[1]))

    plt.subplot(1, 3, 2)
    pp = [pvals[1, 0], pvals[1, 1]]
    reject, pcorr, _, _ = mt.multipletests(pp, alpha=alpha, method='fdr_bh')
    plt.title('Trained on {}\n ttest-rel p: \n 1/0:{:.3f}, 1/2:{:.3f}'.format(
                'peri', pcorr[0], pcorr[1]))

    plt.subplot(1, 3, 3)
    pp = [pvals[2, 0], pvals[2, 1]]
    reject, pcorr, _, _ = mt.multipletests(pp, alpha=alpha, method='fdr_bh')
    plt.title('Trained on {}\n ttest-rel p: \n 2/0:{:.3f}, 2/1:{:.3f}'.format(
                'post', pcorr[0], pcorr[1]))

    if fig_save_dir is not None:
        save_path = os.path.join(
            fig_save_dir,
            '{}-block_comparison_group_by_TRAIN_norm{}.pdf'.format(
                                     region, int(do_normalize)))
        plt.tight_layout()
        plt.gcf().set_size_inches(w=4, h=2)  # Control size of figure in inches

        plt.savefig(save_path)

    plt.figure()

    return pvals


def plot_block_comparison_group_by_test(block_comparison,
                                        dataset_ids,
                                        mouse_colors_dict,
                                        plot_labels,
                                        fig_save_dir=None,
                                        region=None,
                                        do_normalize=False):
    """
    Plot the output of compare_time_window_blocks_predictions().
    Within each subplot, compare the performance of bases
    Trained on each block but all Tested on the same block.

    :param block_comparison: [3 x 3 x num_datasets].
                             [trained_on x tested_on x dataset]. Average
                             prediction performance between each of the
                             blocks of time windows, for each dataset.
    :param dataset_ids: list of ints.
    :param mouse_colors_dict: Keys are dataset_ids.
                    Values are tuples with (names, colors) for plotting.
    :param plot_labels: The name of each of the blocks in block_comparison.
    :param fig_save_dir: If not None, save the plot to this directory.
    :param region: The name of the brain region used for this classification.
    :param do_normalize: bool. If true, then normalize each row of auc_mat (ie
                         normalize scores across tests for one training
                         set).
    :return: pvals: result of ttest_rel (multiple comparison corrected)
                    for the performance of a given trained basis on
                    the test time blocks.
    """

    plt.figure(figsize=(9, 4))
    plot_labels = ['pre', 'peri', 'post']
    for tested_on in range(3):
        ax = plt.subplot(1, 3, tested_on + 1)
        #     plt.boxplot(block_comparison[i, :, :].T)
        for d, d_id in enumerate(dataset_ids):
            plt.plot([1, 2, 3], block_comparison[:, tested_on, d],
                     'o', label=mouse_colors_dict[d_id][1],
                     color=mouse_colors_dict[d_id][0])
        plt.plot([1, 2, 3], np.median(block_comparison[:, tested_on, :],
                 axis=1), '_r',
                 markersize=10)
        if not do_normalize:
            plt.ylim([0.40, 0.85])
        plt.xlim([0.5, 3.5])

        plt.xticks([1, 2, 3], ['pre', 'peri', 'post'])
        plt.xlabel('Train timepoints')
        if tested_on == 0:
            if do_normalize:
                plt.ylabel('Normalized AUC')
            else:
                plt.ylabel('AUC')
        else:
            ax.set_yticklabels([])
        if tested_on == 2:
            plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.axhline(0.5, linestyle='--', color='k')

    pvals = np.zeros((3, 2))

    test_on_pre = block_comparison[:, 0, :]
    test_on_peri = block_comparison[:, 1, :]
    test_on_post = block_comparison[:, 2, :]

    _, pvals[0, 0] = ttest_rel(test_on_pre[0, :], test_on_pre[1, :], )
    _, pvals[0, 1] = ttest_rel(test_on_pre[0, :], test_on_pre[2, :], )

    _, pvals[1, 0] = ttest_rel(test_on_peri[1, :], test_on_peri[0, :], )
    _, pvals[1, 1] = ttest_rel(test_on_peri[1, :], test_on_peri[2, :], )

    _, pvals[2, 0] = ttest_rel(test_on_post[2, :], test_on_post[0, :], )
    _, pvals[2, 1] = ttest_rel(test_on_post[2, :], test_on_post[1, :], )

    pvals /= 2  # Adjust for one-sided test, since you know that
    # testing on the data you train on should always have greater
    # performance that is greater than or equal to when using
    # a predictor trained on other data.

    # mult_comparison = 2  # Two comparisons for each statistical test
    # (i.e. comparing the two other blocks with
    # the one block trained on the data you are
    # testing with).
    # pvals *= mult_comparison  # Bonferonni adjustment (most stringent).
    alpha = 0.05

    # Add pvalues to the subplot titles.
    plt.subplot(1, 3, 1)
    pp = [pvals[0, 0], pvals[0, 1]]
    reject, pcorr, _, _ = mt.multipletests(pp, alpha=alpha, method='fdr_bh')
    plt.title('Tested on {}\n ttest-rel p: \n 0/1:{:.3f}, 0/2:{:.3f}'.format(
                'pre', pcorr[0], pcorr[1]))

    plt.subplot(1, 3, 2)
    pp = [pvals[1, 0], pvals[1, 1]]
    reject, pcorr, _, _ = mt.multipletests(pp, alpha=alpha, method='fdr_bh')
    plt.title('Tested on {}\n ttest-rel p: \n 1/0:{:.3f}, 1/2:{:.3f}'.format(
                'peri', pcorr[0], pcorr[1]))

    plt.subplot(1, 3, 3)
    pp = [pvals[2, 0], pvals[2, 1]]
    reject, pcorr, _, _ = mt.multipletests(pp, alpha=alpha, method='fdr_bh')
    plt.title('Tested on {}\n ttest-rel p: \n 2/0:{:.3f}, 2/1:{:.3f}'.format(
                'post', pcorr[0], pcorr[1]))

    if fig_save_dir is not None:
        save_path = os.path.join(
            fig_save_dir,
            '{}-block_comparison_group_by_TEST_norm{}.pdf'.format(
                                     region, int(do_normalize)))
        plt.tight_layout()
        plt.gcf().set_size_inches(w=4, h=2)  # Control size of figure in inches

        plt.savefig(save_path)

    plt.figure()

    return pvals


def train_all_classifiers_on_windowed_time_bins(datasets, dataset_ids,
                                                region, use_spikes):
    """Run train_classifier_on_windowed_time_bins() across all datasets."""
    all_time_basis = []
    all_training_trials = []
    all_pre_odor_trials = []
    for dataset_id in dataset_ids:
        dataset = datasets[dataset_id]
        out = train_classifier_on_windowed_time_bins(dataset,
                                                     region,
                                                     bin_size=5,
                                                     use_spikes=True)

        all_time_basis.append(out['bases'])
        all_training_trials.append(out['training_trials'])
        all_pre_odor_trials.append(out['pre_lick_trials'])
        time_bins = out['time_bins']

        # If you are curious to look at the parameters of the trained model:
        # time_basis[0].x_weights_.shape

    return all_time_basis, all_training_trials, all_pre_odor_trials, time_bins


def test_all_classifiers_on_windowed_time_bins(datasets,
                                               dataset_ids,
                                               region,
                                               use_spikes,
                                               all_time_basis,
                                               all_training_trials,
                                               all_pre_odor_trials):
    """Run test_classifier_on_windowed_time_bins() across all datasets."""

    auc_mats = []
    for d_ind in range(len(dataset_ids)):
        print('dataset {}'.format(dataset_ids[d_ind]))
        dataset = datasets[dataset_ids[d_ind]]
        time_basis = all_time_basis[d_ind]
        training_trials = all_training_trials[d_ind]
        pre_odor_trials = all_pre_odor_trials[d_ind]

        auc_mat = test_classifiers_on_windowed_time_bins(dataset,
                                                         time_basis,
                                                         training_trials,
                                                         pre_odor_trials,
                                                         region,
                                                         use_spikes=use_spikes)
        auc_mats.append(auc_mat)

    return auc_mats


def get_event_times_for_windowed_classification_plots(datasets,
                                                      dataset_ids,
                                                      time_bins):
    """
    Determine the time range, in seconds, of the a trial, centered
    on the odor-onset time.
    Additionally determine the average lick onset and lick offset time
    within a trial.
    :param datasets: dict of classifier_data object instances.
    :param dataset_ids: keys into datasets dict.
    :param time_bins: np.array indicating the onset time of each time window.
    :return: ax_lims: [start_, end_t] of a trial, in seconds,
             relative to odor onset.
             lick_timing: [lick_onsets, lick_offsets] where
                         lick_on/offsets are lists
                         of the time, in seconds, relative to odor onset,
                         of mean lick
                         on/offset for each dataset.
    """
    BD = datasets[dataset_ids[0]].behavior
    CT = datasets[dataset_ids[0]].cosmos_traces

    event_times = np.array([BD.stimulus_times[0], BD.stimulus_times[0] + 1.5])
    max_t = time_bins[-1] * 1 / CT.fps
    start_t = 0 - event_times[0]
    end_t = max_t - event_times[0]

    (mouse_lick_onsets,
     mouse_lick_offsets) = get_lick_onsets(datasets, dataset_ids,
                                           do_plot=True)
    lick_onsets = mouse_lick_onsets / CT.fps - event_times[0]
    lick_offsets = mouse_lick_offsets / CT.fps - event_times[0]

    ax_lims = [start_t, end_t]
    lick_timing = [lick_onsets, lick_offsets]

    return ax_lims, lick_timing


def plot_all_windowed_prediction_matrices(dataset_ids,
                                          auc_mats,
                                          ax_lims,
                                          lick_timing,
                                          do_normalize,
                                          fig_save_dir=None,
                                          region=None,
                                          mouse_colors_dict=None):
    """
    Run plot_windowed_prediction_matrix() for each dataset.
    :param dataset_ids: id name of each dataset.
    :param auc_mats: output from test_all_classifiers_on_windowed_time_bins()
    :param ax_lims: [start_t, end_t] in seconds. Used for making xlim and ylim.
    :param lick_timing: [lick_onsets, lick_offsets] where lick_onsets contains
                        the time of lick onset for each dataset. Used
                        for overlaying dashed lines on the plot. Should be
                        in the same units etc. as ax_lims.
    :param do_normalize: Were the AUC scores were normalized.
    :param fig_save_dir: If not None, saves plot to this directory.
    :param region: Specifies which brain region the neurons were used.
    :return: nothing
    """
    lick_onsets = lick_timing[0]
    lick_offsets = lick_timing[1]
    for d_ind, a in enumerate(auc_mats):
        plt.figure()
        plot_windowed_prediction_matrix(a, do_normalize,
                                        ax_lims,
                                        lick_onsets[d_ind],
                                        lick_offsets[d_ind])
        plt.title(mouse_colors_dict[dataset_ids[d_ind]][1])

        if fig_save_dir is not None:
            suffix = '{}-block_auc_d{}_norm{}.pdf'.format(region,
                                                          dataset_ids[d_ind],
                                                          int(do_normalize))
            save_path = os.path.join(fig_save_dir, suffix)
            print('Saved to : {}'.format(save_path))

            plt.tight_layout()
            plt.gcf().set_size_inches(w=2, h=2)
            plt.savefig(save_path)


def generate_time_windowed_classification_basis(datasets,
                                                dataset_ids,
                                                region,
                                                use_spikes):

    """
    Train multiple Partial Least Squares bases that discriminate active spout,
    trained based on successive time windows of neural data.
    Then, test the performance of each PLS basis on neural data from
    all of the other time windows.
    :param datasets: dict of classifier_data objects.
    :param dataset_ids: keys into datasets dict.
    :param region: 'all', 'MO', 'VIS', PTLp', 'SSp', 'RSP'
    :param use_spikes: bool. Which neural data to use.
                       If false use smoothed by not deconvolved spikes.
    :return: auc_mats: List of matrices, one for each dataset, containing
                      the prediction performance (as measured by Area Under
                      the ROC Curve) for the basis trained on each time window,
                      tested on all of the other time windows.
             all_time_basis: List containing, for each dataset,
                            a dict with sklearn PLS objects containing the
                            bases trained on each time window.
             time_bins: array indicating the start frame of each time window,
                        corresponding to the bases in all_time_basis.
    """

    print('Training classifiers.')
    (all_time_basis,
     all_training_trials,
     all_pre_odor_trials,
     time_bins) = train_all_classifiers_on_windowed_time_bins(datasets,
                                                              dataset_ids,
                                                              region,
                                                              use_spikes)

    # Test windowed classifiers.
    print('Testing classifiers.')
    auc_mats = test_all_classifiers_on_windowed_time_bins(datasets,
                                                          dataset_ids,
                                                          region,
                                                          use_spikes,
                                                          all_time_basis,
                                                          all_training_trials,
                                                          all_pre_odor_trials)

    return auc_mats, all_time_basis, time_bins


def summarize_time_windowed_classification_basis(
    datasets, dataset_ids, auc_mats, time_bins, all_time_basis,
    region, do_normalize, mouse_colors_dict, fig_save_dir=None,
        do_plot_cosine_distance=False):

    """
    Plots the results of generate_time_windowed_classification_basis().

    :param datasets: dict of classifier_data objects.
    :param dataset_ids: keys into datasets dict.
    :param auc_mats: Output from generate_time_windowed_classification_basis()
    :param time_bins: Output from generate_time_windowed_classification_basis()
    :param all_time_basis:
        Output from generate_time_windowed_classification_basis().
    :param region: 'all', 'MO', 'VIS', PTLp', 'SSp', 'RSP'. Determines filename
    :param do_normalize: bool. If true, then normalize each row of each auc_mat
                         (i.e.
                         normalize scores across tests for one training
                         set).
    :param mouse_colors_dict: keys correspond to dataset_ids.
                         Values are tuples with (names, colors) for plotting.
    :param fig_save_dir: If not None, saves plots to this directory.
    :param do_plot_cosine_distance: Plot the dot product between the bases
                                    derived from each time window.
    :return: pvals: Results of ttest_rel comparing performance of bases
                    trained on three blocks of time windows within a trial.
    """

    # Get times of lick onset and offset, to overlay on plot.
    (ax_lims,
     lick_timing) = get_event_times_for_windowed_classification_plots(
         datasets, dataset_ids, time_bins)

    print('Plotting prediction matrices.')
    plot_all_windowed_prediction_matrices(dataset_ids,
                                          auc_mats,
                                          ax_lims,
                                          lick_timing,
                                          do_normalize,
                                          fig_save_dir=fig_save_dir,
                                          region=region,
                                          mouse_colors_dict=mouse_colors_dict)

    plot_mean_windowed_prediction_matrix(auc_mats,
                                         ax_lims,
                                         lick_timing,
                                         do_normalize,
                                         fig_save_dir=fig_save_dir,
                                         region=region)

    # Quantify how training on each block performs when
    # testing on the other blocks.
    pre_bins = np.arange(0, 15)  # Pre-task
    peri_bins = np.arange(15, 30)  # Peri-task
    post_bins = np.arange(35, 40)  # Post-task

    block_comparison = compare_time_window_blocks_predictions(auc_mats,
                                                              pre_bins,
                                                              peri_bins,
                                                              post_bins,
                                                              do_normalize)
    print('block_comparison')
    print(np.mean(block_comparison, axis=2))
    labels = ['pre', 'peri', 'post']
    pvals = plot_block_comparison_group_by_test(block_comparison,
                                                dataset_ids,
                                                mouse_colors_dict,
                                                plot_labels=labels,
                                                fig_save_dir=fig_save_dir,
                                                region=region,
                                                do_normalize=do_normalize)
    print('group by test, ttest-ind:')
    print(pvals)

    pvals = plot_block_comparison_group_by_train(block_comparison,
                                                 dataset_ids,
                                                 mouse_colors_dict,
                                                 plot_labels=labels,
                                                 fig_save_dir=fig_save_dir,
                                                 region=region,
                                                 do_normalize=do_normalize)
    print('group by train, ttest-rel:')
    print(pvals)

    pvals = plot_block_comparison_compare_vs_chance(block_comparison,
                                                    dataset_ids,
                                                    mouse_colors_dict,
                                                    plot_labels=labels,
                                                    fig_save_dir=fig_save_dir,
                                                    region=region,
                                                    do_normalize=do_normalize)

    print('vs chance, ttest-1samp:')
    print(pvals)

    if do_plot_cosine_distance:
        for time_basis in all_time_basis:
            compute_bases_cosine_distance(time_basis, comps=3)

    return pvals
