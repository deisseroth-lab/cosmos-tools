import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.morphology
import seaborn as sns
import pandas as pd

import cosmos.traces.trace_analysis_utils as utils
import cosmos.behavior.bpod_io as bio


class BpodDataset:

    """
    Loads, parses, plots behavioral
    data collected using bpod.
    """

    def __init__(self, data_fullpath, fig_save_dir, ntrials=None):
        """
        :param data_fullpath: full path to the .mat
                              file containing bpod
                              behavioral data.
                              i.e.
                              '~/Dropbox/cosmos_data/behavior/
                              m72/COSMOSShapeMultiGNG/Session Data/
                              m72_COSMOSShapeMultiGNG.mat'
        :param fig_save_dir: path to specific directory where
                             you want to save all plots.
                             (This includes the mouse name, and date, etc..)
                             i.e. '~/Dropbox/cosmos/trace_analysis/20180212/
                             m72_COSMOSShapeMultiGNG_1'
        :param ntrials: default None. If not None, then trims
                        all Bpod data to be of length ntrials.
                        This is for when camera acquisition
                        stopped before bpod acquisition stopped.
        """

        self.data_fullpath = data_fullpath
        self.fig_save_dir = fig_save_dir
        self.suffix = '.pdf'
        self.protocol = self._get_protocol(data_fullpath)
        self.session_name = data_fullpath.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        self.B = bio.load_bpod_data(data_fullpath)

        # Bpod only counts complete so LED frames (i.e. start trials)
        # will have one more trial, assuming we stop Bpod acquisition mid-trial
        # self.ntrials = self.B['nTrials'] + 1
        self.ntrials = self.B['nTrials']
        if ntrials is not None:
            self.ntrials = ntrials
            print('While loading bpod, enforcing that ntrials is:' + str(ntrials))

        self.spout_positions = self.B['SpoutPositions'][:self.ntrials]
        self.trial_types = self.B['TrialTypes'][:self.ntrials]  # i.e. which odor.
        self.trial_start_times = (self.B['TrialStartTimestamp'][:self.ntrials] -
                                  self.B['TrialStartTimestamp'][0])

        # Get stim event types if they exist
        self.stim_types = None
        self.stim_interval = None
        if 'StimTypes' in self.B:
            self.stim_types = self.B['StimTypes'][:self.ntrials]
            # assert len(self.stim_types) == self.ntrials, (
            #     'Invalid number of opto stim events!')
        if 'StimInterval' in self.B:
            self.stim_interval = self.B['StimInterval']
            print('stim_interval ', self.stim_interval)


        # Lick rate for each spout smoothed to imaging speed
        self.spout_lick_rates = dict()

        # Dict with array of lick times for each trial
        self.spout_lick_times = dict()

        # Binary matrix [trials x time in ms] indicating when lick occured
        self.spout_lick_mat = dict()

        self.nspouts = len(np.unique(self.spout_positions))
        self.port_names = ['Port5In', 'Port6In', 'Port7In', 'Port8In']
        self.port_colors = ['white', 'red', 'green', 'yellow']
        if self.nspouts == 4:
            self.port_theta = np.array([15, 65, 115, 165])
        elif self.nspouts == 3:
            self.port_theta = np.array([15, 90, 165])
        elif self.nspouts == 1:
            self.port_theta = np.array([90])

        for ind, port in enumerate(self.port_names):
            lick_rates, lick_t, lick_times = bio.get_lick_rates(self.B,
                                                                port=port,
                                                                bin_size=0.034)
            nt = self.ntrials
            lick_times_mat, lick_mat_t = bio.lick_times_to_matrix(lick_times,
                                                                  ntrials=nt,
                                                                  hz=1000) #hz=1.0/0.034
            self.spout_lick_rates[ind] = lick_rates[:self.ntrials,:]
            self.spout_lick_times[ind] = lick_times
            self.spout_lick_mat[ind] = lick_times_mat[:self.ntrials,:]
        self.lick_t = lick_t
        self.lick_mat_t = lick_mat_t

        self.states = bio.list_states(
            self.B)  # List of all possible state times you can query.
        self.reward_times = bio.get_state_times(
            self.B, 'Reward', ind=0)  # nan when no reward.
        self.reward_times = self.reward_times[:self.ntrials]
        self.small_reward_times = bio.get_state_times(
            self.B, 'SmallReward', ind=0)
        self.small_reward_times = self.small_reward_times[:self.ntrials]
        self.stimulus_times = bio.get_state_times(
            self.B, 'Stimulus', ind=0)  # i.e. the odor.
        self.stimulus_times = self.stimulus_times[:self.ntrials]
        self.signal_period_times = bio.get_state_times(
            self.B, 'SignalPeriod', ind=0)  # i.e. the odor.
        self.signal_period_times = self.signal_period_times[:self.ntrials]
        self.trigger_times = bio.get_state_times(
            self.B, 'TriggerStart', ind=0)  # i.e. trial start/led on
        self.trigger_times = self.trigger_times[:self.ntrials]
        self.punish_times = bio.get_state_times(self.B, 'Punish', ind=0)
        self.punish_times = self.punish_times[:self.ntrials]
        self.iti_start = bio.get_state_times(self.B, 'ITI', ind=0)
        self.iti_start = self.iti_start[:self.ntrials]
        self.iti_end = bio.get_state_times(self.B, 'ITI', ind=1)
        self.iti_end = self.iti_end[:self.ntrials]
        self.iti_lengths = self.iti_end - self.iti_start

        self.go_trials = self._get_go_trials(self.protocol, self.trial_types)
        self.go_trials = self.go_trials[:self.ntrials]
        self.success = self._get_trial_success(self.protocol, self.trial_types)
        self.success = self.success[:self.ntrials]

        self.trial_inds = np.arange(self.ntrials)
        self.trial_inds = self.trial_inds[:self.ntrials]
        self.ind_within_block = self._get_ind_within_block()
        self.ind_within_block = self.ind_within_block[:self.ntrials]

    def get_on_target_licks(self, trial, frame_range=None, target_spout=None):
        """
        For a given trial, return the fraction of licks to the active spout.
        """
        if target_spout is not None:
            active_spout = target_spout
        else:
            active_spout = self.spout_positions[trial]
        all_licks = 0
        active_licks = 0
        for spout in range(4):
            if frame_range:
                current_licks = np.sum(
                    self.spout_lick_rates[spout][trial, frame_range[0]:
                                                 frame_range[1]])
            else:
                current_licks = np.sum(self.spout_lick_rates[spout][trial, :])
            all_licks += current_licks

            # lick_rates has spouts 0, 2, 3 active
            # spout_positions has 1, 3, 4 active
            if spout == active_spout-1:
                active_licks = current_licks
        if all_licks > 0:
            return active_licks / all_licks
        else:
            return 0

    def get_clean_trials(self, min_selectivity=1,
                         exclude_explore=True, verbose=True):
        """
        Return a list of all trials had a lick selectivity of at least
        min_selectivity, also by default, exclude nogo and "explore" trials.

        Exclude the last (and likely incomplete) trial from this list.
        """

        non_explore_go_idx = 3
        on_target_licks = np.array([self.get_on_target_licks(trial)
                                   for trial in range(self.ntrials - 1)])

        clean_trials = np.where(on_target_licks >= min_selectivity)[0]
        if exclude_explore:
            non_explore_go_trials = np.where(
                self.trial_types == non_explore_go_idx)[0]
            clean_trials = np.intersect1d(clean_trials, non_explore_go_trials)
        if verbose:
            print('Detected', len(clean_trials), '/', (self.ntrials - 1),
                  'trials with lick selectivity >= ', min_selectivity)
        return clean_trials

    def get_no_preodor_lick_trials(self, max_licks=0, do_plot=False):
        """
        Return a list of all trials where the mouse did
        not lick any of the spouts during the preodor
        period.

        :param max_licks: The maximum allowed number of licks
                          during the preodor period.
        :param do_plot: Set to true to make a plot to verify
                        this function is working correctly.
        :return: List of trials.
        """

        odor_time = self.stimulus_times[0]
        pre_odor_lick_frames = np.where(self.lick_mat_t<odor_time)[0]

        spout_pre_odor_licks = dict()
        for spout in self.spout_lick_mat.keys():
            pre_odor_licks = self.spout_lick_mat[spout][:, pre_odor_lick_frames]
            spout_pre_odor_licks[spout] = np.sum(pre_odor_licks, axis=1)

        all_pre_odor_licks = np.vstack([x for x in spout_pre_odor_licks.values()])
        max_pre_odor_licks = np.max(all_pre_odor_licks, axis=0)

        no_preodor_lick_trials = max_pre_odor_licks <= max_licks

        if do_plot:
            self.plot_lick_times(fig=plt.figure(figsize=(20, 20)),
                            trials_subset=np.where(no_preodor_lick_trials)[0])
            plt.xlim([-3, 6])
            plt.suptitle('No preodor licks')

            self.plot_lick_times(fig=plt.figure(figsize=(20, 20)),
                            trials_subset=np.where(~no_preodor_lick_trials)[0])
            plt.xlim([-3, 6])
            plt.suptitle('Just with preodor licks')

        return np.where(no_preodor_lick_trials)[0]



    def plot_go_nogo_licks(self):
        """ Make a raster of go vs nogo licks. """
        trial_types = self.trial_types
        go_trials = np.where(self.go_trials)[0]
        nogo_trials = np.where(~self.go_trials)[0]

        start = self.stimulus_times[0]
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        x = 0
        y = 0
        for trial_ind in self.trial_inds:
            if trial_ind in go_trials:
                x = x + 1
                ax = ax1
                z = x
            else:
                y = y + 1
                ax = ax2
                z = y
            for spout_ind in range(len(self.spout_lick_times.keys())):
                spout = self.spout_lick_times[spout_ind]
                if trial_ind in spout.keys():
                    trial = spout[trial_ind] - start
                    ax.plot(trial, z * np.ones_like(trial),
                            '.', color='k', markersize=5)
        for ax, z, tt in zip([ax1, ax2], [x, y],
                             ['go trials', 'no-go trials']):
            ax.set_xlim([-0.5, 4])
            for lp in [0, 1, 1.5]:
                ax.plot([lp, lp], [-1, z], color=[1, 0, 0, .5])
            ax.set_ylim([0, z])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(tt)
            if z == y:
                ax.set_xlabel('time relative to stimulus (s)')
        plt.tight_layout()

    def plot_success(self):
        """
        Plot the smoothed success rate vs. trial number.
        """
        plt.figure()
        window = 20
        plt.plot(utils.moving_avg(self.success.astype(float),
                                  do_median=False, window=window))
        plt.xlim([window, self.ntrials])
        plt.xlabel('Trial #')
        plt.ylabel('Success rate')
        plt.title(self.session_name)
        save_path = os.path.join(self.fig_save_dir, 'success_rate'+self.suffix)
        print('Saving to: ', save_path)
        plt.savefig(save_path)

    def plot_lick_times(self, fig=None, alt_colors=False, lw=1, markersize=1,
                        trials_subset=None, underlay_stim_trials=False,
                        underlay_nogo=False):
        """
        Show the lick times for each trial,
        overlaid for each spout.

        :param trials_subset: Provide an array of trial numbers
                              if you only want to plot licks from
                              that subset of trials.
        """
        if alt_colors:
            colors = ['orange', 'w', 'c', 'r']
        else:
            colors = self.port_colors
        lick_times = self.spout_lick_times
        if fig is None:
            fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111)
        if alt_colors:
            ax.set_facecolor('white')
        else:
            ax.set_facecolor('black')

        if underlay_stim_trials:
            if self.stim_types is not None:
                for i in np.where(self.stim_types > 0)[0]:
                    plt.plot([-10, 10], [i, i], color=(0.9, 0.9,0.9), alpha=1)

        start = self.stimulus_times[0]
        for spout_ind in range(len(self.spout_lick_times.keys())):
            spout = self.spout_lick_times[spout_ind]
            for trial_ind in spout.keys():
                if trials_subset is None or trial_ind in trials_subset:
                    trial = spout[trial_ind] - start
                    trial = trial[trial < 6] ## For visualization, cut off at 6 seconds.
                    trial = trial[trial > -1] ## For visualization.
                    plt.plot(trial, trial_ind * np.ones_like(trial),
                             '.', color=colors[spout_ind],
                             markersize=markersize)

        # Stimulus onset
        plt.axvline(x=self.stimulus_times[
                    0] - start, color='k', linestyle='--', lw=lw)

        # Stimulus offset
        plt.axvline(x=self.stimulus_times[0] + 1 - start,
                    color='k', linestyle='--',lw=lw)

        # Approximate reward onset
        plt.axvline(x=self.stimulus_times[0] + 1.5 - start,
                    color='k', linestyle='--', lw=lw)
        plt.xlim([-1, 6.4])
        plt.xlabel('Time relative to odor (s)')
        plt.ylabel('Trial #')
        plt.title(self.session_name)

        ### Plot trial type
        for trial, active_spout in enumerate(self.spout_positions):
            plt.plot(6.4, trial, '_', ms=4, color=colors[active_spout - 1])
            if self.trial_types[trial] == 4:
                if underlay_nogo:
                    plt.axhline(trial, color=[0, 1, 0, 0.4], linewidth=0.5)
                    plt.plot(6.4, trial, '_', ms=4, color='g')
                else:
                    plt.plot(6.4, trial, '_', ms=4, color='g')

        save_path = os.path.join(self.fig_save_dir, 'licks'+self.suffix)
        print('Saving to: ', save_path)
        plt.savefig(save_path)

    def plot_mean_lick_rates(self, which_trials, ax=None):
        """
        Overlay the lick rate to each spout for a provided
        set of trials.
        :param which_trials: np.array. Contains trial indices.
        :param ax: axes to plot to.
        """
        if ax is None:
            ax=plt.figure().gca()
        plt.sca(ax)

        for spout in self.spout_lick_rates.keys():
            lick_rates = self.spout_lick_rates[spout]
            correct_spout = self.spout_positions == spout + 1
            spout_trials = np.logical_and(which_trials, correct_spout)
            mean_lick = np.mean(lick_rates[which_trials, :], axis=0)
            plt.plot(self.lick_t, mean_lick, label='spout #'+str(spout+1))
            plt.axvline(x=self.stimulus_times[0], color='r', linestyle='--')
            plt.ylabel('Lick rate')
            plt.xlabel('Time [s]')
        plt.legend()

    def plot_lick_matrix(self, fig=None, dilation_width=10):
        """
        Plots self.spout_lick_mat, which contains a binary
        matrix for each spout, indicating when a lick occurred.
        Each spout is represented by a different color.
        Should look similar to, but potentially not exactly the same
        as, the output of plot_lick_times(). Differences can be caused
        by the order in which things are plotted, and the fact that
        colors may add together to become white within a timebin.

        :param dilation_width: Across how many timebins to spread each
                               lick (so that they are visible in the plot).
        """
        if fig is None:
            fig = plt.figure(figsize=(7, 7))

        ntrials = np.shape(self.spout_lick_mat[0])[0]
        A = np.zeros((ntrials, np.shape(self.spout_lick_mat[0])[1], 3))
        for i in range(4):
            a = self.spout_lick_mat[i]
            o = np.ones((1, dilation_width))
            a = skimage.morphology.binary_dilation(a, o).astype(a.dtype)
            if i == 0:
                A[:, :, 0] = A[:, :, 0] + a
                A[:, :, 1] = A[:, :, 1] + a
                A[:, :, 2] = A[:, :, 2] + a
            if i == 1:
                A[:, :, 0] = A[:, :, 0] + a
            if i == 2:
                A[:, :, 1] = A[:, :, 1] + a
            if i == 3:
                A[:, :, 0] = A[:, :, 0] + a
                A[:, :, 1] = A[:, :, 1] + a
        plt.imshow(A[:, :], aspect='auto', origin='lower',
                   extent=[self.lick_mat_t[0],
                   self.lick_mat_t[-1], 0, ntrials])

        start = 0
        # Stimulus onset
        plt.axvline(x=self.stimulus_times[0] - start,
                    color='r', linestyle='--')

        # Stimulus offset
        plt.axvline(x=self.stimulus_times[0] + 1 - start,
                    color='r', linestyle='--')

        # Approximate reward onset
        plt.axvline(x=self.stimulus_times[0] + 1.5 - start,
                    color='r', linestyle='--')

    def plot_spout_selectivity(self, min_trial=10, do_save=True,
                               trial_subset=None, alt_colors=False,
                               save_prefix=None, subplot_strs=None,
                               title_strs=None):
        """
        Make polar plot of spout selectivity
        before and after reward delivery.

        :param min_trial: Ignore initial trials in the session
                          up to this trial,
                          while the mouse gets warmed up.
        :param do_save: bool. Save out the plots.
        :param trial_subset: an array of the same length as self.spout_positions
                             that indicates whether a trial should or
                             should not be included in plotting the selectivity.
        :returns pre_out, post_out: data to plot polar plots for pre-reward
                                    and post-reward periods.
        """
        save_name = None
        spout_trials = dict()
        trial_ind = np.arange(len(self.spout_positions))
        for i in np.unique(self.spout_positions):
            if trial_subset is None:
                spout_trials[i] = np.where(np.logical_and.reduce((
                                           self.spout_positions == i,
                                           trial_ind > min_trial
                                           )))[0]
            else:
                spout_trials[i] = np.where(np.logical_and.reduce((
                                           self.spout_positions == i,
                                           trial_ind > min_trial,
                                           trial_subset
                                           )))[0]
        # Pre-reward
        if subplot_strs is None:
            plt.figure()
            subplot_strs = ['121', '122']

        if title_strs is None:
            title_strs = ['pre-reward', 'post-reward']

        titlestr = title_strs[0]
        if do_save:
            save_name = titlestr
            if save_prefix is not None:
                save_name = save_prefix + save_name
        t_range = [self.stimulus_times[0], self.stimulus_times[0] + 1.45]
        pre_out = self.polar_lick_plot(spout_trials=spout_trials,
                                       t_range=t_range, plot_trace=False,
                                       save_name=save_name,
                                       titlestr=titlestr,
                                       subplot_str=subplot_strs[0],
                                       alt_colors=alt_colors)

        # Post-reward
        titlestr = title_strs[1]
        if do_save:
            save_name = titlestr
            if save_prefix is not None:
                save_name = save_prefix + save_name
        t_range = [self.stimulus_times[0] + 1.5, self.stimulus_times[0] + 3.5]
        post_out = self.polar_lick_plot(spout_trials=spout_trials,
                                        t_range=t_range, plot_trace=False,
                                        save_name=save_name,
                                        titlestr=titlestr,
                                        subplot_str=subplot_strs[1],
                                        alt_colors=alt_colors,
                                        )

        return pre_out, post_out

    def polar_lick_plot(self, spout_trials=None, t_range=None,
                        plot_trace=False,
                        titlestr=None,
                        save_name=None,
                        subplot_str='111', alt_colors=False):
        """
        Generates a polar plot displaying the distribution
        of licks to each spout, for each target spout.
        Uses self.spout_lick_mat, which is a binary matrix
        with a 1 at each timebin where a lick occurred.

        :param spout_trials: a dict of arrays containing indices of trials
                       to be included for each target spout position.
                       This array is assumed to only have entries corresponding
                       to the spouts that are in use (i.e. for 3 active spouts,
                       this may be less than the total 4 spouts that
                       are being recorded.) Note that spout positions
                       are indexed starting at 1.
        :param t_range: an array containing the start and end time
                      in seconds to use when generating the plot.
                      t=0 is the beginning of the trial (not odor delivery).
        :param plot_trace: In addition to the polar plot, plot the actual
                      histogram of licks across the specified time period
                      for each licked spout for each target spout
        :param subplot_str: Can specify a subplot within an existing
                              figure for this plot. Note: we have to specify
                              the axes this way because of a peculiarity
                              in setting up the polar plot which does not
                              allow us to simply pass in an axis handle.

        :returns out: data adequate for reproducing the polar plot generated
                      by this function.

        """

        if alt_colors:
            port_colors = ['orange', 'w', 'c', 'r']
        else:
            port_colors = self.port_colors

        colors = [port_colors[i - 1] for i in spout_trials.keys()]

        # Distribution of licks to each spout
        spout_dist = np.zeros((self.nspouts, self.nspouts))
        if plot_trace:
            f = plt.figure(figsize=(30, 10))

        # Compute lick distributions for given trials and time range
        for target_ind, target_spout in enumerate(spout_trials):
            if plot_trace:
                ax = f.add_subplot(1, int(self.nspouts), int(target_spout + 1))
                ax.set_title(target_spout + 1)
            for licked_ind, licked_spout in enumerate(spout_trials):
                lick_mat = self.spout_lick_mat[licked_spout - 1]

                if spout_trials is None:
                    trials = np.arange(lick_mat.shape[0])
                else:
                    trials = spout_trials[target_spout]

                if t_range is None:
                    t_range = [0, np.amax(self.lick_mat_t)]

                inds = np.where((self.lick_mat_t > t_range[0]) &
                                (self.lick_mat_t < t_range[1]))[0]
                lick_mat = lick_mat[trials, :][:, inds]
                spout_dist[target_ind, licked_ind] = np.sum(lick_mat)

                if plot_trace:
                    import scipy.signal
                    plt.plot(scipy.signal.savgol_filter(np.mean(lick_mat,
                             axis=0), 1001, 3),
                             color=colors[licked_spout].replace('white',
                             'black'))

        theta = np.deg2rad(self.port_theta)

        # Actually make the polar plots now.
        ax = plt.subplot(subplot_str, projection='polar',
                         facecolor=[.9, .9, .9])
        ax.set_rmax(1)
        ax.set_rticks([])
        ax.set_xticks(np.mod(theta, 2 * np.pi))
        ax.set_xticklabels(np.arange(self.nspouts) + 1)

        for target_spout in range(spout_dist.shape[0]):
            d = spout_dist[target_spout, :]
            d = d / np.max(d)
            spout_dist[target_spout,:] = d
            ax.plot(np.append(theta, theta[0]), np.append(d, d[0]),
                    color=colors[target_spout].replace('white', 'black'),
                    linewidth=5)

        if titlestr is not None:
            plt.title(titlestr)

        if save_name is not None:
            save_path = os.path.join(self.fig_save_dir,
                                     'polar_' + save_name + self.suffix)
            print('Saving to: ', save_path)
            plt.savefig(save_path)

        out = {}
        out['spout_dist'] = spout_dist
        out['theta'] = theta
        return out

    def plot_stim_licks(self, off_code=99, stim_interval=[0, 1],
                        fig_save_dir=None, ax=None, titles=None,
                        max_trial=None, max_subset_trial=None,
                       include_explore_trials=True):
        """
        Make a raster of licks during different stim condition trials.

        """
        if type(self.stim_types) is not np.ndarray:
            print('quitting')
            return

        # We're ignoring go v. nogo in this context right now
        stim_type_trials = self.stim_types
        num_types = len(np.unique(stim_type_trials))

        start = self.stimulus_times[0]
        if ax is None:
            ax = [plt.subplot(num_types, 1, x + 1) for x in range(num_types)]

        ct = np.zeros(len(ax))

        # for each stim type
        cc = ['orange', 'w', 'c', 'r']
        for sdx, stim_type in enumerate(np.unique(stim_type_trials)):
            cax = ax[sdx]
            stim_trials = np.where(self.stim_types == stim_type)[0]
            subset_ind = 0
            for trial_ind in stim_trials:
                if max_trial is None or trial_ind < max_trial:
                    if max_subset_trial is None or subset_ind < max_subset_trial:
                        if self.trial_types[trial_ind] != 4: ### Exclude nogo trials
                            if include_explore_trials or self.trial_types[trial_ind] != 2: ### Only include non-explore GO trials
                                ct[sdx] += 1
                                subset_ind += 1
                                for spout_ind in range(len(self.spout_lick_times.keys())):
                                    spout = self.spout_lick_times[spout_ind]
                                    if trial_ind in spout.keys():
                                        # Plot licks
                                        trial = spout[trial_ind] - start

                                        # ### CUT OFF ALL BUT MAX TIMEPOINT
                                        max_timepoint = 4
                                        if isinstance(trial, np.ndarray):
                                            trial = trial[np.where(trial < max_timepoint)]

                                        cax.plot(trial, ct[sdx] * np.ones_like(trial),
                                                 '.', color=cc[spout_ind], markersize=2)

                                active_spout = self.spout_positions[trial_ind]
                                cax.plot(4.7, ct[sdx], '.', ms=4, color=cc[active_spout - 1])

                                if self.success[trial_ind]:
                                    cax.plot(4.3, ct[sdx], '_', ms=6, color='g')
                                else:
                                    cax.plot(4.3, ct[sdx], '_', ms=6, color='m')

        for sdx, cax in enumerate(ax):
            if titles is not None:
                cax.set_title(titles[sdx])
            cax.set_xlim([-2, 5])
            if max_subset_trial is not None:
                cax.set_ylim([0, max_subset_trial+5])
            cax.spines['top'].set_visible(False)
            cax.spines['right'].set_visible(False)
            if np.unique(stim_type_trials)[sdx] != 99:
                cax.plot(stim_interval, [ct[sdx]*1.05, ct[sdx]*1.05],
                         color=[0, 0, 1, 0.5],
                         linewidth=3)
            for vert in [0, 1, 1.5]:
                cax.axvline(vert, color=[0, 0, 0, 1], linestyle='--')
            if sdx == len(ax) - 1:
                cax.set_xlabel('time relative to stimulus (s)')
            cax.set_ylabel(
                'stim ' + str(np.unique(stim_type_trials)[sdx]) + ' trials')
        plt.tight_layout()

        if fig_save_dir is not None:
            save_path = os.path.join(fig_save_dir, 'stim_licks'+self.suffix)
            print('Saving to: ', save_path)
            plt.savefig(save_path)

    def summarize_licks_during_stim(self, intervals, do_plot=True,
                                    fig_save_dir=None):
        """
        Compare the mean number of licks during different intervals
        in a trial, across stim vs. no-stim trials.
        :param self:
        :param intervals: dict. keys: Name of interval.
                         values: [t_start, t_end], with t=0 being
                         the beginning of the trial.
        :return: a pandas dataframe containing mean lick rate, broken down
                 by trial interval, trial type, spout, etc.
        """

        scores = pd.DataFrame()
        for name, t_range in intervals.items():
            inds = np.where((self.lick_mat_t > t_range[0]) &
                            (self.lick_mat_t < t_range[1]))[0]
            trials = np.arange(self.ntrials)
            for spout in self.spout_lick_mat.keys():
                for stim_type in np.unique(self.stim_types):
                    stim_trials = np.where(self.stim_types == stim_type)[0]
                    lick_mat = self.spout_lick_mat[spout]
                    lick_mat = lick_mat[stim_trials, :][:, inds]
                    if np.sum(lick_mat) > 2:
                        d = {'Mean licks': np.mean(np.sum(lick_mat, 1)),
                             'Interval': name, 'Num licks': np.sum(lick_mat),
                             'Spout': spout, 'Stim Pattern': stim_type}
                        scores = scores.append(d, ignore_index=True)

        if do_plot:
            plt.figure(figsize=(15, 5))
            for ind, spout in enumerate([0, 2, 3]):
                plt.subplot(1, 3, ind + 1)
                sns.barplot(data=scores.loc[scores['Spout'] == spout],
                            x='Interval',
                            y='Mean licks', hue='Stim Pattern')
                plt.ylabel('Mean number of licks')
                plt.title('Spout ' + str(spout))
            if fig_save_dir is not None:
                save_path = os.path.join(self.fig_save_dir,
                                         'stim_lick_rates_per_spout'+self.suffix)
                print('Saving to: ', save_path)
                plt.savefig(save_path)

            plt.figure()
            sns.barplot(data=scores, x='Interval',
                        y='Mean licks', hue='Stim Pattern')
            plt.ylabel('Mean number of licks')
            plt.title('Mean across spouts')
            if fig_save_dir is not None:
                save_path = os.path.join(self.fig_save_dir,
                                         'stim_lick_rates_all_spouts'+self.suffix)
                print('Saving to: ', save_path)
                plt.savefig(save_path)

        return scores

    # Private functions
    def _get_protocol(self, data_fullpath):
        """
        Searches for allowed protocols (i.e. that this
        class has been designed to handle) in the filename,
        otherwise returns an error.
        """
        if 'COSMOSShapeMultiGNG' in data_fullpath:
            protocol = 'COSMOSShapeMultiGNG'
        elif 'COSMOSTrainMultiGNG' in data_fullpath:
            protocol = 'COSMOSTrainMultiGNG'
        elif 'COSMOSTrainMultiBlockGNG' in data_fullpath:
            protocol = 'COSMOSTrainMultiBlockGNG'
        else:
            raise ValueError('The protocol in ', data_fullpath,
                             ' has not yet been implemented for '
                             ' the BpodDataset class.')

        return protocol

    def _get_go_trials(self, protocol, trial_types, invert=False):
        """
        Returns a list indicating whether
        the mouse is supposed to lick on each trial,
        according to the appropriate protocol.
        :param protocol: str. Name of bpod protocol. i.e. self.protocol
        :param trial_types: ndarray. Int for each trial indicating trial type.
        :param invert: bool. If true, return nogo trials.
        """
        if protocol == 'COSMOSShapeMultiGNG':
            go_trials = np.logical_or(
                trial_types == 3,
                trial_types == 4
            )
        elif protocol == 'COSMOSTrainMultiGNG':
            go_trials = trial_types == 3

        elif protocol == 'COSMOSTrainMultiBlockGNG':
            go_trials = np.logical_or.reduce((trial_types == 3,
                                              trial_types == 2))
        else:
            raise ValueError('The protocol ', protocol,
                             ' has not yet been implemented '
                             ' for BpodDataset.get_trial_success.')

        if invert:
            go_trials = ~go_trials

        return go_trials
        # return go_trials.astype(int)

    def _get_trial_success(self, protocol, trial_types):
        """
        Returns a list indicating whether
        a trial was successful or not,
        according to the appropriate protocol.
        """

        if protocol == 'COSMOSShapeMultiGNG':
            success = np.logical_or(
                np.logical_and(trial_types == 3, ~np.isnan(self.reward_times)),
                np.logical_and(trial_types == 4, ~np.isnan(self.reward_times))
            )

        elif protocol == 'COSMOSTrainMultiGNG':
            success = np.logical_or(
                np.logical_and(trial_types == 3, ~np.isnan(self.reward_times)),
                np.logical_and(trial_types == 4, np.isnan(self.punish_times))
            )
        elif protocol == 'COSMOSTrainMultiBlockGNG':
            # sm = self.small_reward_times
            # success = np.logical_or.reduce((
            #     np.logical_and(trial_types == 3, ~np.isnan(sm)),
            #     np.logical_and(trial_types == 3, ~np.isnan(self.reward_times)),
            #     np.logical_and(trial_types == 2, ~np.isnan(self.reward_times)),
            #     np.logical_and(trial_types == 2, ~np.isnan(sm)),
            #     np.logical_and(trial_types == 4, np.isnan(self.punish_times))
            # ))
            success = np.logical_or.reduce((
                ~np.isnan(self.reward_times),
                np.logical_and(trial_types == 4, np.isnan(self.punish_times))
            ))
        else:
            raise ValueError('The protocol ', protocol,
                             ' has not yet been implemented '
                             ' for BpodDataset.get_trial_success.')

        return success.astype(int)

    def _get_ind_within_block(self):
        """
        For protocols where the target spout positions changes
        in blocks, return an array which, for each trial,
        indicates what number trial it is within a block.


        :return: ind_within_block: array of length total # of trials.
        """
        ind_within_block = np.zeros(self.spout_positions.shape)
        iter = 0
        for i in range(1, len(ind_within_block)):
            if self.spout_positions[i] == self.spout_positions[i - 1]:
                iter = iter + 1
            else:
                iter = 0
            ind_within_block[i] = iter

        return ind_within_block.astype(int)