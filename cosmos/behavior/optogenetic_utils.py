import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import scipy.signal
import scipy.stats
import statsmodels.stats.multitest as mt


def get_included_trials(bd, min_trial=15, max_trial=190):
    included_trials = np.ones(bd.stim_types.shape).astype(bool)
    included_trials[:min_trial] = 0
    included_trials[max_trial:] = 0

    return included_trials


def plot_mean_across_mice_licks_by_stim(licks, event_times=None,
                                        t0=2.2, fps=1/.034):
    plt.figure(figsize=(15, 2))
    colors = ['orange', 'w', 'c', 'r']

    iter = 1
    for stim_type in licks.keys():
        for active_spout in licks[stim_type].keys():
            plt.subplot(1, 6, iter)
            mean_across_mice = np.mean(licks[stim_type][active_spout],
                                       axis=2)
            sem_across_mice = scipy.stats.sem(licks[stim_type][active_spout],
                                              axis=2)
            for ind, spout in enumerate(range(mean_across_mice.shape[0])):
                m = mean_across_mice[spout, :]
                s = sem_across_mice[spout, :]
                plt.fill_between(x=np.arange(len(m))/fps-t0, y1=m-s, y2=m+s,
                                 label=spout, color=colors[ind], alpha=0.3)
                plt.plot(np.arange(len(m))/fps-t0, m,
                         label=spout, color=colors[ind])
                if event_times is not None:
                    [plt.axvline(x, color='k', linestyle='--', linewidth=0.5)
                     for x in event_times]
                plt.ylim([-0.2, 8])
            if iter > 1:
                plt.gca().axes.yaxis.set_ticklabels([])
            iter += 1
            plt.title('Stim: {}, Active: {}'.format(stim_type, active_spout))


def concatenate_across_mice_licks_by_stim(all_mean_licks_by_stim,
                                          datasets,
                                          which_mice=None,
                                          inds=None):
    joined_licks_by_stim = dict()  # Concatenate all mice at lowest dict level.
    for kk in inds:
        dd = datasets[kk]
        if dd[2] in which_mice:
            mean_licks_by_stim = all_mean_licks_by_stim[kk]
            for stim_type in mean_licks_by_stim.keys():
                mean_licks_by_active_spout = mean_licks_by_stim[stim_type]
                for active_spout in mean_licks_by_active_spout.keys():

                    if stim_type not in joined_licks_by_stim.keys():
                        joined_licks_by_stim[stim_type] = dict()

                    if (active_spout not in
                            joined_licks_by_stim[stim_type].keys()):
                        joined_licks_by_stim[stim_type][active_spout] = \
                            mean_licks_by_active_spout[active_spout]
                    else:
                        joined_licks_by_stim[stim_type][active_spout] = \
                            np.dstack(
                                (joined_licks_by_stim[stim_type][active_spout],
                                 mean_licks_by_active_spout[active_spout]))

    return joined_licks_by_stim


def plot_total_licks_by_stim(licks_by_stim, patterns, title_str,
                             width=0.4, do_bar=True):
    """Compare total licks (summed across all spouts) during stim vs. not stim.

    :param licks_by_stim: dict. keys: stim pattern. val: dict with keys: 0 or 1
                                      for nostim or stim, val: mean licks for
                                      each training session.
    :param patterns: list of stim pattern names
    :param title_str: string.
    :param width: width of bars in bar chart.
    :param do_bar: bool. True for bar chart, false for paired plot.
    :return: pvals
    """
    # Get paired pvals
    pvals = []
    for ind, pattern in enumerate(patterns):
        _, p = scipy.stats.ttest_rel(licks_by_stim[pattern][0],
                                     licks_by_stim[pattern][1])
        pvals.append(p)

    _, pval_fdr, _, _ = mt.multipletests([pvals[x] for x in range(len(pvals))],
                                         alpha=0.05, method='fdr_bh')
    print(title_str)
    print('pvals uncorrected:')
    print(np.round(np.array(pvals), 4))
    print('pvals fdr-bh:')
    print(np.round(np.array(pval_fdr), 4))

    # Now make paired plot
    if not do_bar:
        plt.figure()
        for ind, pattern in enumerate(patterns):
            y = np.vstack([licks_by_stim[pattern][0],
                           licks_by_stim[pattern][1]])
            x = np.vstack([np.ones(y.shape[0]) * (ind - width / 2),
                           np.ones(y.shape[0]) * (ind + width / 2)])
            plt.plot(x, y, '.-')

        plt.title(title_str + ' paired t, bonf')
        plt.ylabel('Licks')
        plt.gca().set_xticks(np.arange(len(patterns)))
        plt.gca().set_xticklabels(patterns, rotation=45)
        # plt.suptitle(np.round(np.array(pvals)*len(pvals), 3))
        plt.suptitle(np.round(np.array(pval_fdr), 3))

    # Now make bar plot  (factor this!!)
    if do_bar:
        plt.figure()
        for ind, pattern in enumerate(patterns):
            y = licks_by_stim[pattern][0]  # nostim
            plt.bar(ind - width / 2, np.mean(y), width, yerr=np.std(y),
                    color=[.5, .5, .5], label='no-stim')

            y = licks_by_stim[pattern][1]  # stim
            plt.bar(ind + width / 2, np.mean(y), width, yerr=np.std(y),
                    color=[0, .2, .6],
                    label='stim')
        plt.title(title_str + ' paired t, bonf')
        plt.ylabel('Licks')
        plt.gca().set_xticks(np.arange(len(patterns)))
        plt.gca().set_xticklabels(patterns, rotation=45)
        # plt.suptitle(np.round(np.array(pvals)*len(pvals), 3))
        plt.suptitle(np.round(np.array(pval_fdr), 3))

    return pvals


def plot_mean_licks_by_stim(all_mean_licks_by_stim, datasets, inds):

    for kk in inds:
        d = datasets[kk]
        mean_licks_by_stim = all_mean_licks_by_stim[kk]

        plt.figure(figsize=(15, 2))
        iter = 1
        for stim_type in mean_licks_by_stim.keys():
            mean_licks_by_active_spout = mean_licks_by_stim[stim_type]
            for active_spout in mean_licks_by_active_spout.keys():
                plt.subplot(1, 6, iter)
                plt.plot(mean_licks_by_active_spout[active_spout].T)
                iter += 1
                plt.title('Stim: {}, Active: {}'.format(stim_type,
                                                        active_spout))

        plt.suptitle('{}, {}'.format(d[1], d[2]))
        plt.tight_layout()


def get_mean_licks_by_stim(bd, which_trials, smooth=None):
    """
    For stim and non stim trials get the
    mean lick rate to each spout for each target spout.
    :param bd: BpodDataset.
    :param which_trials: boolean array indicating which trials to use.
    :param smooth: int. If not None, will smooth traces by specified window.
    :return: dict. keys are stim_types. For each stim type, there is a dict
                   which contains, for each target spout, the lick
                   rate to each spout.
    """

    mean_licks_by_stim = dict()
    for stim_type in np.unique(bd.stim_types):
        specific_trials = np.logical_and(which_trials,
                                         bd.stim_types == stim_type)
        mean_licks_by_stim[stim_type] = get_mean_licks_by_active_spout(
            bd, specific_trials, smooth=smooth)

    return mean_licks_by_stim


def get_mean_licks_by_active_spout(bd, which_trials, smooth=None):
    """
    For trials to each target spout,
    get the mean lick rate to each spout.
    :param bd: BpodDataset.
    :param which_trials: boolean array indicating which trials to use.
    :param smooth: int. If not None, will smooth traces by specified window.
    :return: dict. keys are active_spout.
    """

    mean_licks_by_active_spout = dict()
    for active_spout in np.unique(bd.spout_positions):
        specific_trials = np.logical_and(which_trials,
                                         bd.spout_positions == active_spout)
        mean_licks_by_active_spout[active_spout] = get_mean_licks(
            bd, specific_trials, smooth=smooth)

    return mean_licks_by_active_spout


def get_mean_licks(bd, which_trials, smooth=None):
    """
    Average over the specified trials
    to obtain the mean lick rate to each spout.
    :param bd: BpodDataset.
    :param which_trials: boolean array indicating which trials to use.
    :param smooth: int. If not None, will smooth traces by specified window.
    :return: array [spouts x timepoints]
    """

    spout_mean_licks = []
    for spout in bd.spout_lick_rates.keys():
        lick_rates = bd.spout_lick_rates[spout]
        mean_lick = np.mean(lick_rates[which_trials, :], axis=0)
        if smooth is not None:
            mean_lick = scipy.signal.savgol_filter(mean_lick, smooth, 1)
        spout_mean_licks.append(mean_lick)

    spout_mean_licks = np.vstack(spout_mean_licks)
    return spout_mean_licks


def plot_stim_vs_nostim_licks(bd, nostim, stim, nostim_success, stim_success):
    """
    Generates a plot showing the actual licks as well as a rose plot
    histogram summary for stim and nonstim conditions.

    :param bd: BpodDataset object.
    :param nostim: bool array indicating which trials are non-stim trials.
    :param stim: bool array indicating which trials are stim trials.
    :param nostim_success: float. fraction of nonstim trials where the mouse
                           successfully performed the task.
    :param stim_success: float. fraction of stim trials where the mouse
                           successfully performed the task.
    :return:
    """
    titles = ['Nostim: {:.2f}'.format(nostim_success / np.sum(nostim)),
              'Stim: {:.2f}'.format(stim_success / np.sum(stim))]

    plt.figure(figsize=(15, 5))
    stim_interval = bd.stim_interval - bd.stimulus_times[0]  # Check?
    if stim_interval[1] == 1:  # This was a bug in the logging code
        stim_interval[1] = 1.5
    bd.plot_stim_licks(fig_save_dir=None, stim_interval=stim_interval,
                       ax=[plt.subplot(221), plt.subplot(223)],
                       titles=titles)

    bd.plot_spout_selectivity(trial_subset=nostim,
                              alt_colors=True, do_save=False,
                              min_trial=15, subplot_strs=['243', '244'],
                              title_strs=['nostim, pre', 'nostim, post'])

    bd.plot_spout_selectivity(trial_subset=stim,
                              alt_colors=True, do_save=False,
                              min_trial=15, subplot_strs=['247', '248'],
                              title_strs=['stim, pre', 'stim, post'])


def summarize_stim_vs_nostim(bd, fig_save_dir=None,
                             min_trial=15, max_trial=190,
                             do_plot=False,
                             just_nogo=False, just_go=False):
    """
    Computes fraction of stim and nonstim trials where the mouse
    successfully performed the task.
    Optionally, plots a summary of the actual licks under each condition.

    :param bd: BpodDataset.
    :param fig_save_dir: If not None, saves plot.
    :param min_trial: int. cut off the first few trials of a session.
    :param max_trial: int. cut off the last few trials of a session.
    :param do_plot: bool.
    :return nostim_success: float. fraction of nonstim trials where the mouse
                           successfully performed the task.
            stim_success: float. fraction of stim trials where the mouse
                           successfully performed the task.
    """

    stim = bd.stim_types.astype(bool)
    nostim = ~bd.stim_types.astype(bool)

    include_trials = np.ones(stim.shape).astype(bool)
    include_trials[:min_trial] = 0
    include_trials[max_trial:] = 0

    if just_nogo:
        print('Just including nogo')
        include_trials[~(bd.trial_types == 4)] = 0
        print('Total nogo trials {}'.format(np.sum(include_trials)))

    if just_go:
        print('Just including go')
        include_trials[(bd.trial_types == 4)] = 0
        print('Total go trials {}'.format(np.sum(include_trials)))

    stim = stim*include_trials
    nostim = nostim*include_trials

    stim_success = np.sum(np.logical_and.reduce((stim,
                                                 bd.success,
                                                 include_trials)))
    nostim_success = np.sum(np.logical_and.reduce((nostim,
                                                   bd.success,
                                                   include_trials)))

    if do_plot:
        plot_stim_vs_nostim_licks(bd, nostim, stim, nostim_success,
                                  stim_success)

        if fig_save_dir is not None:
            print('Saving to: ',
                  os.path.join(fig_save_dir, 'spout_stim_nostim_licks.png'))
            plt.savefig(
                os.path.join(fig_save_dir, 'spout_stim_nostim_licks.png'))

    nostim_success_fraction = nostim_success / np.sum(nostim)
    stim_success_fraction = stim_success / np.sum(stim)

    return stim_success_fraction, nostim_success_fraction


def get_dataset_by_property(datasets, property_str):
    """
    Find the indices of all datasets with a certain
    property.

    :param datasets: list of arrays, where each array
                     contains info about that dataset.
    :param property_str: i.e. 'm15' or 'pre-m'
    :return: inds: list of indices
    """
    inds = []
    for kk, d in enumerate(datasets):
        for entry in d:
            if property_str in entry:
                inds.append(kk)
                break
    return inds


def polar_plot_with_errbar(mean_vals, err_vals, theta, colors,
                           do_normalize=True, ax=None):
    """
    Make polar plot with shaded error bars around each point.
    :param mean_vals: [n_active_spouts x n_licked_spouts]. For each
                      active spout, the distribution of mean licks to
                      each of the spouts.
    :param err_vals: [n_active_spouts x n_licked_spouts]. For each
                      active spout, the error (i.e. SEM) of licks to
                      each of the spouts.
    :param theta: [n_active_spouts]. In radians.
    :param colors: [n_active_spouts]
    :return: nothing.
    """
    if ax is None:
        ax = plt.gca()

    nspouts = mean_vals.shape[0]

    ax.set_rmax(1)
    ax.set_rticks([])
    ax.set_xticks(np.mod(theta, 2 * np.pi))
    ax.set_xticklabels(np.arange(nspouts) + 1)

    for target_spout in range(mean_vals.shape[0]):
        d = mean_vals[target_spout, :]
        s = err_vals[target_spout, :]
        d = d / np.max(d)
        ax.plot(np.append(theta, [0, theta[0]]), np.append(d, [0, d[0]]),
                color=colors[target_spout].replace('white', 'black'),
                linewidth=1)

        ax.fill_between(np.append(theta, [0, theta[0]]),
                        np.append(d, [0, d[0]]) - np.append(s, [0, s[0]]),
                        np.append(d, [0, d[0]]) + np.append(s, [0, s[0]]),
                        color=colors[target_spout].replace('white',
                                                           'black'),
                        alpha=0.3)

        import matplotlib

        v = matplotlib.__version__.split('.')
        if int(v[0]) > 1 and int(v[1]) > 0:
            # If you upgrade to matplotlib v2.1 then you should be able to use:
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_rmax(1)
        else:
            pass
            # warning('Please upgrade matplotlib to v2.1 or greater.')


def polar_lick_plot_across_mice(joined_licks_by_stim, frame_range=None,
                                do_normalize=True):
    """
    Make polar plot with shaded error bars around each point to compare stim
    vs. no stim.
    :param joined_licks_by_stim: [stim/nostim][spout][nspouts x time x session]
    :param t_range: an array containing the start and end time
                  in seconds to use when generating the plot.
                  t=0 is the beginning of the trial (not odor delivery).
    :param do_normalize: normalize lick rates to max rate across the spouts,
                        before computing mean and sem across mice.
    :return:
    """

    colors = ['orange', 'c', 'r']
    # colors = colors[::-1]  ### Flip order for sake of plotting order

    fig, axarr = plt.subplots(
        1, 2, subplot_kw=dict(projection='polar'), figsize=(9, 3))
    theta = np.deg2rad(np.array([15, 90, 165]))
    titles = ['nostim', 'stim']

    for sind, stim_type in enumerate(joined_licks_by_stim.keys()):
        active_spouts = joined_licks_by_stim[stim_type].keys()
        active_spouts = np.array([x for x in active_spouts])

        active_spouts = active_spouts[::-1]  # Flip order for plotting
        nspouts = len(active_spouts)
        spout_dist = np.zeros((nspouts, nspouts))
        sem_spout_dist = np.zeros((nspouts, nspouts))
        for active_spout_iter, active_spout in enumerate(active_spouts):
            if frame_range is None:
                frame_range = np.arange(mean_across_mice.shape[1])

            if do_normalize:
                spout_licks = joined_licks_by_stim[stim_type][active_spout]
                mean_across_time = np.mean(
                    spout_licks[:, frame_range, :], axis=1)
                mean_across_time /= np.max(mean_across_time, axis=0)
                m = np.mean(mean_across_time[active_spouts-1, :], axis=1)
                s = scipy.stats.sem(
                    mean_across_time[active_spouts-1, :], axis=1)
            else:
                mean_across_mice = np.mean(
                    joined_licks_by_stim[stim_type][active_spout], axis=2)
                sem_across_mice = scipy.stats.sem(
                    joined_licks_by_stim[stim_type][active_spout], axis=2)

                m = np.mean(
                    mean_across_mice[:, frame_range][active_spouts-1], axis=1)
                s = np.mean(
                    sem_across_mice[:, frame_range][active_spouts-1], axis=1)
            spout_dist[active_spout_iter, :] = m
            sem_spout_dist[active_spout_iter, :] = s

        ax = axarr[sind]
        plt.sca(ax)

        polar_plot_with_errbar(spout_dist, sem_spout_dist, theta, colors,
                               do_normalize=True, ax=None)

        plt.xlabel(titles[sind])
        plt.gcf().subplots_adjust(
            wspace=-.2, bottom=0, top=1, right=1)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().xaxis.labelpad = -15  # Contol padding of xlabel


# def get_pvalue_compared_with_led_control(mean_scores, which_mice,
#                                          pattern, p_method='ttest'):
#     """
#     Compare behavioral performance with a provided stim pattern
#     with the corresponding LED control.
#     Not corrected for multiple comparisons.
#     :param mean_scores: pandas dataframe.
#     :param pattern: string. i.e. 'odor-a'
#     :param p_method: 'ttest' or 'ranksums'
#     :return: pvalue
#     """
#     cat1 = mean_scores.loc[((mean_scores['Pattern'] == pattern) &
#                             (mean_scores['Mouse'].isin(which_mice)))]
#
#     if pattern.find('odor') != -1:
#         cat2 = mean_scores.loc[((mean_scores['Pattern'] == 'odor-led') &
#                                 (mean_scores['Mouse'].isin(which_mice)))]
#     else:
#         cat2 = mean_scores.loc[((mean_scores['Pattern'] == 'pre-led') &
#                                 (mean_scores['Mouse'].isin(which_mice)))]
#
#     if p_method == 'ttest':
#         _, p = scipy.stats.ttest_rel(cat2['success_ratio'],
#                                      cat1['success_ratio'])
#     elif p_method == 'ranksums':
#         _, p = scipy.stats.ranksums(cat2['success_ratio'],
#                                     cat1['success_ratio'])
#     else:
#         p = -1
#
#     return p


def get_stim_pval(mean_scores, which_mice,
                  pattern1, pattern2,
                  p_method='ttest',
                  which_stat='success_ratio'):
    """
    Compare behavioral performance between
    two specified stim patterns.
    Not corrected for multiple comparisons.
    :param mean_scores: pandas dataframe.
    :param pattern: string. i.e. 'odor-a'
    :param p_method: 'ttest' or 'ranksums'
    :return: pvalue
    """
    cat1 = mean_scores.loc[((mean_scores['Pattern'] == pattern1) &
                            (mean_scores['Mouse'].isin(which_mice)))]
    cat2 = mean_scores.loc[((mean_scores['Pattern'] == pattern2) &
                            (mean_scores['Mouse'].isin(which_mice)))]

    # print(cat1['success_ratio'])
    # print(cat2['success_ratio'])
    if p_method == 'ttest':
        _, p = scipy.stats.ttest_rel(cat2[which_stat],
                                     cat1[which_stat])
    elif p_method == 'ranksums':
        _, p = scipy.stats.ranksums(cat2[which_stat],
                                    cat1[which_stat])
    else:
        print('{} not yet implemented.'.format(p_method))
        p = -1

    return p


def get_stim_pvals(mean_scores, patterns, which_mice,
                   which_comparison='peri-odor',
                   which_stat='success_ratio'):
    """
    Compute the specific statistical tests for each
    group of experiments.
    :param patterns:
    :param which_mice:
    :param which_comparison: 'peri-odor', or 'preodor'
    :return:
    """

    if which_comparison == 'peri-odor':
        # Repeated measures Anova before post-hoc ttests
        # from statsmodels.stats.anova import AnovaRM
        # aovrm = AnovaRM(
        #   mean_scores, 'success_ratio', 'Mouse', within=['Pattern'])
        # res = aovrm.fit()
        # print(res)

        comparisons = [
            ['a/l', 'odor-a', 'odor-led'],
            ['m/l', 'odor-m', 'odor-led'],
            ['p/l', 'odor-p', 'odor-led'],
            ['s/l', 'odor-s', 'odor-led'],
            ['a/m', 'odor-a', 'odor-m'],
            ['m/p', 'odor-m', 'odor-p'],
            ['a/p', 'odor-a', 'odor-p'],
            ['p/s', 'odor-p', 'odor-s'],
        ]
        pattern_p = [get_stim_pval(mean_scores, which_mice,
                                   comparisons[i][1],
                                   comparisons[i][2],
                                   which_stat=which_stat)
                     for i in range(len(comparisons))]

        p = np.array(pattern_p)
        print('Before fdr correction: {}'.format(p))
        sig = mt.multipletests([p[x] for x in range(len(p))],
                               alpha=0.05, method='fdr_bh')
        s = sig[0].astype(int)
        p = sig[1]
        pval_str = ['{}: {:.3}, '.format(comparisons[i][0], p[i])
                    for i in range(len(comparisons))]
        pval_str = ('ttest-rel, fdr_bh {}: '.format(which_stat) +
                    ''.join(pval_str))

    if which_comparison == 'pre-odor':
        comparisons = [
            ['a/l', 'pre-a', 'pre-led'],
            ['m/l', 'pre-m', 'pre-led'],
            ['p/l', 'pre-p', 'pre-led'],
        ]
        pattern_p = [get_stim_pval(mean_scores, which_mice,
                                   comparisons[i][1],
                                   comparisons[i][2],
                                   which_stat=which_stat)
                     for i in range(len(comparisons))]

        p = np.array(pattern_p)
        print('Before fdr correction: {}'.format(p))
        sig = mt.multipletests([p[x] for x in range(len(p))],
                               alpha=0.05, method='fdr_bh')
        s = sig[0].astype(int)
        p = sig[1]
        pval_str = ['{}: {:.3}, '.format(comparisons[i][0], p[i])
                    for i in range(len(comparisons))]
        pval_str = 'ttest-rel, fdr_bh ' + ''.join(pval_str)

    return pval_str


def plot_raw_stim_performance(patterns, which_mice, mean_scores,
                              mouse_colors_dict,
                              p_method='ttest',
                              show_indiv_points=True,
                              stat1='stim_success',
                              stat2='nostim_success'):
    """
    Compare behavioral performance, across mice, for different
    stimulation patterns.

    :param patterns: list. i.e. ['odor-a', 'odor-m']
    :param which_mice: list. i.e. ['m12', 'm14']
    :param mean_scores: a pandas dataframe containing
                        all of the data.
    :return:
    """

    plt.figure()

    # Plot stim/nostim success ratio
    if show_indiv_points:
        for ind, pattern in enumerate(patterns):
            s_pattern_mean = 0
            ns_pattern_mean = 0
            mean_iter = 0
            for mouse in which_mice:
                a = mean_scores[mean_scores['Mouse'] == mouse]
                # print(np.array(a[a['Pattern'] == pattern]['success_ratio']))

                stim_success = np.array(a[a['Pattern'] == pattern][stat1])
                nostim_success = np.array(a[a['Pattern'] == pattern][stat2])
                for ss, ns in zip(stim_success, nostim_success):
                    s_pattern_mean += ss
                    ns_pattern_mean += ns
                    mean_iter += 1
                    w = 0.2
                    plt.plot([ind-w, ind+w], [ns, ss], 'o-',
                             color=mouse_colors_dict[mouse],
                             markersize=2,
                             linewidth=1)

            s_pattern_mean /= mean_iter
            ns_pattern_mean /= mean_iter

        plt.xticks(np.arange(len(patterns)), patterns, rotation=30)
    else:
        flatui = ["#9b59b6", "#95a5a6", "#3498db", "#e74c3c", "#34495e",
                  "#2ecc71"]
        cc = sns.color_palette(flatui)
        ax = sns.factorplot(x='Pattern', y='success_ratio', hue='Control_stim',
                            kind='bar',
                            palette=cc,
                            data=mean_scores, legend=False, legend_out=True,
                            aspect=2,
                            order=patterns, dodge=False)
    #     plt.gca().set_xticklabels(rotation=30)
    plt.gca().set(ylabel='Success rate')
    plt.gca().set_ylim([0.35, 1.3])
    plt.gca().set_xlim([-0.5, len(patterns)-0.5])


def plot_stim_performance(patterns, which_mice, mean_scores,
                          mouse_colors_dict,
                          p_method='ttest',
                          show_indiv_points=True, ):
    """
    Compare behavioral performance, across mice, for different
    stimulation patterns.

    :param patterns: list. i.e. ['odor-a', 'odor-m']
    :param which_mice: list. i.e. ['m12', 'm14']
    :param mean_scores: a pandas dataframe containing
                        all of the data.
    :return:
    """

    plt.figure()

    # Plot stim/nostim success ratio
    if show_indiv_points:
        for ind, pattern in enumerate(patterns):
            pattern_mean = 0
            mean_iter = 0
            for mouse in which_mice:
                a = mean_scores[mean_scores['Mouse'] == mouse]
                # print(np.array(a[a['Pattern'] == pattern]['success_ratio']))

                success_ratio = np.array(
                    a[a['Pattern'] == pattern]['success_ratio'])
                for sr in success_ratio:
                    pattern_mean += sr
                    mean_iter += 1
                    plt.plot(ind, sr, 'o',
                             color=mouse_colors_dict[mouse],
                             markersize=3)

            pattern_mean /= mean_iter
            plt.plot(ind, pattern_mean, 'r_', markersize=20, markeredgewidth=2)
            if pattern.find('led') != -1:
                plt.plot(ind, pattern_mean, 'k_', markersize=20,
                         markeredgewidth=2)

        plt.xticks(np.arange(len(patterns)), patterns, rotation=30)
    else:
        flatui = ["#9b59b6", "#95a5a6", "#3498db", "#e74c3c", "#34495e",
                  "#2ecc71"]
        cc = sns.color_palette(flatui)
        ax = sns.factorplot(x='Pattern', y='success_ratio', hue='Control_stim',
                            kind='bar',
                            palette=cc,
                            data=mean_scores, legend=False, legend_out=True,
                            aspect=2,
                            order=patterns, dodge=False)
    #     plt.gca().set_xticklabels(rotation=30)
    plt.gca().set(ylabel='Stim / Nostim success')
    plt.gca().set_ylim([0.4, 1.2])
    plt.gca().set_xlim([-0.5, len(patterns)-0.5])


def organize_stim_data_into_dataframe(all_bd, datasets,
                                      do_plot_all=False,
                                      do_plot_indiv=False):
    """
    Organizes behavior data into a pandas dataframe that
    categorizes according to stim/nostim, stim pattern, and mouse
    and that contains entries describing the performance of
    the mouse under those conditions.

    :param all_bd: list of BpodDataset objects.
    :param datasets: list of metadata associated with all_bd.
    :param do_plot_all: Plot comparison between stim and control animals.
    :param do_plot_indiv: Make individual plots for each dataset.
    :return: scores. A pandas dataframe.
    """

    scores = pd.DataFrame()

    for stim_pattern in ['pre-m', 'pre-a', 'odor-m', 'odor-a', 'odor-led',
                         'pre-led', 'odor-p', 'odor-lat', 'odor-s', 'odor-v',
                         'pre-p']:
        all_success_ratio = []
        inds = get_dataset_by_property(datasets, stim_pattern)
        labels = []
        for kk in inds:
            bd = all_bd[kk]
            d = datasets[kk]
            stim_success, nostim_success = summarize_stim_vs_nostim(
                bd,
                fig_save_dir=None,
                min_trial=15,
                max_trial=190,
                do_plot=do_plot_indiv,
                just_nogo=False,
                just_go=False)

            ng_stim_success, ng_nostim_success = summarize_stim_vs_nostim(
                bd,
                fig_save_dir=None,
                min_trial=15,
                max_trial=190,
                do_plot=do_plot_indiv,
                just_nogo=True,
                just_go=False)

            g_stim_success, g_nostim_success = summarize_stim_vs_nostim(
                bd,
                fig_save_dir=None,
                min_trial=15,
                max_trial=190,
                do_plot=do_plot_indiv,
                just_nogo=False,
                just_go=True)

            success_ratio = stim_success / nostim_success
            all_success_ratio.append(success_ratio)
            labels.append([d[2], d[1]])

            is_control_mouse = d[2].find('thy1') != -1
            is_control_stim = d[1].find('led') != -1
            is_pre = d[1].find('pre') != -1
            #         if not is_control_mouse:
            data = {'Pattern': d[1], 'Mouse': d[2],
                    'Control': is_control_mouse,
                    'Control_stim': is_control_stim, 'Pre': is_pre,
                    'stim_success': stim_success,
                    'nostim_success': nostim_success,
                    'success_ratio': stim_success / nostim_success,
                    'success_diff': nostim_success - stim_success,
                    'ng_stim_success': ng_stim_success,
                    'ng_nostim_success': ng_nostim_success,
                    'ng_success_diff': ng_nostim_success - ng_stim_success,
                    'g_stim_success': g_stim_success,
                    'g_nostim_success': g_nostim_success,
                    'g_success_diff': g_nostim_success - g_stim_success,
                    }
            scores = scores.append(data, ignore_index=True)

            if do_plot_indiv:
                plt.suptitle(
                    '{} {}: {:.2f}'.format(d[2], d[1], success_ratio),
                    fontsize=20)

        if do_plot_all:
            thy1_inds = get_dataset_by_property(labels, 'thy1')
            plt.figure()
            for i in range(len(labels)):
                if i in thy1_inds:
                    x = 1
                else:
                    x = 2
                plt.plot(x, all_success_ratio[i], 'o')
                plt.xlim([0.5, 2.5])
                plt.ylabel('stim/nostim success ratio')
                plt.xticks([1, 2], ['thy1', 'vgat'])
                plt.title(stim_pattern)
                plt.ylim([0.4, 1])

    return scores
