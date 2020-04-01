#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Helper functions to support analysis of individual fluorescence traces.

Created on Nov 1 2017

@author: tamachado@stanford.edu
"""
from collections import defaultdict
from past.utils import old_div
from warnings import warn
import warnings
import cycler
import math

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF, PCA, SparsePCA
from sklearn.cross_decomposition import PLSRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import rc, animation
from IPython.display import HTML
import scipy.sparse as sparse
from scipy.sparse import csc_matrix

import scipy.stats as stats
import pandas as pd
import seaborn as sns
import pycircstat as circ
from scipy.ndimage.measurements import center_of_mass
import numpy as np
import os
import scipy.io as sio

from sklearn.cluster.bicluster import SpectralCoclustering

from scipy.ndimage.filters import gaussian_filter1d
import scipy.ndimage.measurements as measurements
import scikit_posthocs as sp
from skimage import measure
import scipy.spatial
import skimage.draw

import statsmodels.stats.multitest as mt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib as mpl
from matplotlib.collections import LineCollection

from ipywidgets import widgets
from bokeh.io import output_file, show, output_notebook, push_notebook
from bokeh.layouts import widgetbox, column, row, layout
from bokeh.models.widgets import Slider, TextInput
from bokeh.models import Range1d
import bokeh.models as models
from bokeh.plotting import figure
from IPython.display import display

from IPython.core.debugger import set_trace
from ipywidgets import widgets

try:
    import bokeh
    import bokeh.models as models
    import bokeh.plotting as bpl
    from bokeh.models import CustomJS, ColumnDataSource, Range1d
    from bokeh.plotting import figure
except:
    print("Bokeh could not be loaded.")

from cosmos.traces.cell_plotter import CellPlotter
import cosmos.imaging.atlas_registration as reg

import random

def flatten_traces(traces):
    """
    Flatten a matrix [nsources x time x trial]
    to [nsources x time*trial]
    :param traces:
    :return:
    """
    flattened = np.reshape(traces,
                        (traces.shape[0], traces.shape[1]*traces.shape[2]),
                        order='F')
    return flattened


def make_visual_to_other_area_comparison_plots(all_gs):
    """
    This analysis looks at all 'populated areas' that
    have more than k cells; k is a parameter to CosmosTraces.
    """
    score = '$r^2$'
    summary = pd.DataFrame()

    for gs in all_gs:
        for area in gs.traces.populated_areas:
            idx = gs.traces.cells_in_region[gs.traces.regions[area]]
            for ind in idx:

                # Get hemisphere
                if gs.traces.hemisphere_of_cell[ind] == 0:
                    side = 'Left'
                else:
                    side = 'Right'
                aa = 'Visual' if 'VIS' in area else 'Other'
                d = {'Cortical Region': aa, 'Hemisphere': side,
                     'OSI': gs.osis[ind], 'DSI': gs.dsis[ind],
                     score: gs.corr[ind]**2}
                summary = summary.append(d, ignore_index=True)

    plt.figure(figsize=(2, 2))
    regions = ['Visual', 'Other']
    areas = ['Left', 'Right']
    p_vals = []
    sns.boxplot(data=summary, x='Cortical Region',
                y='OSI', hue='Hemisphere', order=regions, hue_order=areas)
    plt.grid(True, alpha=0.2)
    plt.xlabel('Area')

    # Print the n of sources in each condition
    for region in ['Visual', 'Other']:
        for side in ['Left', 'Right']:
            print(region, side, 'n =',
                  len(summary.loc[(summary['Cortical Region'] ==
                                   region) &
                                  (summary['Hemisphere'] == side)]))

    # Do pairwise mann whitney tests, and then bonferroni correct them
    # There are six valid comparisons between the four conditions 4*3/2
    comp = 0
    print('BH-FDR, a=0.05 corrected p values from Mann-Whitney U Test.')
    c1 = []
    c2 = []
    pvals = []
    names = []
    for ii, region1 in enumerate(regions):
        for jj, region2 in enumerate(regions):
            for kk, area1 in enumerate(areas):
                for ll, area2 in enumerate(areas):
                    if area1 == area2 and region1 == region2:
                        continue
                    if area1 + region1 in c2 and area2 + region2 in c1:
                        continue
                    c1.append(area1 + region1)
                    c2.append(area2 + region2)
                    comp += 1
                    sx = ((summary['Cortical Region'] == region1) &
                          (summary['Hemisphere'] == area1))
                    sy = ((summary['Cortical Region'] == region2) &
                          (summary['Hemisphere'] == area2))
                    x = summary.where(sx)['OSI'].as_matrix()
                    y = summary.where(sy)['OSI'].as_matrix()
                    p = stats.mannwhitneyu(x[np.isfinite(x)],
                                           y[np.isfinite(y)]).pvalue
                    pvals.append(p)
                    names.append(region1 + ' ' + area1 +
                                 ' vs. ' + region2 + ' ' + area2)

    alpha = 0.05
    reject, pvals, _, _ = mt.multipletests(pvals, alpha=alpha, method='fdr_bh')
    for p, name in zip(pvals, names):
        #p *= nComps
        stars = ''
        if p < 0.0001:
            stars = '****'
        elif p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        print(stars, p, name)
    #if comp != nComps:
    #    raise ValueError('Unexpected number of comparisons!')
    sns.despine()
    p_vals.append(p)
    plt.tight_layout()
    return summary, p_vals


def moving_avg(x, do_median=False, window=20):
    env = np.zeros_like(x)
    for i in range(len(x)):
        # Ignore warning that arises when
        # an array is only composed of nans.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if do_median:
                env[i] = np.nanmedian(x[max(i - window, 0):i + 1])
            else:
                env[i] = np.nanmean(x[max(i - window, 0):i + 1])

    return env


def df(trace, baseline_idx=None, return_f0=False, percentile=5, fixed_f0=None):
    """ Compute the df/f for a fluorescence trace. """
    if np.min(trace) != 0:
        trace = trace - np.min(trace)
    if baseline_idx is None:
        f0 = np.percentile(trace, percentile)
    elif fixed_f0 is None:
        f0 = fixed_f0
    else:
        f0 = np.mean(trace[baseline_idx])
    if f0 > 10 ** float(-1 * np.finfo(np.float64).precision):
        trace = (trace - f0) / f0
    if return_f0:
        return trace, f0
    else:
        return trace


def norm_trace(trace, thresh=97):
    """ Normalize a fluorescence trace. """
    if np.min(trace) != 0:
        trace = trace - np.min(trace)
    trace = trace / np.percentile(trace, thresh)
    trace = trace - np.percentile(trace, 5)
    return trace


def get_task_modulated(rates, trial_sets, max_pval=0.05):
    """
    Assess whether each source is significantly task modulated,
    i.e. that the mean baseline activity is significantly different than the
    mean task activity.
    :param rates: [ncells, nframes_per_trial, ntrials]
    :param trial_sets: tuple of bool arrays each of length [ntrials], showing
                        whether a trial is in the respective trial type.
    :param max_pval: i.e. 0.05. Significance level.
    """
    pval = np.zeros(rates.shape[0])
    go_trials = np.logical_or.reduce(
        (trial_sets[0], trial_sets[1], trial_sets[2]))

    for i in range(rates.shape[0]):
        baseline = np.mean(rates[i, 5:50, :], axis=0)
        task = np.mean(rates[i, 65:150, :], axis=0)
        _, p = scipy.stats.ttest_rel(baseline[go_trials], task[go_trials])
        pval[i] = p
    task_modulated = pval < max_pval / len(pval)

    return task_modulated

def get_task_modulated_gonogo(rates, trial_sets, max_pval=0.05):
    """
    Assess whether each source is significantly task modulated,
    i.e. that the mean baseline activity is significantly different than the
    mean task activity.
    :param rates: [ncells, nframes_per_trial, ntrials]
    :param trial_sets: tuple of bool arrays each of length [ntrials], showing
                        whether a trial is in the respective trial type.
    :param max_pval: i.e. 0.05. Significance level.
    """
    pval = np.zeros((rates.shape[0], 2))

    go_trials = np.logical_or.reduce(
        (trial_sets[0], trial_sets[1], trial_sets[2]))
    nogo_trials = trial_sets[3]

    for i in range(rates.shape[0]):
        baseline = np.mean(rates[i, 5:50, :], axis=0)
        task = np.mean(rates[i, 65:155, :], axis=0)
        _, p_go = scipy.stats.ttest_rel(baseline[go_trials], task[go_trials])
        pval[i, 0] = p_go
        _, p_nogo = scipy.stats.ttest_rel(baseline[nogo_trials], task[nogo_trials])
        pval[i, 1] = p_nogo
    task_modulated = pval < max_pval / len(pval)

    return task_modulated


def get_task_modulated_trial_type(rates, trial_sets, max_pval=0.05):
    """
    Assess whether each source is significantly task modulated,
    i.e. that the mean baseline activity is significantly different than the
    mean task activity.
    :param rates: [ncells, nframes_per_trial, ntrials]
    :param trial_sets: tuple of bool arrays each of length [ntrials], showing
                        whether a trial is in the respective trial type.
    :param max_pval: i.e. 0.05. Significance level.
    """
    ncells = rates.shape[0]
    pval = np.zeros((ncells, 4))

    for k in range(4):
        for i in range(rates.shape[0]):
            baseline = np.mean(rates[i, 5:50, :], axis=0)
            task = np.mean(rates[i, 65:150, :], axis=0)
            _, p = scipy.stats.ttest_rel(baseline[trial_sets[k]], task[trial_sets[k]])
            pval[i, k] = p

    task_modulated = pval < (max_pval / ncells)

    return task_modulated


def centroid_movie(cell_weights, cell_ids, centroids, atlas_tform,
                   max_radius=10, save_dir=None, do_square=False,
                   title_list=None, do_max_norm=False):
    """
    cell_weights: [ncells x ntime]
    """
    cell_weights = cell_weights / np.amax(cell_weights)
    ncells, nt = cell_weights.shape
    _, _, atlas_outline = reg.load_atlas()
    plt.figure(figsize=(10, 10))
    if do_square:
        cell_weights = np.copy(cell_weights)**2
    for t in range(nt):
        print(t, end='...')
        centroids_on_atlas(cell_weights[:, t], cell_ids,
                           centroids, atlas_tform,
                           atlas_outline=atlas_outline, max_radius=max_radius,
                           do_max_norm=do_max_norm)
        plt.xlabel(str(t))
        if title_list is not None:
            plt.title(title_list[t])
        if save_dir is not None:
            plt.savefig(save_dir + '//{:05}.png'.format(t))
        plt.gcf().clear()


def get_atlas_coords(centroids, atlas_tform):
    """
    Transform centroids [cell x 2] into atlas coordinates.
    """
    _, _, atlas_outline = reg.load_atlas()

    atlas_coords = np.zeros((centroids.shape[0], 2))
    for ind in range(centroids.shape[0]):
        atlas_coords[ind, :] = atlas_tform.inverse((centroids[ind, 1],
                                                    centroids[ind, 0]))[0]
    return atlas_coords


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def transform_centroids_to_atlas(centroids, atlas_tform):
    """
    Transform source centroids in pixel space into atlas coordinates
    using a provided sklearn transform function (i.e. CT.atlas_tform).
    :param centroids: [ncells x 2] centroid coordinates.
    :param atlas_tform: sklearn transformation function.

    :returns atlas_coords: [ncells x 2] The transformed coordinates of each
                           centroid.
    """

    centroids = np.copy(centroids)
    cell_ids = np.arange(centroids.shape[0])
    atlas_coords = np.zeros((centroids.shape[0], 2))
    for ind, cell in enumerate(cell_ids):
        atlas_coords[ind, :] = atlas_tform.inverse((centroids[cell, 1],
                                                    centroids[cell, 0]))[0]

    return atlas_coords


def get_colors(ncolors, cmap=plt.cm.jet):
    """
    Return RGBA colors for ncolors spaced
    evently through a given colormap.

    :param ncolors: int.
    :param cmap: plt.cm colormap
    :return:
    """
    print(cmap)
    colors = []
    for i in range(ncolors):
        # print(int(float(i) * 256.0 / 4.0))
        colors.append(cmap(int(float(i) * 256.0 / float(ncolors))))

    return colors


def centroids_on_atlas(cell_weights, cell_ids, centroids, atlas_tform,
                       atlas_outline=None, max_radius=10,
                       do_max_norm=False, set_alpha=False,
                       set_radius=True, highlight_inds=None,
                       vmin=None, vmax=None, fill_highlight=True,
                       rotate90=True, cmap='jet'):
    """
    Plots the centroid of each cell atop an atlas outline,
    where the size and color of each cell marker is determined
    by a cell_weights vector.

    :param cell_weights: ndarray [ncells]. Weight corresponding
                         to each cell.
                         If instead this has dimensions [ncells x 4]
                         then it represents the RGBA
    :param cell_ids: ndarray [ncells]. Indices of the cells to be shown.
    :param centroids: ndarray [total ncells x 2]. The centroid of every
                      cell in the dataset. cell_ids indexes
                      into this array.
    :param atlas_tform: sklearn transform function that was generated
                        based on manually selected keypoints.
                        See CosmosTraces class for more information
                        about this. If this is None, then centroids
                        have been assumed to have been transformed
                        already (before being passed into this
                        function).
    :param atlas_outline: Provide if preloading the atlas
                          using reg.load_atlas() (may be useful
                          if calling this function many times
                          i.e. to generate a movie).
                          Otherwise, will do automatically.
                from sklearn.cluster.bicluster import SpectralCoclustering

    :param max_radius: Maximum radius of plotted circles
    :param set_radius: bool. If False, then radius of all markers
                             is constant (i.e. not weighted according to
                             cell_weights).
    :param highlight_inds: np.array. List of cell_ids that should
                           have a highlight circle outline around
                           the plotted circle.
    :return:
    """

    if atlas_outline is None:
        _, _, atlas_outline = reg.load_atlas()

    if atlas_tform is not None:
        atlas_coords = np.zeros((len(cell_ids), 2))
        for ind, cell in enumerate(cell_ids):
            atlas_coords[ind, :] = \
                atlas_tform.inverse(
                    (centroids[cell, 1], centroids[cell, 0]))[0]
    else:
        atlas_coords = centroids[cell_ids, :]

    atlas_outline = scipy.ndimage.filters.gaussian_filter(
        atlas_outline.astype('float'), sigma=0.5)
    if rotate90:
        plt.imshow(np.rot90(atlas_outline, axes=(1, 0)), cmap='Greys')
    else:
        plt.imshow(atlas_outline, cmap='Greys')

    if len(cell_weights.shape) > 1 and cell_weights.shape[1] == 4:
        colors_provided = True
        rgba = cell_weights
        radius = max_radius
    else:
        if do_max_norm:
            cell_weights = np.abs(cell_weights / np.amax(cell_weights))

        # cmap = plt.cm.get_cmap('jet', 500)
        # rgba = cmap(cell_weights/np.max(cell_weights))
        # rgba = cmap(cell_weights*2)

        if vmax is None:
            vmax = 1
        if vmin is None:
            vmin = 0
        cc = np.linspace(vmin, vmax, 256)
        norm = mpl.colors.Normalize(vmin=min(cc), vmax=max(cc))
        rgba = plt.cm.ScalarMappable(
            norm=norm, cmap=cmap).to_rgba(cell_weights)
                # norm=norm, cmap=plt.cm.hsv).to_rgba(cell_weights)
                # norm=norm, cmap=plt.cm.plasma).to_rgba(cell_weights)
        # elif  cmap == 'jet':
        #     rgba = plt.cm.ScalarMappable(
        #         # norm=norm, cmap=plt.cm.hsv).to_rgba(cell_weights)
        #         norm=norm, cmap=plt.cm.jet).to_rgba(cell_weights)
        #         # norm=norm, cmap=plt.cm.plasma).to_rgba(cell_weights)

        if set_alpha:
            # import pdb; pdb.set_trace()
            alpha = cell_weights / np.nanmax(cell_weights)
            alpha = alpha**0.8
            alpha[alpha > 1] = 1
            alpha[alpha < 0] = 0
            rgba[:, 3] = alpha
        # plt.scatter(atlas_coords[:, 0], atlas_coords[:, 1],
        #             s=max_radius * cell_weights, c=cell_weights, cmap='jet')
        # plt.scatter(atlas_coords[:, 0], atlas_coords[:, 1],
        #            c=cell_weights, alpha=cell_weights, cmap='jet')
        if set_radius:
            radius = max_radius * cell_weights
        else:
            radius = max_radius

    if rotate90:
        # rot_coords = np.zeros(atlas_coords.shape)

        plt.scatter(-atlas_coords[:, 1] + atlas_outline.shape[0],
                    atlas_coords[:, 0],
                    s=radius, c=rgba, linewidths=0)
    else:
        plt.scatter(atlas_coords[:, 0], atlas_coords[:, 1],
                    s=radius, c=rgba, linewidths=0)

    if highlight_inds is not None:
        if fill_highlight:
            fillc = 'k'
        else:
            fillc = rgba[highlight_inds, :]

        if rotate90:
            plt.scatter(
                -atlas_coords[highlight_inds, 1] + atlas_outline.shape[0],
                atlas_coords[highlight_inds, 0],
                s=radius[highlight_inds],
                c=fillc,
                edgecolors='k', linewidths=5)
        else:
            plt.scatter(atlas_coords[highlight_inds, 0],
                        atlas_coords[highlight_inds, 1],
                        s=radius[highlight_inds],
                        c=fillc,
                        edgecolors='k', linewidths=5)

    if rotate90:
        plt.ylim([90, 415])
        plt.xlim([20, 435])
    else:
        plt.xlim([90, 415])
        plt.ylim([20, 435])
    plt.gca().invert_yaxis()
    plt.yticks([])
    plt.xticks([])


def get_trial_sets_COSMOSTrainMultiBlockGNG(BD, min_block_trial,
                                            min_spout_selectivity,
                                            only_successful=False):
    """
    Get the sets of trials included in each trial type,
    specifically for the 3-spout version of
    COSMOSTrainMultiBlockGNG bpod protocol.

    :param BD: BpodDataset class.
    :param min_block_trial: int. Within each block of a certain trial type,
                            how many of the beginning trials to exclude.
    :param min_spout_selectivity: float between 0, 1. If 0, then ignores this.
                                  Else, only includes trials where the mouse
                                  licked selectively to the correct spout.
    :param only_successful: bool. Only include trials where the mouse licked
                                  successfully to the correct spout and
                                  received the biggest possible reward, as well
                                  as correct lick withholding on no-go trials.
    :return: trial_sets: tuple of boolean arrays, indicating which trials
                         are of each trial type.
    """
    if min_spout_selectivity > 0:
        clean_trials = np.zeros(BD.success.shape)
        clean_trials[
            BD.get_clean_trials(min_selectivity=min_spout_selectivity)] = 1
    else:
        clean_trials = np.ones(BD.success.shape)

    if only_successful:
        success = BD.success
    else:
        success = np.ones(BD.success.shape)

    lick_spout1 = np.logical_and.reduce((BD.go_trials.astype('bool'),
                                         success,
                                         BD.spout_positions == 1,
                                         BD.ind_within_block >=
                                         min_block_trial,
                                         clean_trials))
    lick_spout3 = np.logical_and.reduce((BD.go_trials.astype('bool'),
                                         success,
                                         BD.spout_positions == 3,
                                         BD.ind_within_block >=
                                         min_block_trial,
                                         clean_trials))
    lick_spout4 = np.logical_and.reduce((BD.go_trials.astype('bool'),
                                         success,
                                         BD.spout_positions == 4,
                                         BD.ind_within_block >=
                                         min_block_trial,
                                         clean_trials))
    nolick = np.logical_and.reduce((~BD.go_trials.astype('bool'),
                                    success))

    trial_sets = (lick_spout1, lick_spout3, lick_spout4, nolick)
    trial_labels = ('Go 1', 'Go 2', 'Go 3', 'No go')
    return trial_sets, trial_labels


def rescale_covariates(covariate, covariate_led_frames, neural_led_frames,
                       do_debug=False):
    """
    Anti-alias and downsample a covariate so that it has the same sampling
    rate as the neural traces.

    :param covariate:[ntime] Array containing the time series of the covariate.
    :param covariate_led_frames: [ntrials] The covariate-video frame time
                                 of each trial.
    :param neural_led_frames: [ntrials] The neural-video frame
                              time of each trial.

    :return rescaled_covariate: The properly scaled covariate.
    """

    # Check that they have the same number of led frames.
    print('# covariate_led_frames: {}, # neural_led_frames: {}'.format(
        len(covariate_led_frames),
        len(neural_led_frames)))

    import warnings
    warnings.warn(
        'NEED TO DEAL CASE WHERE ARE FEWER COVARIATE LED FRAMES THAN NEURAL.')
    covariate_led_frames = covariate_led_frames[:len(neural_led_frames)]
    neural_led_frames = neural_led_frames[:len(covariate_led_frames)]
    diff_ratio = np.diff(covariate_led_frames) / np.diff(neural_led_frames)
    scale_factor = np.mean(diff_ratio)
    if do_debug:
        plt.figure()
        plt.title('Distribution of ratio of time between ' +
                  'covariate/neural LED frames')
        plt.hist(diff_ratio)

    # Trim covariate to the first LED frame, and also shift the led_frames
    from scipy import interpolate
    x = np.arange(0, len(covariate))
    y = covariate
    ysmooth = scipy.signal.savgol_filter(y, 3, 2)
    f = interpolate.interp1d(x, ysmooth)

    xnew = covariate_led_frames[0] + scale_factor * np.arange(
        neural_led_frames[-1] - neural_led_frames[0])
    ynew = f(xnew)

    rescaled_covariate_led_frames = np.round(
        covariate_led_frames / scale_factor).astype(int)
    rescaled_covariate_led_frames = rescaled_covariate_led_frames - \
        rescaled_covariate_led_frames[0]
    rescaled_covariate = np.expand_dims(ynew, axis=0)

    if do_debug:
        plt.figure(figsize=((400, 5)))
        plt.plot(x, y, 'b', label='raw covariate')
        plt.plot(xnew, ynew, 'r.-', label='rescaled covariate')
        plt.plot(xnew[rescaled_covariate_led_frames[:-1]],
                 ynew[rescaled_covariate_led_frames[:-1]], 'go',
                 label='rescaled LED frames')
        plt.legend()
        plt.title('Comparison between raw and rescaled covariate')

    if np.max(np.abs(np.diff(rescaled_covariate_led_frames) -
                     np.diff(neural_led_frames))) > 2:
        raise('There is a mismatch between covariate LED frames' +
              'and neural LED frames.')

    return rescaled_covariate, rescaled_covariate_led_frames


def reshape_to_trials(C, trial_start_frames, nt, dt, ntrials=None):
    """
    Provided a data matrix [neurons x time], and an array with the
    start frame of each trial, returns a matrix [neurons x time x trial].
    :param C: [neurons x time] time series matrix
    :param trial_start_frames: np.array. For example, led_frames
    :param nt: in seconds. How many seconds worth of each trial to include.
    :param dt: seconds per frame.
    :param ntrials: number of trials to include in the matrix. Optional. If
                    not provided, then uses len(trial_start_frames).
    :returns Ct: a trial structured matrix of the data.
    """

    if ntrials is None:
        ntrials = len(trial_start_frames) - 1
    ncells = C.shape[0]
    nframes = int(nt / dt)
    Ct = np.zeros((ncells, nframes, ntrials))

    total_trials = 0
    for trial in range(ntrials):
        start_frame = trial_start_frames[trial]
        if start_frame + nframes < C.shape[1]:
            total_trials += 1
            Ct[:, :, trial] = C[:, start_frame:start_frame + nframes]

    Ct = Ct[:, :, :total_trials]
    return Ct


def in_ranges(ranges, vals):
    """
    Test whether the values in array vals
    are within any of the ranges provides in
    ranges (a list of lists: [[start1, end1], [start2, end2]]).
    """
    is_in = np.zeros(vals.shape).astype(bool)
    for r in ranges:
        is_in = np.logical_or.reduce(
            (is_in, np.logical_and(vals > r[0], vals <= r[1])))

    return is_in


def cocluster_corr(data, n_clusters):
    """
    Use spectral co-clustering to cluster and reorder
    a correlation matrix.

    :param data: correlation matrix.
    :param n_clusters: int. number of clusters to use.
    :return: ordered_corr: the ordered matrix.
             row_labels: ordering of the rows
             col_labels: ordering of the columns.
    """
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(data)
    row_order = np.argsort(model.row_labels_)
    col_order = np.argsort(model.column_labels_)
    fit_data = data[row_order]
    fit_data = fit_data[:, col_order]

    # db = DBSCAN(eps=0.3, min_samples=10).fit(data)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_
    # fit_data = data[np.argsort(db.labels_)]
    # fit_data = fit_data[:, np.argsort(db.labels_)]

    plt.figure(figsize=(10, 10))
    plt.imshow(fit_data, cmap=plt.cm.bwr, clim=[-1, 1])

    ordered_corr = fit_data
    return ordered_corr, row_order, col_order


def get_windowed_data(data, window):
    """
    Helper function used in get_windowed_corr.

    :param data: [ncells x nframes]
    :param window: int.
    :return: list of windowed chunks of the data.
    """
    windowed_data = []
    window_inds = np.arange(window, data.shape[1]-window, 2*window)
    for t in window_inds:
        windowed_data.append(data[:, t-window:t+window])
    return windowed_data, window_inds


def mean_corr_coeff(data):
    """
    Returns the mean correlation coefficient
    across rows of data, excluding the diagonal.

    :param data: [ncells x nframes]. data.
    :return: mean_corr: float.
    """
    c = np.corrcoef(data)
    d = np.triu(c, k=1).flatten()
    mean_corr = np.nanmean(d[np.where(d)[0]])
    return mean_corr


def get_windowed_corr(data, window=50, do_parallel=True):
    """
    Compute the mean correlation coefficient between
    rows of `data` for a sliding window of specified size.
    The stride of the sliding window is equal to the size
    of the window.

    :param data: [ncells x nframes]
    :param window: int. Half the width of window, in frames.
    :param do_parallel: bool. Use parallel processing.

    :return wc: array. Windowed correlation. Note, this is
                not of length nframes. Index into the corresponding
                data based on w_ind.
    :return w_ind: array. the frame number corresponding to the center
                   of the window used for each entry in wc.
    """

    w, w_ind = get_windowed_data(data, window=window)
    np.seterr(divide='ignore', invalid='ignore')  # Nans are ignored anyways.
    if do_parallel:
        import multiprocessing
        pool = multiprocessing.Pool(14)
        wc = pool.map(mean_corr_coeff, w)
    else:
        wc = list(map(mean_corr_coeff, w))

    wc = np.array(wc)
    return wc, w_ind




def plot_average_visual_response(trial_calcium, grating_onsets,
                                 chosen=None, scale=2.5, dt=1 / 34, ax=None):
    """
    :param trial_calcium: traces in shape (nCells, nTrialLength, nTrials)
    :param grating_onsets: frames where gratings started
    :param chosen: neuron indices to plot
    :param scale: distance between neurons (which are normalized--not df)
    """
    text_offset = 1.5
    nTrials = np.shape(trial_calcium)[2]
    nGratings = int(len(grating_onsets) / nTrials)
    dirs = np.arange(0, 360, 360 / nGratings, dtype=int)
    average_trial_calcium = np.mean(trial_calcium, 2)
    time = np.linspace(0, dt * np.shape(trial_calcium)[1],
                       np.shape(trial_calcium)[1])

    if chosen is None:
        chosen = range(np.shape(trial_calcium)[0])

    if ax is None:
        plt.figure(figsize=(8.5, len(chosen)))
        ax = plt.subplot(1, 1, 1)

    # Plot traces for all neurons correlated with the stimulus
    for ii, idx in enumerate(np.flipud(chosen)):
        for jdx in range(nTrials):
            tr = norm_trace(trial_calcium[idx, :, jdx])
            plt.plot(time, tr + scale * ii, color=[.6, .6, .6])
        nt = norm_trace(average_trial_calcium[idx, :]) + scale * ii
        plt.plot(time, nt, color='k')
        plt.text(-5, scale * ii + .8, 'cell', size=15)
        plt.text(-5, scale * ii, str(idx), size=15)

    # Plot trial onset markers
    onsets = time[grating_onsets[:nGratings] - grating_onsets[0]]
    plt.plot([onsets, onsets], [0, scale * len(chosen)], 'r')

    # Plot grating direction at top
    for idx in range(len(dirs)):
        plt.text(time[grating_onsets[idx] - grating_onsets[0]] + text_offset,
                 scale * len(chosen),
                 str(dirs[idx]) + '$^\circ$', size=15)

    # Clean up plot
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # plt.xticks([])
    plt.ylim([0, scale * len(chosen)])
    plt.xticks(fontsize=15)
    plt.xlabel('Time (s)', fontsize=15)
    plt.yticks([])
    plt.box('off')


def plot_cell_across_trial_types(cell, C, footprints, T,
                                 fps, mean_image, atlas_outline,
                                 frame_range, range_mean, range_sem,
                                 trial_sets, names, colors, pvals):
    """
    For a specified cell, plots:
    - the mean trace for each trial_type specified in trial_sets.
    - the mean value during the specified frame range.
    - the location of the cell.
    - for each trial type, the trace for each trial.
    :param cell: int. overall index of cell.
    :param C: [ncells x nt x ntrials]
    :param footprints: [nx x ny x ncells[
    :param T: [nt]
    :param fps: frames per second
    :param mean_image: [nx x ny]
    :param atlas_outline: [nx x ny]
    :param frame_range: list. [start_frame, end_frame]
    :param range_mean: [ncells x ntrial_types]
    :param range_sem: [ncells x ntrial_types]
    :param trial_sets: tuple of bool arrays. (type1, type2, type3, ...)
                       where type1 is a boolean np.array of length ntrials
                       indicating whether each trial is parts of a trial type.
    :param names: tuple of strings. (type1_name, type2_name, ...)
    :param colors: tuple of strings. Color for plotting each trial type.
    :param pvals: [ncells], pvalue from get_trial_type_selective_cells().
    :return:
    """

    fig = plt.figure(figsize=(10, 6))
    gs = []
    gs.append(plt.subplot2grid((2, 4), (0, 0), colspan=1))
    gs.append(plt.subplot2grid((2, 4), (0, 1), colspan=1))
    gs.append(plt.subplot2grid((2, 4), (0, 2), colspan=1))
    gs.append(plt.subplot2grid((2, 4), (1, 0), colspan=1))
    gs.append(plt.subplot2grid((2, 4), (1, 1), colspan=1))
    gs.append(plt.subplot2grid((2, 4), (1, 2), colspan=1))
    gs.append(plt.subplot2grid((2, 4), (1, 3), colspan=1))
    gs.append(plt.subplot2grid((2, 4), (0, 3), colspan=1))

    C_cell = np.squeeze(C[cell, :, :])
    for ind, trials in enumerate(trial_sets):
        # Generate trial averages for specified trials and cells.
        traces = C_cell[:, np.where(trials[:-1])[0]]
        trial_means = np.mean(traces, axis=1)
        trial_sems = scipy.stats.sem(traces, axis=1)

        # Plot mean traces for each trial type.
        doshade = True
        plt.subplot(gs[0])
        if doshade:
            plt.fill_between(T, trial_means - trial_sems,
                             trial_means + trial_sems,
                             facecolor=colors[ind], alpha=0.3)
        else:
            plt.plot(T, traces, color=colors[ind], linewidth=0.5, alpha=0.1)
        plt.plot(T, trial_means, color=colors[ind], linewidth=2,
                 label=names[ind])
        plt.axvline(frame_range[0] / fps)
        plt.axvline(frame_range[1] / fps)
        plt.ylabel('Fluorescence')

        # Plot mean across frame range for each trial type.
        plt.subplot(gs[1])
        if range_mean is not None:
            plt.plot(ind, range_mean[cell, ind], color=colors[ind], marker='o')
            plt.errorbar(ind, range_mean[cell, ind], yerr=range_sem[cell, ind],
                         color=colors[ind])
            plt.title("p={:.2E}".format(pvals[cell]))

        # Plot the contour for that cell.
        plt.subplot(gs[2])
        if ind == 0:
            all_means = np.mean(C, axis=2)
            CP = CellPlotter(all_means, all_means, footprints,
                             mean_image, cmap=plt.cm.hsv)
            CP.set_highlighted_neurons(np.array([cell]))
            CP.plot_contours(highlight_neurons=True, ax=gs[2],
                             just_show_highlighted=True,
                             atlas_outline=atlas_outline,
                             contour_linewidth=2)

        # Plot all the trials for that cell for each trial type.
        plt.subplot(gs[ind + 3])
        im = plt.imshow(traces.T, aspect='auto')
        plt.axvline(frame_range[0], color='r')
        plt.axvline(frame_range[1], color='r')
        plt.clim([0, 70])
        plt.title(names[ind])
        if ind == 0:
            plt.ylabel('Trial')
        # if ind == 2:
        #     plt.colorbar()

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.97, 0.15, 0.02, 0.3])
    fig.colorbar(im, cax=cbar_ax, label='Fluorescence')

    plt.subplot(gs[0])
    plt.title(str(cell))
    # plt.legend()


def get_trial_type_selective_cells(C, frame_range,
                                   trial_sets,
                                   maxpval=1e-4,
                                   do_plot_histograms=False,
                                   post_hoc='maxdist'):
    """
    Determines which cells exhibit a statistically different
    response to one of the trial types, as compared with the other types.

    Specifically, the mean trace value within a user-defined
    range of frames is computed for each trial for each cell.
    A non-parametric anova (kruskal-wallis) is then used to determine
    the extent to which different trial types have different mean values.

    For reference: a one-way anova is intended for 'testing mean
    difference between three or more independent groups on a single
    dependent variable.' Here, the independent group is the trial type
    (i.e. spout position), and the dependent variable is fluorescence.

    We use Kruskal-Wallis one-way anova because the data did not appear
    to be normally distributed, even after a log-transform. We can
    assume that each measurement/trial is essentially a random sample
    from that trial-type, although there may be some hysteresis within
    each block of a given trial type.
    :param C: [ncells x ntimepoints x ntrials] traces.
    :param frame_range: list. [start_frame, end_frame]. The range
                        over which to compute the mean trace value
                        for each trial.
    :param trial_sets: tuple of bool arrays. (type1, type2, type3, ...)
                       where type1 is a boolean np.array of length ntrials
                       indicating whether each trial is parts of a trial type.
    :param maxpval: Threshold used for classifying a cell as selective.
    :param do_plot_histograms: bool. Plot the histogram of mean values
                                     within the frame_range, for each
                                     cell for each trial type.
                                     This makes a lot of plots - may
                                     crash python unless only
                                     feed in a subset of cells.
    :param post_hoc: None to perform not post-hoc test (selection[4] will not
                                                        be filled)
                     'dunn': to perform dunn test.
                     'maxdist': find condition with mean that is farthest
                                from the other conditions.
                     'max': find condition with largest activation


    :return: selection: tuple.
                        selection[0] - selective_cells: list of the indices
                                       of the selective cells.
                        selection[1] - range_mean: [ncells x ntrialtypes]
                                                   mean trace value for each
                                                   cell for each trial type
                                                   within the frame_range.
                                                   Can be plotted to verify
                                                   the selective cells.
                        selection[2] - range_sem: [ncells x ntrialtypes]
                                                   standard error of the mean
                                                   for each cell and trial type
                                                   within the frame_range
                        selection[3] - pvals: the p-values used to compute
                                              cell selectivity.
                        selection[4] - most_significant: which of the trial
                                                         conditions is most
                                                         sig different
                                                         than the others ones,
                                                         on average according
                                                         to the post_hoc test.
                        selection[5] - sig_vals: the actual significance value
                                                 of the most sig condition.


    """
    ncells = np.shape(C)[0]
    nsets = len(trial_sets)
    range_mean = np.zeros((ncells, nsets))
    range_sem = np.zeros((ncells, nsets))

    pvals = []
    most_significant = []
    sig_vals = []
    selective_cells = []
    cells = np.arange(0, ncells)
    for cell in cells:
        if do_plot_histograms:
            plt.figure(figsize=(16, nsets))

        all_means = []
        for ind, trials in enumerate(trial_sets):
            # Generate trial averages for specified trials and cells.
            traces = np.squeeze(C[cell, :, :])
            traces = traces[:, np.where(trials[:-1])[0]]

            means = np.mean(traces[frame_range[0]:frame_range[1], :], axis=0)
            all_means.append(means)

            range_mean[cell, ind] = np.mean(means)
            range_sem[cell, ind] = scipy.stats.sem(means)

            if do_plot_histograms:
                plt.subplot('1' + str(nsets) + str(ind))
                plt.title(str(cell) + ': ' + str(np.var(means)))
                plt.hist(np.log(means))

        # Because of the peculiar way scipy.stats.kruskal takes
        # in arguments, we have to do this janky approach to enable
        # flexible numbers of trial_types.
        if nsets == 2:
            h, p = scipy.stats.kruskal(all_means[0], all_means[1])
        elif nsets == 3:
            h, p = scipy.stats.kruskal(
                all_means[0], all_means[1], all_means[2])
        elif nsets == 4:
            h, p = scipy.stats.kruskal(all_means[0], all_means[1],
                                       all_means[2],
                                       all_means[3])
        elif nsets == 5:
            h, p = scipy.stats.kruskal(all_means[0], all_means[1],
                                       all_means[2],
                                       all_means[3], all_means[4])
        elif nsets == 6:
            h, p = scipy.stats.kruskal(all_means[0], all_means[1],
                                       all_means[2], all_means[3],
                                       all_means[4], all_means[5])
        else:
            error('get_trial_type_selective_cells not implemented for more'
                  'than 6 trials')

        pvals.append(p)

        # Use post-hoc dunn test to determine which of
        # the trial conditions is most significantly different
        # from the other conditions.
        if p < maxpval:
            selective_cells.append(cell)
            if post_hoc == 'dunn':
                d = sp.posthoc_dunn(all_means, p_adjust='holm')
                md = np.mean(np.abs(d), axis=1)
                sig_ind = np.argmin(md)
                sig_val = md[sig_ind]
            elif post_hoc == 'maxdist':
                dists = []
                for i in range(range_mean.shape[1]):
                    dists.append(np.sum(np.abs(range_mean[cell, :]
                                               - range_mean[cell, i])))
                sig_ind = np.argmax(np.array(dists))
                sig_val = dists[sig_ind]
            elif post_hoc == 'max':
                sig_ind = np.argmax(range_mean[cell, :])
                sig_val = range_mean[cell, sig_ind]

            most_significant.append(sig_ind)
            sig_vals.append(sig_val)

    pvals = np.array(pvals)
    selective_cells = np.array(selective_cells)
    most_significant = np.array(most_significant)
    sig_vals = np.array(sig_vals)
    selection = (selective_cells, range_mean, range_sem, pvals,
                 most_significant, sig_vals)
    return selection


def compute_average_trial_response(C, trial_onsets,
                                   trial_length=200, baseline_idx=None):
    """
    :param C: raw fluorescence or calcium data
    :param trial_onsets: list of frames where trials started
    :param trial_length: length of each trial in frames to average
    :param baseline_idx: frames to set as F_0 (defaults to mean)
    """
    if baseline_idx is None:
        baseline_idx = np.arange(trial_length)

    avg = np.zeros([np.shape(C)[0], trial_length], dtype='float64')

    for frame in trial_onsets:
        avg += df(C[:, frame:frame + trial_length], baseline_idx)
    avg = avg / len(trial_onsets)

    for idx in np.arange(np.shape(C)[0]):
        avg[idx, :] = avg[idx, :] - np.mean(avg[idx, baseline_idx])
    return avg


def order_cells(traces, ordering='var', trials=None):
    """
    Provided a [cells x time x trials] matrix,
    return an array which contains an index
    for ordering the cells, according to some
    criterion.

    :param traces: [cells x time x trials]
    :param ordering: 'var' - order by variance across trials
                     'peak' - order by peak time across trials
    :param trials: ndarray. If not None, then will compute the ordering
                   only based on this defined subset of trials.
    :return:
    """
    if trials is None:
        trials = np.arange(traces.shape[2])

    trial_means = np.mean(traces[:, :, trials], axis=2)
    trial_sems = scipy.stats.sem(traces[:, :, trials], axis=2)

    if ordering == 'peak':
        inds = np.argmax(trial_means, axis=1)
        ordering = np.argsort(inds)
    elif ordering == 'var':
        # Plot the neurons with the smallest relative standard error.
        cell_scale_factor = (np.max(trial_means, axis=1) -
                             np.percentile(trial_means, 15, axis=1))
        cell_sem = np.median(trial_sems, axis=1)
        ordering = np.argsort(cell_sem / cell_scale_factor)

    return ordering


def plot_variance_shaded_traces(avgs, errs,
                                cell_inds=None,
                                footprints=None,
                                atlas_outline=None,
                                mean_image=None,
                                bpod_data=None,
                                ordering=None,
                                do_normalize=True,
                                title=None,
                                dt=0.034,
                                alpha=None):
    """
    Provided a [cells x frames] array of mean traces and
    corresponding variance across trials,
    plot time series across cells, according to various options.
    :param avgs: [cells x frames] array of mean trace
    :param errs: [cells x frames] array of variance across trials
    :param cell_inds: np.array containing indices of cells to plot
                      (after ordering). i.e. np.arange(0, 40)
    :param footprints: [X x Y x cells] footprints.
                       If not None, will plot contours corresponding
                       to plotted cells.
    :param atlas_outline: Must be provided if footprints is not None.
    :param mean_image: Must be provided if footprint is not None.
    :param bpod_data: (optional) bpod_data struct for
                      marking event times.
    :param ordering: None for no ordering.
                     'peak' for ordering by peak time.
                     'var' for ordering by variance.
                     If an np.array is is provided,
                     then order according to that.
    :param do_normalize: Normalize each trace by max.
    :param title: (optional) Title for the plot.
    :param dt: time of one frame in seconds.
    :param alpha: (optional) np.array. assign an alpha value to each
                  plotted contour.
    :return: ordering - np.array of the ordering used
                        for plotting the raster.

    """

    ncells = np.shape(avgs)[0]
    # nframes = np.shape(avgs)[1]

    if type(ordering) is np.ndarray:
        pass
    elif ordering == 'peak':
        inds = np.argmax(avgs, axis=1)
        ordering = np.argsort(inds)
    elif ordering == 'var':
        # Plot the neurons with the smallest relative standard error.
        cell_scale_factor = (np.max(avgs, axis=1) -
                             np.percentile(avgs, 15, axis=1))
        cell_sem = np.median(errs, axis=1)
        ordering = np.argsort(cell_sem / cell_scale_factor)
    else:
        ordering = np.arange(ncells)

    fig = plt.figure(figsize=(20, 10))
    CP = CellPlotter(avgs, avgs, footprints,
                     mean_image, spikes=None,
                     errbar=errs,
                     date='',
                     name=title,
                     fig_save_path=None,
                     suffix=None,
                     cmap=plt.cm.winter)

    CP.set_highlighted_neurons(ordering[cell_inds])
    CP.neuron_ind_alpha = alpha

    event_frames = (1.0 / dt) * np.array([bpod_data.stimulus_times[0],
                                          bpod_data.stimulus_times[0] + 1.5])
    CP.plot_traces(n_timepoints=avgs.shape[1],
                   ax=plt.subplot(142),
                   # Even though only two plots, using subplot(142)
                   # squeezes the trace plot to improve aesthetics.
                   save_plot=False, \
                   event_frames=event_frames, \
                   dt=dt)

    if (footprints is not None and mean_image is not None and
            atlas_outline is not None):
        CP.plot_contours(highlight_neurons=True,
                         display_numbers=False,
                         ax=plt.subplot(122),
                         atlas_outline=atlas_outline,
                         maxthr=0.8,
                         just_show_highlighted=False,
                         highlight_color=None)

    return ordering


def get_region_labels(cells_in_region, regions, which_regions,
                      hemisphere_of_cell, which_hemispheres,
                      mean_traces=None, ordering=None):
    """
    Order cells by region, potentially ordering by peak within each region.
    :param cells_in_region: dict. keys are id number of each region.
                                  values are indices of cells that
                                  are in that region.
    :param regions: dict. keys are string abbreviation for each region.
                          values are id number for that region (i.e.
                          for indexing into cells_in_region).
    :param which_regions: list. abbreviations of regions to include.
    :param hemisphere_of_cell: binary np.array of length ncells.
                               value indicates which hemisphere
                               a cell is in.
                               Set to None in order to ignore information
                               about hemisphere.
    :param which_hemispheres: array [0, 1], [0], or [1]
    :param mean_traces: np.array with trace for each cell. Must provide this
                        if ordering by peak.
    :param ordering: None, 'peak'. Order the cells within each region by
                     their peak firing time.

    :returns region_ordered_ind: The index of each cell when ordered.
                                 i.e. neurons[region_ordered_ind] will
                                 order them.
    :returns region_labels: The region associated with each ordered cell.
    :returns label_ind: the tick locations for the region labels.
    :returns region_str: the region label strings.
    """

    region_ordered_ind = []
    region_labels = []
    label_ind = []
    region_str = []
    hem_names = ['l', 'r']  # NEED TO CHECK THAT THIS IS CORRECT!
    for ind, region in enumerate(which_regions):
        for hemisphere in which_hemispheres:
            which_cells = np.array(cells_in_region[regions[region]])
            which_cells = which_cells[np.where(hemisphere_of_cell[which_cells]
                                               == hemisphere)[0]]
            start_ind = len(region_ordered_ind)

            if ordering == 'peak':
                if mean_traces is None:
                    raise (
                        'Must provide mean_traces to get_region_labels ' +
                        'to order by peak.')
                inds = np.argmax(mean_traces[which_cells, :], axis=1)
                peak_ordering = np.argsort(inds)
                which_cells = which_cells[peak_ordering]
            region_ordered_ind.extend(which_cells)

            end_ind = len(region_ordered_ind)
            label_ind.extend([(start_ind + end_ind) / 2])
            region_labels.extend([ind + hemisphere * 0.5] * len(which_cells))
            region_str.append(hem_names[hemisphere] + region[:3])
    region_ordered_ind = np.array(region_ordered_ind)
    region_labels = np.expand_dims(np.array(region_labels), axis=1)

    return region_ordered_ind, region_labels, label_ind, region_str


def zscore_flattened(data):
    """
    Flatten a matrix, zscore, and then reshape
    back to the original shape.
    :param data: [ncells x ntime x ntrials]
    :return: zdata: [ncells x ntime x ntrials]
    """
    data_flat = np.reshape(data,
                           (data.shape[0], data.shape[1] * data.shape[2]),
                           order='F')
    zdata_flat = scipy.stats.zscore(data_flat, axis=1)
    zdata = np.reshape(zdata_flat,
                       (data.shape[0], data.shape[1], data.shape[2]),
                       order='F')
    return zdata


def plot_average_by_region(
    traces, cells_in_region, regions, hemisphere_of_cell,
    dt, which_trials,
    nframes=200, startframe=0, event_frames=None,
    which_regions=['MO', 'PTLp', 'RSP', 'SSp', 'VIS'],
    ordering='peak', left_labels=True, right_labels=True,
        titlestr=None, region_cmap='jet', traces_cmap='gray'):
    """
    Average trace for each source across a specified set of trials.

    :param traces: [cells x time x trials]
    :param cells_in_region: dict. keys are id number of each region.
                                  values are indices of cells that
                                  are in that region.
    :param regions: dict. keys are string abbreviation for each region.
                          values are id number for that region (i.e.
                          for indexing into cells_in_region).
    :param hemisphere_of_cell: binary np.array of length ncells.
                               value indicates which hemisphere
                               a cell is in.
                               Set to None in order to ignore information
                               about hemisphere.
    :param dt: float. time of one frame in seconds.
    :param which_trials: array of integers indicating the indices of
           trials to include.
    :param which_regions: list. abbreviations of regions to include.
    :param event_frames: (optional). List of frames at which to plot
                         a vertical line.
    :param ordering: 'peak' orders the averaged traces by their peak time.
                     or, provide an np.array of the ordering to use 
                     (i.e. to use
                     an ordering computed somewhere else.)
    :param left_labels: boolean. Display labels of the regions.
    :param right_labels: boolean. Display cell numbers.
    :param titlestr: string. Supply title of the plot.
    :param region_cmap: string, i.e. 'jet'. colormap to use for regions.
    :param traces_cmap: string. i.e. 'gray'. colormap for plotting the traces.

    :returns ordering: The ordering used for each cell.
                       (i.e. the position of that cell
                       in the ordered array).
             region_labels: the region of each (ordered) cell.
    """

    trial_traces = traces[:, :, which_trials]
    mean_traces = np.mean(trial_traces, axis=2)
    mean_traces = scipy.stats.zscore(mean_traces,
                                     axis=1)  # APPLY Z-SCORE TO TRACES

    if hemisphere_of_cell is None:
        hemisphere_of_cell = np.zeros(traces.shape[0], )
        which_hemispheres = [0]
    else:
        which_hemispheres = [0, 1]

    # Generate the full ordering, by peak, within region
    region_idx, region_labels, label_ind, region_str = get_region_labels(
        cells_in_region,
        regions, which_regions,
        hemisphere_of_cell, which_hemispheres,
        mean_traces=mean_traces, ordering='peak')
    region_ordered_ind = region_idx

    if isinstance(ordering, np.ndarray):
        print('Using provided ordering.')
        region_ordered_ind = ordering

    if event_frames is not None:
        origin_frame = event_frames[1]
    else:
        origin_frame = 0

    # Now, plot.
    fig = plt.figure(figsize=(10, 20))
    gs = []
    gs.append(plt.subplot2grid((50, 50), (0, 0), colspan=3, rowspan=47))
    gs.append(plt.subplot2grid((50, 50), (0, 3), colspan=47, rowspan=47))

    plt.subplot(gs[0])
    plt.imshow(region_labels, aspect='auto', cmap=region_cmap)
    gs[0].get_xaxis().set_visible(False)
    plt.yticks(label_ind, region_str)
    gs[0].spines['left'].set_visible(False)
    gs[0].tick_params(length=0)

    if not left_labels:
        gs[0].set_yticks([])

    plt.subplot(gs[1])
    nframes = np.amin([nframes, traces.shape[1] - startframe])
    cmax = np.percentile(traces[:, startframe:startframe + nframes], 99)
    cmin = 0
    rr = region_ordered_ind
    plt.imshow(mean_traces[rr, startframe:startframe + nframes],
               aspect='auto',
               cmap=traces_cmap,  # 'gray_r'
               extent=[(startframe - origin_frame) * dt,
                       dt * (startframe - origin_frame + nframes),
                       0, traces.shape[0]],
               rasterized=True)
    gs[1].yaxis.tick_right()
    plt.ylabel('cells')
    gs[1].yaxis.set_label_position("right")
    plt.xlabel('time [s]')
    plt.xticks([0, 4])
    #     plt.clim([cmin, cmax])
    plt.clim([0, 1])
    plt.gca().yaxis.labelpad = -1  # Shift ylabel closer in.
    if titlestr is not None:
        plt.title(titlestr)
    else:
        plt.title('clims: {}, {:.2f}'.format(cmin, cmax))

    if not right_labels:
        gs[1].set_yticks([])
        plt.ylabel('')

    if event_frames is not None:
        for frame in event_frames:
            if frame > startframe and frame < startframe + nframes:
                plt.axvline(
                    (frame - origin_frame) * dt, color='m', linewidth=1)

    return region_ordered_ind, region_labels


def plot_raster_by_region(cells_in_region,
                          regions,
                          hemisphere_of_cell,
                          dt,
                          traces,
                          nframes=3000,
                          startframe=0,
                          which_regions=['MO', 'PTLp', 'RSP', 'SSp', 'VIS'],
                          event_frames=None):
    """
    Raster plot (each row is a the trace of a cell) of the specified
    traces matrix, ordered such that cells are grouped by region.

    :param cells_in_region: dict. keys are id number of each region.
                                  values are indices of cells that
                                  are in that region.
    :param regions: dict. keys are string abbreviation for each region.
                          values are id number for that region (i.e.
                          for indexing into cells_in_region).
    :param hemisphere_of_cell: binary np.array of length ncells.
                               value indicates which hemisphere
                               a cell is in.
    :param dt: float. time of one frame in seconds.
    :param traces: np.array. [ncells x nt] matrix to plot.
    :param nframes: int. number of frames to plot.
    :param startframe: int. first frames to plot.
    :param which_regions: list. abbreviations of regions to include.
    :param event_frames: (optional). List of frames at which to plot
                         a vertical line.
    :return: Nothing.
    """

    # Order cells by region.
    region_ordered_ind = []
    region_labels = []
    label_ind = []
    region_str = []
    for ind, region in enumerate(which_regions):
        for hemisphere in [0, 1]:
            which_cells = np.array(cells_in_region[regions[region]])
            which_cells = which_cells[
                np.where(hemisphere_of_cell[which_cells] == hemisphere)[0]]
            start_ind = len(region_ordered_ind)
            region_ordered_ind.extend(np.random.permutation(which_cells))
            # region_ordered_ind.extend(which_cells)
            end_ind = len(region_ordered_ind)
            label_ind.extend([(start_ind + end_ind) / 2])
            region_labels.extend([ind + hemisphere * 0.5] * len(which_cells))
            region_str.append(region + '-' + str(hemisphere))
    region_ordered_ind = np.array(region_ordered_ind)
    region_labels = np.expand_dims(np.array(region_labels), axis=1)

    # Now, plot.
    fig = plt.figure(figsize=(20, 20))
    gs = []
    gs.append(plt.subplot2grid((50, 50), (0, 0), colspan=1, rowspan=49))
    gs.append(plt.subplot2grid((50, 50), (0, 1), colspan=49, rowspan=49))

    plt.subplot(gs[0])
    plt.imshow(region_labels, aspect='auto', cmap='jet')
    gs[0].get_xaxis().set_visible(False)
    plt.yticks(label_ind, region_str)

    plt.subplot(gs[1])
    nframes = np.amin([nframes, traces.shape[1] - startframe])
    cmax = np.percentile(traces[:, startframe:startframe + nframes], 99)
    cmin = 0
    plt.imshow(traces[region_ordered_ind, startframe:startframe + nframes],
               aspect='auto',
               cmap='gray_r',
               extent=[startframe * dt, dt * (startframe + nframes),
                       0, traces.shape[0]])
    gs[1].yaxis.tick_right()
    # gs[1].get_yaxis().set_visible(False)
    plt.ylabel('cells')
    gs[1].yaxis.set_label_position("right")
    plt.xlabel('time [s]')
    plt.clim([cmin, cmax])
    plt.title('clims: {}, {:.2f}'.format(cmin, cmax))
    if event_frames is not None:
        for frame in event_frames:
            if frame > startframe and frame < startframe + nframes:
                plt.axvline(frame * dt, color='c')

def get_lick_onsets(lick_rates):
    """
    Get a list of lick onset frames for each trial.
    :param lick_rates:
    :return:
    """
    summed_licks = sum([lick_rates[i] for i in range(4)])

    # Get lick onsets
    onsets = []
    for trial in range(summed_licks.shape[0]):
        licks = np.where(summed_licks[trial, :] > 0)[0]
        if len(licks) > 0:
            onsets.append(np.min(licks))
        else:
            onsets.append(np.nan)

    return np.array(onsets)

def plot_formatted_cell_across_trials(cell_id, C, T, trial_sets, trial_names,
                                      trial_colors, event_frames,
                                      centroids, atlas_tform, clim=[0, 0.5],
                                      do_merge_go_means=False,
                                      lick_onsets=None,
                                      use_trials=None,
                                      xlims=None,
                                      ylims_trials=None,
                                      ylims_avg=None):
    """
    For a specified cell, plots:
    - the trace across trials, separated by trial type
    - a summary of each trial type
    - the location of the cell.
    This is a modified/cleaned/formatted version of
    plot_cell_across_trial_types().

    :param cell_id: int. overall index of cell.
    :param C: [ncells x nt x ntrials]. The neural data, preferably
              smoothed spikes.
    :param T: [nt], time in seconds of each frame.
    :param trial_sets: tuple of bool arrays. (type1, type2, type3, ...)
                       where type1 is a boolean np.array of length ntrials
                       indicating whether each trial is parts of a trial type.
    :param trial_names: tuple of strings. (type1_name, type2_name, ...)
                        name of each trial type.
    :param trial_colors: tuple of strings. Color for plotting each trial type.
    :param event_frames: list of frames at which events occur,
                         to be marked with vertical line.
    :param centroids: ndarray [total ncells x 2]. The centroid of every
                      cell in the dataset. cell_ids indexes
                      into this array.
    :param atlas_tform: sklearn transform function that was generated
                        based on manually selected keypoints.
                        See CosmosTraces class for more information
                        about this.
    """

    dt = T[1] - T[0]
    odor_onset = np.round(event_frames[0]).astype(int)

    # fig = plt.figure(figsize=(16, 19))
    fig = plt.figure(figsize=(5, 5))

    plt.suptitle(cell_id)
    gs = []
    if centroids is not None:
        gs.append(plt.subplot2grid((6, 3), (0, 0), colspan=1))
    else:
        gs.append(plt.subplot2grid((6, 3), (0, 0), colspan=0))

    gs.append(plt.subplot2grid((6, 3), (1, 0), colspan=3))
    gs.append(plt.subplot2grid((6, 3), (2, 0), colspan=3))
    gs.append(plt.subplot2grid((6, 3), (3, 0), colspan=3))
    gs.append(plt.subplot2grid((6, 3), (4, 0), colspan=3))
    gs.append(plt.subplot2grid((6, 3), (5, 0), colspan=3))

    C_cell = np.squeeze(C[cell_id, :, :])
    for ind, trials in enumerate(trial_sets):
        # Generate trial averages for specified trials and cells.
        if use_trials is not None:
            trials = np.logical_and(trials, use_trials)


        traces = C_cell[:, np.where(trials[:-1])[0]]
        trial_means = np.mean(traces, axis=1)
        trial_sems = scipy.stats.sem(traces, axis=1)
        # print(traces.shape)

        # Plot all the trials for that cell for each trial type.
        plt.subplot(gs[ind + 1])
        # print([T[0] * dt, T[-1] * dt, 0, traces.shape[0]])

        if not do_merge_go_means:
            im = plt.imshow(traces.T, aspect='auto',
                            extent=[T[0] - T[odor_onset], T[-1] - T[odor_onset],
                                    0, traces.shape[1]], cmap='gray_r')
            plt.axvline(0, color='m', linestyle='--', linewidth=0.5)
            plt.axvline(1.5, color='m', linestyle='--', linewidth=0.5)
            if xlims is None:
                plt.xlim([T[0] - T[odor_onset], T[-1] - T[odor_onset]])
            else:
                plt.xlim(xlims)

        if do_merge_go_means:
            im = plt.imshow(traces.T, aspect='auto',
                            extent=[T[0]-T[odor_onset],T[-1]-T[odor_onset],
                                    0, traces.shape[1]], cmap='gray_r')
            plt.axvline(0, color='m', linestyle='--', linewidth=0.5)
            # plt.axvline(1.5, color='k')
            if xlims is None:
                plt.xlim([T[0]-T[odor_onset], T[-1]-T[odor_onset]])
            else:
                plt.xlim(xlims)
        if ylims_trials is not None:
            plt.ylim(ylims_trials)

        plt.gca().tick_params(length=1)

        if lick_onsets is not None:
            plt.plot(lick_onsets[trials]*dt-T[odor_onset],
                     np.arange(len(lick_onsets[trials]))+0.5, 'ro', markersize=0.5, markeredgewidth=0)


        plt.clim(clim)
        #         plt.colorbar()
        plt.title(trial_names[ind], fontsize=8, loc='right', y=0.8)
        plt.gca().axes.xaxis.set_ticks([])
        plt.gca().axes.xaxis.set_ticklabels([])

        if ind == 0:
            plt.ylabel('Trial')

        # Plot mean traces for each trial type.
        if not do_merge_go_means:
            doshade = False
            plt.subplot(gs[5])
            if doshade:
                plt.fill_between(T-T[odor_onset], trial_means - trial_sems,
                                 trial_means + trial_sems,
                                 facecolor=trial_colors[ind], alpha=0.3)
            # else:
            #     plt.plot(T-T[odor_onset], traces, color=trial_colors[ind], linewidth=0.5,
            #              alpha=0.1)
            # spike_rate = trial_means / (dt)  # Divide by bin size to get a rate in Hz
            spike_rate = trial_means  # Divide by bin size to get a rate in Hz

            plt.plot(T-T[odor_onset], spike_rate, color=trial_colors[ind], linewidth=0.5,
                     label=trial_names[ind])
            plt.axvline(0, color='m', linestyle='--', linewidth=0.5)
            plt.axvline(1.5, color='m', linestyle='--', linewidth=0.5)
            # plt.axvline(event_frames[0] * dt, color='m', linestyle='--', linewidth=0.5)
            # plt.axvline(event_frames[1] * dt, color='m', linestyle='--', linewidth=0.5)
            #         plt.ylabel('Fluorescence')
            plt.ylabel('Spike \n rate [Hz]')
            if xlims is None:
                plt.xlim([0, 6])
            else:
                plt.xlim(xlims)
            plt.xlabel('Time (s)')
            # plt.gca().axes.xaxis.set_ticks(
            #     [event_frames[0] * dt, event_frames[0] * dt + 2])
            plt.gca().axes.xaxis.set_ticks(
                [0, 2])
            plt.gca().axes.xaxis.set_ticklabels([0, 2])

    if do_merge_go_means:
        plt.subplot(gs[5])
        go_trials = np.logical_or.reduce(
            (trial_sets[0], trial_sets[1], trial_sets[2]))
        nogo_trials = trial_sets[-1]

        go_traces = np.mean(C_cell[:,go_trials], axis=1)
        nogo_traces = np.mean(C_cell[:,nogo_trials], axis=1)

        # plt.plot(T-T[odor_onset], go_traces/dt, color='k', linewidth=0.5)  # Divide by bin size to get a rate in Hz
        # plt.plot(T-T[odor_onset], nogo_traces/dt, color='g', linewidth=0.5)  # Divide by bin size to get a rate in Hz
        plt.plot(T-T[odor_onset], go_traces, color='k', linewidth=0.5)
        plt.plot(T-T[odor_onset], nogo_traces, color='g', linewidth=0.5)

        plt.axvline(0, color='m', linestyle='--', linewidth=0.5)
        # plt.axvline(1.5, color='m')
        if xlims is None:
            plt.xlim([T[0]-T[odor_onset], T[-1]-T[odor_onset]])
        else:
            plt.xlim(xlims)
            plt.xticks([-2, 0, 2, 4])
    if ylims_avg is not None:
        plt.ylim(ylims_avg)
        plt.yticks(ylims_avg)
    plt.gca().tick_params(length=1)


    fig.subplots_adjust(hspace=0.55, wspace=-.2,
                        bottom=0.1, top=1, right=1,
                        left=0.2)  # Control padding around subplots

    # Plot atlas with location of cell
    if centroids is not None:
        plt.subplot(gs[0])
        plt.axis('off')
        a = plt.axes([.235, .76, .3, .3], facecolor='y')
        centroids_on_atlas(np.array([1]), np.array([cell_id]), centroids, atlas_tform)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.gca().axes.xaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axis('off')
        plt.imshow([[0]])


def plot_average_trial_raster(avgs,
                              bpod_data=None,
                              ordering=None,
                              do_normalize=True,
                              title=None,
                              dt=0.034,
                              ax=None):
    """
    Provided a [cells x frames] array of mean traces,
    plot a raster across cells, according to various options.
    :param avgs: [cells x frames] array
    :param bpod_data:  (optional) bpod_data struct for
                       marking event times.
    :param ordering: None for no ordering.
                     'peak' for ordering by peak time.
                     If an np.array is is provided,
                     then order according to that.
    :param do_normalize: Normalize each trace by max.
    :param title: (optional) Title for the plot.
    :param dt: time of one frame in seconds.
    :param ax: (optional) axes to plot to.
    :return: ordering - np.array of the ordering used
                        for plotting the raster.
    """

    ncells = np.shape(avgs)[0]
    nframes = np.shape(avgs)[1]

    if ax is None:
        ax = plt.figure().gca()
    plt.sca(ax)

    if do_normalize:
        avgs = avgs / np.expand_dims(np.amax(avgs, axis=1), axis=1)

    if type(ordering) is np.ndarray:
        pass
    elif ordering == 'peak':
        inds = np.argmax(avgs, axis=1)
        ordering = np.argsort(inds)
    else:
        ordering = np.arange(ncells)

    x = plt.imshow(avgs[ordering, :], aspect='auto',
                   extent=[0, dt * nframes, 0, ncells])

    plt.ylabel('cell #')
    plt.xlabel('time [s]')

    if bpod_data is not None:
        avg_reward = np.nanmedian(bpod_data.reward_times)  # in seconds.
        avg_stim = np.nanmedian(bpod_data.stimulus_times)  # in seconds.
        plt.axvline(avg_stim, color='r')
        plt.axvline(avg_stim + 1, color='r')
        plt.axvline(avg_reward, color='b')

    plt.colorbar()
    plt.title(title)
    return ordering


def get_population_difference(traces, trials1, trials2):
    """
    Computes the vector 2-norm of the difference between
    the average trace corresponding to two trial types.

    :param trials1: np.array. indices of trials for trial type 1.
    :param trials2: np.array. indicies of trials for trial type 2.
    :param traces: np.array. [neurons x time x trials]
    """
    trials1 = trials1[np.where(trials1 < traces.shape[2])]
    trials2 = trials2[np.where(trials2 < traces.shape[2])]

    # step 1: for each condition, average together all trials.
    traces1 = traces[:, :, trials1]
    traces2 = traces[:, :, trials2]

    avg1 = np.mean(traces1, axis=2)
    avg2 = np.mean(traces2, axis=2)

    # step 2: subtract these matrices for different conditions.
    diffmat = avg1 - avg2

    # step 3: take the vector 2-norm of each time point (across cells).
    diff = np.sum(diffmat * diffmat, axis=0)

    return diff


def get_significant_population_difference(traces, trials1, trials2,
                                          nbootstrap=101, ax=None,
                                          t=None, event_times=None):
    """
    Computes the vector 2-norm across cells at each time point
    of the difference between the average trace corresponding
    to two trial types.
    Additionally computes a bootstrap threshold for
    non-parametric statistical significance of the difference.
    Based on Stavisky et al, 2017, Journal of Neuroscience.

    :param trials1: np.array. indices of trials for trial type 1.
    :param trials2: np.array. indicies of trials for trial type 2.
    :param traces: np.array. [neurons x time x trials]
    :param nbootstrap: int. number of bootstrap repetitions to perform.
                       if 101 then any value greater than max across bootstraps
                       has a significance of p<0.01.
                       if 1001 then it has p<0.001.
                       Note, this takes a fairly long time to run.
    :param ax: figure axes. If not None, then will plot results.
    :param t: time array, for x-axis when plotting.
    :param event_times: list. for plotting vertical lines at events.
    :returns out: a tuple with:
                    diff - time series of difference magnitude vs time.
                    diff_bootstrap - time series for each bootstrap shuffle.
    """
    random.seed(10)
    diff = get_population_difference(traces, trials1, trials2)

    n1 = len(trials1)
    n2 = len(trials2)
    alltrials = np.hstack((trials1, trials2))
    diff_bootstrap = np.zeros((nbootstrap, traces.shape[1]))
    for i in range(nbootstrap):
        if np.mod(i, 100) == 0:
            print(str(i) + ', ', end="")
        b = random.sample(range(len(alltrials)), n1)
        bb = np.ones(len(alltrials))
        bb[b] = 0
        b1 = alltrials[np.where(bb == 0)]
        b2 = alltrials[np.where(bb == 1)]
        diff_b = get_population_difference(traces, b1, b2)
        diff_bootstrap[i, :] = diff_b

    if ax is not None:
        plt.sca(ax)
        plt.plot(t, diff, 'b')
        plt.plot(t, np.mean(diff_bootstrap, axis=0), 'r')
        plt.plot(t, np.percentile(diff_bootstrap, 99, axis=0), 'm')
        plt.plot(t, np.max(diff_bootstrap, axis=0), 'y')
        for et in event_times:
            plt.axvline(et, color='k')
            plt.axvline(et, color='k')

    return (diff, diff_bootstrap)


def average_within_trial_types(traces, trial_sets, use_median=False):
    """
    For each neuron, averages all trials within
    a given trial type.

    :param traces: [neurons x time x trials]
    :param trial_sets: tuple of np.arrays where each
                       array contains index of trials
                       of a certain type.
    :returns trial_type_avgs: [neurons x time x trialtypes]
                              array of average trace.

    """

    trial_type_avgs = np.zeros((traces.shape[0], traces.shape[1],
                                len(trial_sets)))
    for ind, trials in enumerate(trial_sets):
        trials = trials[:-2]
        condition_traces = traces[:, :, np.where(trials)[0]]
        if use_median:
            trial_type_avg = np.median(condition_traces, axis=2)
        else:
            trial_type_avg = np.mean(condition_traces, axis=2)
        trial_type_avgs[:, :, ind] = trial_type_avg

    return trial_type_avgs


def get_coding_plane(traces, trial_sets, frame_range,
                     do_orthogonalize=True,
                     do_plot=False):
    """
    Computes two neural dimensions that define a plane
    along which three trial types are most separated.
    One can then project neural data onto this plane
    for visualization.


    These vectors should lie in a plane
    (which should be the same plane containing the 3 points).
    This plane should be defined by two directions,
    which are vectors with nneurons entries that can
    be dot-producted with each timepoint across neurons.

    :param traces: [neurons x time x trials] traces
                   used for computing coding direction.
    :param trial_sets: tuple containing three lists,
                       each containing the trial indices
                       into traces that define each trial type.
    :param frame_range: Temporal frame range used for computing
                        a mean value that defines the coordinate
                        of each neuron under each trial type.
    :param do_plot: bool. Plot projection of each trial type trace onto
                          the corresponding mode.
    :returns w1, w2: [neurons x 1], the two vectors defining
                     the coordinates along the coding plane.
                     Projection of a timepoint is achieved by dot
                     product with these basis vectors.
    """

    # For a specified time period, compute average response
    # of each neuron for each condition -- 3 vectors
    #     x_condition = np.zeros((traces.shape[0], len(trial_sets)))

    condition_avgs = average_within_trial_types(traces, trial_sets)
    x_condition = np.mean(condition_avgs[:, frame_range[0]:frame_range[1], :],
                          axis=1)

    # Compute coding direction for each of the 3 pairs.
    w1 = np.expand_dims(x_condition[:, 0] - x_condition[:, 1], axis=1)
    w2 = np.expand_dims(x_condition[:, 0] - x_condition[:, 2], axis=1)
    w3 = np.expand_dims(x_condition[:, 2] - x_condition[:, 1],
                        axis=1)  # Can compute with w1 and w2

    # Plot each condition along these vectors
    if do_plot:
        plt.figure(figsize=(14, 4))
        plt.subplot('131')
        plt.plot(np.matmul(condition_avgs[:, :, 0].T, w1))
        plt.plot(np.matmul(condition_avgs[:, :, 1].T, w1), 'r')

        plt.subplot('132')
        plt.plot(np.matmul(condition_avgs[:, :, 0].T, w2))
        plt.plot(np.matmul(condition_avgs[:, :, 2].T, w2), 'r')

        plt.subplot('133')
        plt.plot(np.matmul(condition_avgs[:, :, 1].T, w3))
        plt.plot(np.matmul(condition_avgs[:, :, 2].T, w3), 'r')

    basis = np.hstack((w1, w2))
    if do_orthogonalize:
        basis = scipy.linalg.orth(basis)

    return basis


def plot_2D_trajectories(trajectories, cmaps, event_frames):
    """
    Plot trajectories defined by 2D coordinates for each
    timepoint/frame.

    :param trajectories: tuple of [time x 2] np.array
    :param cmaps: tuple of colormap names
    :param event_frames: indices of frames to mark on trajectories
    """
    # t = np.linspace(0, 10, trajectories[0].shape[0])
    t = np.linspace(0, 10, 200)
    for ind, tt in enumerate(trajectories):
        x = tt[:, 0]
        y = tt[:, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if ind == 0:
            xmin = np.amin(x)
            xmax = np.amax(x)
            ymin = np.amin(y)
            ymax = np.amax(y)
        else:
            xmin = min(np.amin(x), xmin)
            xmax = max(np.amax(x), xmax)
            ymin = min(np.amin(y), ymin)
            ymax = max(np.amax(y), ymax)

        lc = LineCollection(segments, cmap=plt.get_cmap(cmaps[ind]),
                            norm=plt.Normalize(0, 10))
        lc.set_array(t)
        lc.set_linewidth(2)

        plt.gca().add_collection(lc)

        cc = ['k', 'r', 'm', 'b']
        for color, ef in zip(cc, event_frames.astype('int')):
            plt.plot(x[ef], y[ef], 'o', color=color)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


def explained_variance_of_basis(traces, basis, do_orthonormalize=True):
    """
    Computes how much variance is explained by projecting
    onto a provided basis. The basis can have multiple basis
    vectors, that are each of length number of neurons,
    where traces contains a time series for each neuron.

    :param traces: [neurons x samples] np array (samples can be concatenated
                    time series for each trial type).
                    Can be, for example, single trial traces, or the average
                    trace for each neuron.
    :param basis: [neurons x ndirections] np array coefficients for projection
                    onto one basis direction.
    :param do_orthonormalize: bool. If True then orthogonalizes the basis set
                             using q-r decomposition.

    :returns explained_variance_ratio: the explained variance of each
                                       of the provided basis vectors.
                                       The total explained variance
                                       of the basis is
                                       np.sum(explained_variance_ratio).
                                       # Note: previously, this function
                                       # just returned the sum, instead
                                       # of the individual variance explained
                                       # by each basis. This change may
                                       # cause bugs.
    """
    import pdb
    # pdb.set_trace()
    if do_orthonormalize:
        basis, _ = np.linalg.qr(basis)
    else:
        basis = basis / (np.sqrt(np.sum(basis * basis, axis=0)) + 1e-15)
    if traces.shape[0] == 1:
        total_variance = np.var(traces)
    else:
        total_variance = np.trace(np.cov(traces))

    projection = np.matmul(traces.T, basis)

    projection_variance = np.cov(projection.T)

    if len(projection_variance.shape) > 1:
        projection_variance = np.diag(projection_variance)
        # projection_variance = np.trace(projection_variance)

    explained_variance_ratio = projection_variance / total_variance
    # indiv_variance_ratio = indiv_variance/total_variance

    return explained_variance_ratio


def predict_lick_direction(traces, shuffle=False,
                           n_components=10, n_folds=5,
                           exclude_last_trials=3, time_idx=None,
                           predict_two_only=False, use_pca=True,
                           decomposition='PCA'):
    """
    Wrapper for fit_lda_model that trains and evaluates a set
    of LDA models to predict active spout position from fluorescence data.
    :param traces: CosmosTraces object
    :param shuffle: bool, shuffle labels as a control?
    :param n_components: how many PCA components to use to make training vecs
    :param n_folds: how many folds to use in performing cross-validation
    :param exclude_last_trials: throw away the last k trials in the dataset
    :param time_idx: use a subset of all data indices?
    :param predict_two_only: predict only left and right, or also forward?
    :param decomposition: use 'PCA', 'NMF', or 'SparsePCA'?
    """
    trace_trials = traces.bd.ntrials - exclude_last_trials

    if time_idx is None:
        time_idx = np.arange(np.shape(traces.Ct)[1])

    # Spout position labels
    labels = traces.bd.spout_positions[:trace_trials]

    # Throw away all trials that were incorrect or not go
    # "not go" means either "explore" trials or "no-go" trials
    go_trials = np.where(traces.bd.trial_types[:trace_trials] == 3)[0]
    success_trials = np.where(traces.bd.success[:trace_trials])[0]
    go_correct_idx = np.intersect1d(go_trials, success_trials)

    # Also only look at trials to the left or right (not middle)
    if predict_two_only:
        lr = np.concatenate([np.where(traces.bd.spout_positions == 1)[0],
                            np.where(traces.bd.spout_positions == 4)[0]])
        go_correct_idx = np.intersect1d(go_correct_idx, lr)
    labels = labels[go_correct_idx]

    # Shuffle control
    if shuffle:
        np.random.seed(666)
        idx = np.random.permutation(len(labels))
        labels = labels[idx]

    # Average each neuron across time to yield x = (neurons, trials)
    x = np.mean(traces.Ct[:, time_idx, :trace_trials], 1)

    # Do PCA/NMF and reshape result to be of size (trials, n_components)
    if decomposition == 'PCA':
        px = PCA(n_components=n_components)
    elif decomposition == 'SparsePCA':
        px = SparsePCA(n_components=n_components, tol=1e-02,
                       n_jobs=4, alpha=0.05)
    elif decomposition == 'NMF':
        px = NMF(n_components=n_components)
    Ct = px.fit_transform(x.T)
    Ct = np.reshape(Ct, (trace_trials, -1))
    Ct = Ct[go_correct_idx, :]
    components = px

    # Fit LDA model
    scores, models = fit_lda_model(Ct, labels, k_folds=n_folds)

    return scores, models, components


def fit_lda_model(trial_data, labels, k_folds=5, seed=11111):
    """
    Using the provided trial data (samples x dimensions)
    and labels (of length samples), fit an LDA model
    and use k fold cross validation.

    :param trial_data: np.ndarray of size trials x dimensions
    :param labels: array of length equal to trials
    :param k_folds: how many models to generate/k-fold validate?
    :param seed: random number generator seed for reproducibility
    """

    # Check if the k_folds parameter is too big
    if len(labels) <= k_folds:
        raise ValueError('len(labels) <= k_folds!')

    # Permute the data (so that we don't sample all from one dataset)
    print(np.shape(labels), np.shape(trial_data))
    if seed:
        np.random.seed(seed)
        idx = np.random.permutation(len(labels))
        labels = labels[idx]
        trial_data = trial_data[idx, :]

    # Figure out places to break data
    batch_size = int(len(labels) / k_folds)
    batch_start = np.arange(0, len(labels), batch_size, dtype='int')
    print('batch size =', batch_size, '  k folds =', k_folds)

    # Train and test a bunch of models
    scores = np.zeros((k_folds, 2))
    models = []
    for k in range(k_folds):

        # Get the next training set and test set
        test_set = np.zeros(len(labels))
        test_set[batch_start[k]:batch_start[k] + batch_size] = 1
        train_set = np.where(test_set == 0)[0]
        test_set = np.where(test_set == 1)[0]

        train_labels = labels[train_set]
        train_data = trial_data[train_set, :]
        test_labels = labels[test_set]
        test_data = trial_data[test_set, :]

        # Make an LDA model
        lda = LDA()
        lda.fit(train_data, train_labels)

        # Predict on training data
        pred_train = lda.predict(train_data)
        train_score = lda.score(train_data, train_labels)

        # Predict on testing data
        pred_test = lda.predict(test_data)
        test_score = lda.score(test_data, test_labels)
        scores[k, :] = [train_score, test_score]
        models.append(lda)
    return scores, models


def get_pca_matrices(trace_matrix, trace_object, normalize=True):
    """
    Return three dicts containing condition matrices to run
    PCA on: dir_matrix broken down by lick direction,
    region_matrix broken down by brain region, and single_trial_matrix
    broken down by lick direction but not averaged across trials.

    :param trace_matrix: Matrix containing calcium or spike observations
    :param trace_object: CosmosTraces object containing all data

    :returns dir_matrix, region_matrix, single_trial_matrix, each are dicts
    """
    tr = trace_matrix  # Matrix containing calcium or spike observations
    traces = trace_object  # CosmosTraces object

    regions = trace_object.populated_areas
    directions = {'right': 1, 'middle': 3, 'left': 4}

    # Get lists of trials and neurons to operate on
    clean = traces.bd.get_clean_trials(min_selectivity=1)
    left_h_idx = np.where(traces.hemisphere_of_cell == 0)[0]
    right_h_idx = np.where(traces.hemisphere_of_cell == 1)[0]

    # Get cells by region
    region_matrix = {}
    all_go = np.mean(tr[:, :, clean], 2).T
    for ind, region in enumerate(regions):
        which_cells = np.array(traces.cells_in_region[traces.regions[region]])
        idx = np.zeros(np.shape(all_go)[1])
        idx[which_cells] = 1
        region_matrix[region] = all_go.copy()
        region_matrix[region][:, np.where(idx == 0)[0]] = 0

        if normalize:
            offset = np.repeat(np.reshape(region_matrix[region][0, :],
                               [-1, 1]), np.shape(region_matrix[region])[0], 1)
            scale = len(
                np.where(np.abs(np.sum(region_matrix[region], 0)) > 0)[0])
            region_matrix[region] = region_matrix[region] - offset.T
            if scale != 0:
                region_matrix[region] = region_matrix[region] * (1 / scale)

    # Get cells by direction
    dir_matrix = {}
    dir_matrix['all'] = np.mean(tr[:, :, :], 2).T
    dir_matrix['go'] = np.mean(tr[:, :, clean], 2).T
    dir_matrix['nogo'] = np.mean(tr[:, :,  # noqa: E712
                                 np.where(traces.bd.go_trials == False)[0]],
                                 2).T

    for direction in directions.keys():
        selected_trials = np.where(
            traces.bd.spout_positions == directions[direction])
        dir_matrix[direction] = np.mean(
            tr[:, :, np.intersect1d(clean, selected_trials)], 2).T
        dir_matrix[direction + '_lefth'] = dir_matrix[direction].copy()
        dir_matrix[direction + '_lefth'][:, right_h_idx] = 0
        dir_matrix[direction + '_righth'] = dir_matrix[direction].copy()
        dir_matrix[direction + '_righth'][:, left_h_idx] = 0

    if normalize:
        for key in dir_matrix.keys():
            scale = len(
                np.where(np.abs(np.sum(dir_matrix[key], 0)) > 0)[0])
            offset = np.repeat(np.reshape(dir_matrix[key][0, :], [-1, 1]),
                               np.shape(dir_matrix[key])[0], 1)
            dir_matrix[key] = dir_matrix[key] - offset.T
            if scale != 0:
                dir_matrix[key] = dir_matrix[key] * (1 / scale)

    # Get cells by direction but don't trial average
    single_trial_matrix = {}

    scale = {}
    scale['go'] = len(clean)
    nogo = np.where(traces.bd.go_trials == False)[0]  # noqa: E712
    scale['nogo'] = len(nogo)
    single_trial_matrix['go'] = tr[:, :, clean]
    single_trial_matrix['nogo'] = tr[:, :, nogo]

    for direction in directions.keys():
        selected_trials = np.intersect1d(np.where(
            traces.bd.spout_positions == directions[direction])[0], clean)
        scale[direction] = len(selected_trials)
        single_trial_matrix[direction] = tr[:, :, selected_trials]

    if normalize:
        for key in single_trial_matrix.keys():
            sh = np.shape(single_trial_matrix[key])
            tc = single_trial_matrix[key]
            offset = np.repeat(np.expand_dims(tc[:, 0, :], 1),
                               np.shape(tc)[1], 1)
            single_trial_matrix[key] = single_trial_matrix[key] - offset
            if scale[key] != 0:
                print(scale[key])
                single_trial_matrix[key] = (
                    single_trial_matrix[key] * (1 / scale[key]))

    return dir_matrix, region_matrix, single_trial_matrix


def update_lines(num, dataLines, lines, markers, points):
    """ This is a helper function for make_trajectory_animation() """
    for idx, (line, data, curr_points) in enumerate(zip(lines,
                                                        dataLines, points)):

        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])

        for pointData, point in zip(markers, curr_points):
            ptr = int(np.round(pointData))
            if ptr > num:
                continue
            point.set_data([[data[0, ptr], data[0, ptr]],
                            [data[1, ptr], data[1, ptr]]])
            point.set_3d_properties([data[2, ptr], data[2, ptr]])

    return lines


def make_trajectory_animation(trajectories, markers,
                              colors_markers, labels, interval=60):
    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d', aspect='auto')

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1],
             label=labels[ii])[0] for ii, dat in enumerate(trajectories)]

    points = [[ax.plot([-100, -100], [-100, -100], [-100, -100],
              colors_markers[ii] + '.')[0]
              for ii, pt in enumerate(markers)] for traj in trajectories]

    ax.legend(labels)

    # Setting the axes properties
    ax.set_xlim3d([np.min(np.array(trajectories)[:, 0, :]),
                   np.max(np.array(trajectories)[:, 0, :])])
    ax.set_ylim3d([np.min(np.array(trajectories)[:, 1, :]),
                   np.max(np.array(trajectories)[:, 1, :])])
    ax.set_zlim3d([np.min(np.array(trajectories)[:, 2, :]),
                   np.max(np.array(trajectories)[:, 2, :])])

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines,
                                       np.shape(trajectories)[2],
                                       fargs=(trajectories, lines,
                                              markers, points),
                                       interval=interval, blit=False)

    plt.close()
    rc('animation', html='html5')
    return line_ani


class GratingStimulus():

    """
    Hold and manipulate parameters for a visual grating stimulus

    The default corr_threshold is about the 80-85% percentile correlation value

    :param traces: CosmosTraces object
    """

    def __init__(self, traces, is_cosmos, grating_length=4, blank_length=4,
                 num_gratings=8, num_trials=5, dirs=None, osi_thresh=.65,
                 make_plot=True, corr_thresh=.013, use_filtered='F',
                 first_trial_frames=None, grating_onset_frames=None):

        # We will identify the spatial location of sources for COSMOS data only
        self.is_cosmos = is_cosmos

        if dirs is None:
            dirs = np.arange(0, 360, 45)
        self.dirs = dirs

        self.grating_length = grating_length
        self.blank_length = blank_length
        self.num_gratings = num_gratings
        self.num_trials = num_trials
        self.make_plot = make_plot
        self.traces = traces
        self.basis = None

        # Every self.num_gratings we repeat (until nTrials have passed)
        self.first_trial_frames = first_trial_frames
        if self.first_trial_frames is None:
            self.first_trial_frames = traces.led_frames[::self.num_gratings]

        # All grating onset times
        self.grating_onset_frames = grating_onset_frames
        if self.grating_onset_frames is None:
            self.grating_onset_frames = np.array(traces.led_frames)

        self._get_grating_information()

        # Identify tuned neurons at default thresholds
        self.identify_tuned_neurons(corr_thresh=corr_thresh,
                                    osi_thresh=osi_thresh,
                                    use_filtered=use_filtered)

    def _get_grating_information(self):
        """
        Get fluorescence traces aligned to grating repeats
        """
        traces = self.traces

        # Make a stimulus regressor
        stim = np.zeros(np.shape(traces.F)[1])
        stim_graded = stim.copy()
        stim_graded_no_blank = stim.copy()
        blank_vector = stim.copy()
        grating_frames = int(self.grating_length / traces.dt)
        blank_frames = int(self.blank_length * (1 / self.traces.dt))
        for idx, frame in enumerate(self.grating_onset_frames):
            stim[frame:frame + grating_frames] = 1
            stim_graded[frame:frame + grating_frames] = idx
            stim_graded_no_blank[
                frame:frame + grating_frames + blank_frames] = idx
            blank_start = frame + grating_frames
            blank_end = blank_start + blank_frames
            blank_vector[blank_start:blank_end] = 1
        self.stimulus_vector = stim
        self.blank_vector = blank_vector
        self.stimulus_vector_graded = stim_graded
        self.stimulus_vector_graded_no_blank = stim_graded_no_blank

        # Get trial-triggered fluorescence
        for idx, C_ in enumerate([traces.C, traces.F, traces.S]):
            if idx == 1:
                # Trust the baseline value found by CNMF-E (simplest)
                #
                # Or: Subtract the average fluorescence during the blank period
                # for ii in range(np.shape(C_)[0]):
                #     blank = np.where(self.blank_vector == 1)[0]
                #     C_[ii, :] -= np.mean(C_[ii, blank])
                pass
            if idx == 2:
                # Normalize raw spikes by their max
                # C and F are already baseline centered by CNMF-E
                C_ = gaussian_filter1d(C_, 1.5, axis=1, mode='constant')
                for ii in range(np.shape(C_)[0]):
                    mx = np.max(C_[ii, :])
                    if mx == 0:
                        print('Source', ii, 'max is zero!')
                    else:
                        C_[ii, :] = C_[ii, :] / mx
            n_cells = np.shape(traces.C)[0]
            trial_length = int(
                np.median(np.diff(self.grating_onset_frames)) *
                self.num_gratings)
            average_trial_calcium = np.zeros([np.shape(C_)[0], trial_length])
            trial_calcium = np.zeros(
                [np.shape(C_)[0], trial_length, len(self.first_trial_frames)])
            for trial_idx, trial_frame in enumerate(self.first_trial_frames):
                end_ = trial_frame + trial_length
                offset = 0
                if end_ > np.shape(C_)[1]:
                    offset = end_ - np.shape(C_)[1]
                    end_ = np.shape(C_)[1]
                trial_calcium[:, :trial_length - offset, trial_idx] = (
                    C_[:, trial_frame:end_])
                average_trial_calcium += trial_calcium[:, :, trial_idx]
                trigger_offset = self.num_gratings * trial_idx
                grating_onsets = (self.grating_onset_frames[
                    trigger_offset:trigger_offset + self.num_gratings] -
                    trial_frame)

                if self.make_plot:
                    plt.figure(figsize=(10, 9))
                    ax = plt.subplot(2, 1, 1)
                    plt.imshow(trial_calcium[:, :, trial_idx],
                               aspect='auto', cmap='bone')
                    plt.clim([0, 100])
                    plt.plot([grating_onsets, grating_onsets],
                             [0, n_cells], 'r')
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(idx)
            if idx == 0:
                # Denoised calcium
                self.average_trial_C = average_trial_calcium
                self.trial_C = trial_calcium
            elif idx == 1:
                # Raw fluorescence (baseline removed)
                self.average_trial_F = average_trial_calcium
                self.trial_F = trial_calcium
            else:
                # Smoothed spikes (1)
                self.average_trial_S = average_trial_calcium
                self.trial_S = trial_calcium

    def _compute_neuron_selectivity(self, mean_trial_response,
                                    osi_thresh=0.65, dsi_thresh=0.65):

        # Grating steps to get to 90/180 degrees
        steps_orth = int(90 / np.median(np.diff(self.dirs)))
        steps_null = int(180 / np.median(np.diff(self.dirs)))

        # Compute OSI/DSI for each source
        self.osis = []
        self.dsis = []
        for ii in range(self.traces.ncells):
            # Get the current cell
            mc = mean_trial_response[ii, :].copy()

            # Set negative response bins to zero
            # mc[np.where(mc < 0)[0]] = 0

            pref_idx = np.argmax(mc)

            # Compute OSI/DSI
            r_pref = mc[pref_idx]
            r_orth = mc[pref_idx - steps_orth]
            r_null = mc[pref_idx - steps_null]
            d = r_pref + r_orth
            osi = (r_pref - r_orth) / d if d != 0 else 0
            d = r_pref + r_null
            dsi = (r_pref - r_null) / d if d != 0 else 0

            # Constrain statistic values to be between 0 and 1
            osi = osi if osi >= 0 and osi <= 1 else int(bool(osi))
            dsi = dsi if dsi >= 0 and dsi <= 1 else int(bool(dsi))

            # If the max of the mean response is negative, forget this cell
            if np.max(mc) < 0:
                osi = 0
                dsi = 0

            self.osis.append(osi)
            self.dsis.append(dsi)
        self.osis = np.array(self.osis)
        self.dsis = np.array(self.dsis)

        # Indices of sources with OSI/DSI > thresh
        self.osi_idx = np.where(self.osis > osi_thresh)[0]
        self.dsi_idx = np.where(self.dsis > dsi_thresh)[0]

    def _identify_sources(self, corr_thresh=.013, p_cutoff=0.01):

        # Correlate stimulus regressor with each source
        stim = self.stimulus_vector
        corr = np.array([np.corrcoef(self.traces.C[cell, :], stim)[0, 1]
                        for cell in np.arange(np.shape(self.traces.C)[0])])

        # For sources with invalid corr values (because C = 0, for all T)
        # set corr to zero
        bad_idx = np.where(np.isnan(corr))[0]
        for idx in bad_idx:
            print('Resetting corr to zero for zero source', idx)
            corr[idx] = 0
        self.corr = corr

        # Save these thresholds on correlations
        if corr_thresh is None:
            self.corr_thresh = np.percentile(corr**2, 80)
        else:
            self.corr_thresh = corr_thresh

        # Correlation cutoff
        self.chosen_corr = np.where(self.corr**2 > self.corr_thresh)[0]
        self.chosen_all = np.intersect1d(self.osi_idx, self.chosen_corr)
        self.chosen_on = np.intersect1d(self.chosen_all,
                                        np.where(self.corr > 0)[0])

        # ANOVA tuning
        self.chosen_anova = []
        for idx in range(self.traces.ncells):
            all_trials = [self.mean_strength_trial[idx, grating, :]
                          for grating in range(self.num_gratings)]
            bk = self.mean_strength_blank_trial
            all_blanks = np.concatenate([bk[idx, grating, :]
                                         for grating in range(
                                             self.num_gratings)])
            all_trials.append(all_blanks)
            f, p = stats.f_oneway(*all_trials)
            if p < p_cutoff:
                self.chosen_anova.append(idx)
        self.chosen_anova = np.array(self.chosen_anova)
        self.chosen_on_anova = np.intersect1d(self.osi_idx, self.chosen_anova)

        # Find sources in visual areas (not applicable to 2P data)
        if self.is_cosmos:
            traces = self.traces
            self.left_v1 = []  # All sources in the LEFT PRIMARY visual area
            self.right_v1 = []  # All sources in the RIGHT PRIMARY visual area
            self.left_vis_all = []  # All sources in any LEFT visual area
            self.right_vis_all = []  # All sources in any RIGHT visual area
            self.both_vis_all = []  # All sources in any visual area
            for area in traces.populated_areas:
                idx = traces.cells_in_region[traces.regions[area]]
                for ind in idx:
                    if 'VIS' in area:
                        self.both_vis_all.append(ind)
                        if traces.hemisphere_of_cell[ind] == 0:
                            self.left_vis_all.append(ind)
                            if 'p1' in area:
                                self.left_v1.append(ind)
                        else:
                            self.right_vis_all.append(ind)
                            if 'p1' in area:
                                self.right_v1.append(ind)

    def _find_basis_vectors(self, use_average=True, use_pca=True, pls_dim=5):
        """
        Given traces (either average or single trial) find basis vectors
        for plotting neural trajectories (either using PCA or PLSRegression)

        :param use_average: Use trial averaged data for finding the basis
        :param use_pca: Use PCA or PLSRegression
        """

        # This is for fitting the basis vectors
        if use_average:
            data = self.trial_averaged_ori.T
            regressor = self.grating_ori
        else:
            trial = 0
            data = self.single_trial_ori[:, :, trial].T
            regressor = self.grating_ori

        sps = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]
        gof = self.grating_onset_frames[:self.num_gratings]
        color = sns.color_palette(sns.color_palette('hls', 4), 8)
        plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

        if use_pca:
            self.basis = PCA()
            self.basis.fit(data)
            proj = self.basis.transform(data)
        else:
            self.basis = PLSRegression(pls_dim)
            self.basis.fit(data, regressor)
            proj = self.basis.transform(data)

        plt.figure(figsize=(10, 5))
        for sp in range(len(sps)):
            plt.subplot(2, 3, sp+1, aspect='equal')
            for start, end in zip(self.grating_start, self.grating_end):
                plt.plot(proj[start:end, sps[sp][0]],
                         proj[start:end, sps[sp][1]])
        plt.tight_layout()

    def _plot_neural_trajectory_data(self):
        plt.figure(figsize=(3, 3))
        ax = plt.subplot(111)
        trial_averaged_norm = self.trial_averaged_ori.copy()
        trial_averaged_norm = trial_averaged_norm[
            np.argsort(np.argmax(trial_averaged_norm, 1)), :]
        trial_averaged_norm -= np.min(trial_averaged_norm, 1)[:, None]
        trial_averaged_norm /= np.max(trial_averaged_norm, 1)[:, None]
        plt.imshow(trial_averaged_norm, aspect='auto', cmap='gray')
        for onset in np.concatenate([[0], np.where(
                np.diff(self.grating_ori) > 0)[0]]):
            plt.axvline(onset, color='r')
        sns.despine()
        ax.set_xticklabels(np.round(ax.get_xticks() * (self.traces.dt)))
        plt.xlabel('Time (s)')
        plt.ylabel('Sources')
        plt.savefig('plots/ori-avg-visual-iscosmos-' +
                    str(self.is_cosmos) + '-' + self.traces.name + '.pdf')

    def _get_average_trajectories(self):
        traj = []
        proj = self.basis.transform(self.trial_averaged_ori.T)
        for ix, [start, end] in enumerate(
                zip(self.grating_start, self.grating_end)):
            traj.append(proj[start:end, :])
        return traj

    def compute_angles_between_trajectories(self):
        """ Measure the pairwise angle between trial avg trajectories. """

        if not self.basis:
            raise ValueError('Run compute_neural_trajectories() first!')
        traj = self._get_average_trajectories()

        def angle_between(v1, v2):
            """ Compute the angle between two vectors (in degrees). """
            def unit_vector(vector):
                return vector / np.linalg.norm(vector)
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.degrees(
                np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

        # This is for the average data
        median_angles = []
        pairwise_medians = []
        for i in range(4):
            for j in range(4):
                if i > j:
                    angles = [
                        angle_between(traj[i][x, :], traj[j][x, :])
                        for x in range(np.shape(traj[0])[0])]
                    pm = np.median(angles)
                    print(i, j, pm)
                    median_angles.append(angles)
                    pairwise_medians.append(pm)

        sns.violinplot(np.concatenate(median_angles), orient='vertical')
        plt.axhline(90, color='k')
        _ = plt.ylim([0, 180])
        return pairwise_medians

    def plot_neural_trajectories(self):
        if not self.basis:
            raise ValueError('Run compute_neural_trajectories() first!')
        for single_trial in [False, True]:

            # Scales for drawing reference axes onto the plot
            if self.is_cosmos:
                if single_trial:
                    k = 10
                    s = 3
                else:
                    k = 60
                    s = 3
            else:
                if single_trial:
                    k = 1e4
                    s = 3
                else:
                    k = 6e4
                    s = 3

            # Set up a plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colors = sns.color_palette(sns.color_palette('hls', 4), 8)

            # Plot each trajectory
            if single_trial:
                traj_st = defaultdict(list)
                for trial in range(self.num_trials):
                    proj = self.basis.transform(
                                self.single_trial_ori[:, :, trial].T)
                    for ix, [start, end] in enumerate(
                            zip(self.grating_start, self.grating_end)):
                        traj_st[ix].append(proj[start:end, :])
                        plt.plot(proj[start:end, 1],
                                 proj[start:end, 2],
                                 proj[start:end, 3], color=colors[ix])
            else:
                traj = self._get_average_trajectories()
                proj = self.basis.transform(self.trial_averaged_ori.T)
                for ix, [start, end] in enumerate(
                        zip(self.grating_start, self.grating_end)):
                    plt.plot(proj[start:end, 1],
                             proj[start:end, 2],
                             proj[start:end, 3], color=colors[ix])

            # Plot reference axes
            plt.plot(np.array([0, -k])-k*s,
                     np.array([0, 0])-k*s*2,
                     np.array([0, 0])-k*s, 'k')
            plt.plot(np.array([0, 0])-k*s,
                     np.array([0, -k])-k*s*2,
                     np.array([0, 0])-k*s, 'k')
            plt.plot(np.array([0, 0])-k*s,
                     np.array([0, 0])-k*s*2,
                     np.array([0, -k])-k*s, 'k')

            # Dont show grid axes
            ax.set_axis_off()
            if self.is_cosmos:
                if self.traces.name == 'cux2ai148m943_visual_stim_1':
                    ax.view_init(50, -50)
                else:
                    ax.view_init(70, -40)
            else:
                ax.view_init(45, 100)
            plt.savefig('plots/trajectories-' + self.traces.name +
                        '-iscosmos-' + str(self.is_cosmos) +
                        '-single-trial-' +
                        str(single_trial) + '.pdf')

    def generate_neural_trajectories(self):

        # Choose the data to process
        if self.is_cosmos:
            # For 1P data -- use visual responsive neurons from visual areas
            idx_chosen = np.intersect1d(self.chosen_anova, self.both_vis_all)
        else:
            # For 2P data -- use visual responsive neurons
            idx_chosen = self.chosen_anova

        # Get traces to do dimensionality reduction on
        trial_averaged = self.average_trial_C[idx_chosen, :]
        single_trial = self.trial_C[idx_chosen, :, :]

        # Get regression vector for supervised techniques
        dt = np.where(self.stimulus_vector_graded_no_blank)[0][0]
        grating = self.stimulus_vector_graded_no_blank[
            dt:trial_averaged.shape[1]+dt]

        # The regressor is equal to 5 at the beginning of the 5th grating
        g_idx = 5

        # Get out the data during the first four visual gratings
        if len(self.dirs) == 8:
            first_half_length = np.where(grating == g_idx)[0][0]
            self.grating_ori = grating[:first_half_length]
            self.grating_ori[np.where(self.grating_ori < 0)[0]] = 0
            self.trial_averaged_ori = trial_averaged[:, :first_half_length]
            self.single_trial_ori = single_trial[:, :first_half_length]
        elif len(self.dirs) == 4:
            self.grating_ori = grating
            self.grating_ori[np.where(self.grating_ori < 0)[0]] = 0
            self.trial_averaged_ori = trial_averaged
            self.single_trial_ori = single_trial
        else:
            raise ValueError('Invalid number of gratings (not 4 or 8)!')

        # Get grating onsets
        self.grating_start = np.concatenate(
            [[0], np.where(np.diff(self.grating_ori) > 0)[0]])
        self.grating_end = np.concatenate(
            [np.where(np.diff(self.grating_ori) < 0)[0]])

        # Cull duplicate grating onsets/offsets
        if len(self.dirs) == 8:
            self.grating_start = np.delete(self.grating_start,
                                           np.where(np.diff(
                                            self.grating_start) < g_idx)[0])
            self.grating_end = np.delete(self.grating_end,
                                         np.where(np.diff(
                                            self.grating_end) < g_idx)[0])

        # Plot data used for dimensionality reduction
        self._plot_neural_trajectory_data()

        # Find basis vectors
        self._find_basis_vectors(use_average=True, use_pca=True)

    def identify_tuned_neurons(self, osi_thresh=0.65, dsi_thresh=0.65,
                               corr_thresh=.013, p_cutoff=0.01,
                               use_filtered='F'):
        """
        Use squared correlation to identify sources correlated with the visual
        stimulus and OSI/DSI to identify sources that are strongly tuned.

        self.chosen_all contains a list of all correlated + tuned sources
        self.chosen_corr contains a list of all correlated sources
        self.chosen_anova contains sources tuned given an ANOVA
        self.chosen_on contains all positively correlated + tuned sources
        self.chosen_on_anova contains sources tuned given an ANOVA and OSI

        :param osi_thresh: defaults to 0.65 (used for chosing sources)
        :param dsi_thresh: defaults to 0.65 (not used for chosing sources)
        :param corr_thresh: default is approx 80% (used for chosing sources)
        :param p_cutoff: p-value for ANOVA determining a source is responsive
        :param use_C: use denoised fluorescence (C) or raw fluorescence F?
        """

        # Get a list of grating onset frames for trial averaged data
        grating_frames = self.grating_onset_frames
        vals = np.array(
            [grating_frames[:self.num_gratings] - grating_frames[0]])[0]

        sz = (self.traces.ncells, self.num_gratings, self.num_trials)
        mean_strength = np.zeros(sz)
        std_strength = np.zeros(sz)
        mean_strength_b = np.zeros(sz)
        std_strength_b = np.zeros(sz)

        grating_block_len = int(self.grating_length * (1 / self.traces.dt))
        blank_block_len = int(self.blank_length * (1 / self.traces.dt))

        # Get the fluorescence response to each grating and blank period
        for idx in range(self.traces.ncells):
            for jdx in range(self.num_gratings):
                for trial in range(self.num_trials):
                    if use_filtered == 'C':
                        raw_traces = self.trial_C[idx, :, trial]
                    elif use_filtered == 'F':
                        raw_traces = self.trial_F[idx, :, trial]
                    elif use_filtered == 'S':
                        raw_traces = self.trial_S[idx, :, trial]
                    else:
                        raise ValueError('Wrong argument for use_filtered!')

                    mean_strength[idx, jdx, trial] = np.mean(
                        raw_traces[vals[jdx]:vals[jdx] + grating_block_len])
                    std_strength[idx, jdx, trial] = np.std(
                        raw_traces[vals[jdx]:vals[jdx] + grating_block_len])

                    blank_start = vals[jdx] + grating_block_len
                    mean_strength_b[idx, jdx, trial] = np.mean(
                        raw_traces[blank_start:blank_start + blank_block_len])
                    std_strength_b[idx, jdx, trial] = np.std(
                        raw_traces[blank_start:blank_start + blank_block_len])

        self.mean_strength_trial = mean_strength
        self.mean_strength_blank_trial = mean_strength_b
        self.mean_strength = np.squeeze(np.mean(mean_strength, 2))
        self.std_strength = np.squeeze(np.mean(std_strength, 2))

        # Compute DSI/OSI
        self._compute_neuron_selectivity(np.squeeze(np.mean(mean_strength, 2)),
                                         osi_thresh, dsi_thresh)

        # Compute lists of responsive cells
        self._identify_sources(corr_thresh, p_cutoff)

    def make_active_cell_raster(self, which_idx, start=5000, stop=8000,
                                cells_to_plot=25, spacing=8, new_figure=True):
        """
        Plot all traces denoted in which_idx.
        """
        traces = self.traces
        frames = stop - start
        time = np.linspace(0, frames * traces.dt, frames)  # approximate time

        if new_figure:
            plt.figure(figsize=(13, 13))
        frames = np.intersect1d(np.where(traces.led_frames > start)[0],
                                np.where(traces.led_frames < stop)[0])
        onsets = traces.led_frames[frames] - start

        if cells_to_plot < len(which_idx):
            which_idx = which_idx[:cells_to_plot]

        off = np.arange(len(which_idx))[::-1]

        for off_idx, idx in enumerate(which_idx):

            # Get raw F in Z units
            tr = scipy.stats.zscore(traces.F[idx, start:stop])
            f0 = np.mean(traces.F[idx, start:stop])
            s0 = np.std(traces.F[idx, start:stop])

            # Normalize C to F
            tr2 = traces.C[idx, start:stop]
            tr2 = (tr2 - f0)/s0

            # Plot everything
            to = time[:len(tr)]
            plt.fill_between(to, tr + off[off_idx] * spacing,
                             off[off_idx] * spacing,
                             color='w', zorder=off_idx)
            plt.fill_between(to, tr2 + off[off_idx] * spacing,
                             off[off_idx] * spacing,
                             color='w', zorder=off_idx)
            plt.plot(to, tr + off[off_idx] * spacing,
                     color=[.6, .6, .6], zorder=off_idx)
            plt.plot(to, tr2 + off[off_idx] * spacing, 'k',
                     linewidth=1.5, zorder=off_idx)

        ym = (len(which_idx)) * spacing
        for onset in time[onsets]:
            plt.plot([onset, onset], [-1, ym],
                     color=[1, 0, 0, .3], zorder=cells_to_plot + ym)

        ind = np.where(self.first_trial_frames < stop)[0]
        ind = np.intersect1d(ind, np.where(self.first_trial_frames > start)[0])
        first = time[self.first_trial_frames[ind]]
        plt.plot([first, first], [-1, ym], color=[1, 0, 0, 1],
                 zorder=cells_to_plot + ym + 1)
        plt.axis('off')

    def make_single_trace_plot(self, which_ind, end=9900, title=True):
        """
        Plot a single trace denoted in which_ind.
        """
        plt.figure(figsize=(20, 5))
        stim_idx = np.where(self.stimulus_vector > 0)[0]

        plt.plot([self.first_trial_frames, self.first_trial_frames],
                 [[-.1] * len(self.first_trial_frames),
                 [1] * len(self.first_trial_frames)], 'r')
        FF = norm_trace(self.traces.F[which_ind, :end], 99.9999)
        FF = FF - np.percentile(FF, 30)
        CC = norm_trace(self.traces.C[which_ind, :end], 99.9999)

        plt.plot(FF, color=[.6, .6, .6])
        plt.plot(CC, 'k', linewidth=2)

        plt.plot(np.where(self.stimulus_vector > 0)[0],
                 [-.1] * len(stim_idx), 'r.')
        plt.yticks([])
        plt.xticks([])
        plt.box('off')
        if title:
            plt.title(which_ind)

    def make_linear_plots(self, which_idx, fixed_size=False):
        """
        Make linear tuning plots using phase tuning for specified sources

        This function importantly uses F instead of C

        :param which_idx: list denoting which sources to plot
        """
        if fixed_size:
            plt.figure(figsize=(2, 8))
        else:
            plt.figure(figsize=(7, len(which_idx) * 1.5))
        tt = range(self.num_gratings)
        for ct, idx in enumerate(which_idx):
            ax = plt.subplot(len(which_idx), 1, ct + 1)

            strength = np.array([self.mean_strength[idx, jdx]
                                for jdx in tt])
            std = np.array([self.std_strength[idx, jdx] /
                            np.sqrt(self.num_trials)
                           for jdx in tt])

            # Plot tuning in all directions
            plt.axhline(y=0, color=[.3, .3, .3, .3])
            plt.axhline(y=np.max(strength), color=[.3, .3, .3, .3])
            plt.errorbar(tt, strength, yerr=std, color='k')
            plt.axis('off')
            plt.title('')

            # Plot grating direction at top
            for jdx, dir in enumerate(self.dirs):
                if not fixed_size or ct == 0:
                    plt.text(jdx - .25,
                             np.max(strength) + .25 * np.max(strength),
                             str(dir) + '$^\circ$', size=10)
                if not fixed_size:
                    plt.text(-1, 0.25 * np.max(strength),
                             'cell\n' + str(idx), size=15)

    def make_polar_plots(self, which_idx, plot=True):
        """
        Make polar plots using mean phase tuning for specified sources

        :param which_idx: list denoting which sources to plot
        """
        dirs_r = np.deg2rad(self.dirs)
        sz = int(np.sqrt(len(which_idx))) + 1
        db = np.mean(np.diff(dirs_r))
        if plot:
            plt.figure(figsize=(sz * 3, sz * 2))
        rs = []
        thetas = []
        for ct, idx in enumerate(which_idx):
            strength = np.array([self.mean_strength[idx, jdx]
                                for jdx in range(len(dirs_r))])
            strength[np.where(strength < 0)[0]] = 0
            theta = circ.mean(dirs_r, w=strength, d=db)
            thetas.append(theta)
            r = circ.resultant_vector_length(dirs_r, w=strength, d=db)
            rs.append(r)
            osi = np.round(self.osis[idx], 2)
            dsi = np.round(self.dsis[idx], 2)

            if plot:
                ax = plt.subplot(sz, sz, ct + 1, polar=True)
                ax.set_rorigin(0)
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)

                # Plot tuning in all directions
                plt.polar(np.concatenate([dirs_r, [dirs_r[0]]]),
                          np.concatenate([strength, [strength[0]]]), 'k')

                # Plot mean tuning vector
                plt.polar([0, theta], [0, r * plt.ylim()[1]], 'r')
                ax.yaxis.set_ticklabels([])
                plt.title(str(idx) + ' ' + str(np.around(np.rad2deg(theta))) +
                          '$^\circ$ \n' +
                          ' osi=' + str(osi) +
                          ' dsi=' + str(dsi) + '\n\n', loc='left')

        if plot:
            plt.tight_layout()
        return rs, thetas

    def plot_source_tuning(self, chosen):

        r_, theta_ = self.make_polar_plots(chosen, plot=False)

        # Plot color bar (repeat 4 colors twice)
        colors = sns.color_palette(sns.color_palette('hls', 4), 8)

        sns.palplot(colors, .5)
        plt.axis('off')

        for idx, dir in enumerate(self.dirs):
            tc = 'k' if idx < 4 else 'w'
            plt.text(idx, 0, '$' + str(dir) + '^\circ$', fontsize=10, color=tc,
                     horizontalalignment='center', verticalalignment='center')

        # Get colors to plot
        tuning = [np.argmin(np.abs(self.dirs - np.rad2deg(theta_)[cc]))
                  for cc in range(len(chosen))]
        color_list = np.array([np.concatenate([colors[tt],
                              [r_[tt]]]) for tt in tuning])
        
        # Set alpha to opaque
        color_list[:, 3] = 1

        cp = CellPlotter(self.traces.C, self.traces.F,
                         self.traces.footprints, self.traces.mean_image,
                         date=self.traces.date, name=self.traces.name,
                         fig_save_path=self.traces.fig_save_path,
                         suffix='tuning.pdf')

        cp.set_highlighted_neurons(chosen)
        cp.plot_contours(highlight_neurons=True, edge_color=(0, 0, 0, 0),
                         highlight_color=color_list, maxthr=.7,
                         atlas_outline=self.traces.atlas_outline,
                         just_show_highlighted=True)

        plt.axis('off')
        plt.title('')
        print('all sources = ', np.shape(self.traces.C)[0],
              ' chosen = ', len(chosen))

    def make_correlated_neuron_plot(self, fast_plot=False, corr_or_osi='corr'):

        traces = self.traces
        cp = CellPlotter(traces.C, traces.F, traces.footprints,
                         traces.mean_image,
                         date=traces.date, name=traces.name,
                         fig_save_path=traces.fig_save_path,
                         suffix='postmerge.pdf')
        if corr_or_osi == 'corr':
            # plot corrcoef relative to peak
            highlighted = np.arange(len(self.corr))
            corr_ceil = np.percentile(self.corr**2, 99)
            corr_norm = self.corr**2 / corr_ceil
            corr_norm[np.where(corr_norm > 1)[0]] = 1

            thr = 80 if fast_plot else 0
            min_corr_idx = np.where(
                corr_norm > np.percentile(corr_norm, thr))[0]
            highlighted = highlighted[min_corr_idx]
            corr_norm = corr_norm[min_corr_idx]

            cp.set_highlighted_neurons(highlighted, alpha=corr_norm)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            cp.plot_contours(highlight_neurons=True, edge_color=(1, 0, 0, 0),
                             highlight_color=(1, 0, 0, 1), maxthr=.7, ax=ax,
                             atlas_outline=traces.atlas_outline,
                             just_show_highlighted=True)

            plt.axis('off')
            plt.title('')
            print('all sources = ', np.shape(traces.C)[0],
                  ' chosen = ', len(corr_norm))
            color_label = '$r^2$ between sources and visual stimulus'
        else:
            highlighted = np.arange(len(self.osis))
            corr_norm = self.osis
            corr_ceil = 1

            min_corr_idx = np.where(corr_norm > .8)[0]
            highlighted = highlighted[min_corr_idx]
            corr_norm = corr_norm[min_corr_idx]

            cp.set_highlighted_neurons(highlighted, alpha=corr_norm)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            cp.plot_contours(highlight_neurons=True, edge_color=(1, 0, 0, 0),
                             highlight_color=(1, 0, 0, 1), maxthr=.7, ax=ax,
                             atlas_outline=traces.atlas_outline,
                             just_show_highlighted=True)

            plt.axis('off')
            plt.title('')
            print('all sources = ', np.shape(traces.C)[0],
                  ' chosen = ', len(corr_norm))
            color_label = 'OSI'

        cbar_ax = fig.add_axes([0.95, 0.25, 0.01, .5], frameon=False)

        # Make a colormap to match the red colors we used
        cols = np.zeros((50, 4))
        cols[:, 3] = 1
        cols[:, 0] = np.linspace(0, 1, 50)
        red_map = mpl.colors.ListedColormap(cols, 'red')
        plt.register_cmap(cmap=red_map)

        sm = plt.cm.ScalarMappable(cmap=red_map,
                                   norm=mpl.colors.Normalize(vmin=0,
                                                             vmax=corr_ceil))
        sm.set_array([])
        fig.colorbar(sm, ax=cbar_ax, cax=cbar_ax)
        cbar_ax.set_ylabel(color_label)
        cbar_ax.yaxis.set_label_position('left')

    def make_visual_to_other_area_comparison_plots(self):
        """
        This analysis looks at all 'populated areas' that
        have more than k cells; k is a parameter to CosmosTraces.
        """
        score = '$r^2$'
        summary = pd.DataFrame()

        for area in self.traces.populated_areas:
            idx = self.traces.cells_in_region[self.traces.regions[area]]
            for ind in idx:

                # Get hemisphere
                if self.traces.hemisphere_of_cell[ind] == 0:
                    side = 'Left'
                else:
                    side = 'Right'
                aa = 'Visual' if 'VIS' in area else 'Other'
                d = {'Cortical Region': aa, 'Hemisphere': side,
                     'OSI': self.osis[ind], 'DSI': self.dsis[ind],
                     score: self.corr[ind]**2}
                summary = summary.append(d, ignore_index=True)

        plt.figure(figsize=(2, 2))
        regions = ['Visual', 'Other']
        areas = ['Left', 'Right']
        p_vals = []
        sns.boxplot(data=summary, x='Cortical Region',
                    y='OSI', hue='Hemisphere', order=regions, hue_order=areas)
        plt.grid(True, alpha=0.2)
        plt.xlabel('Area')

        # Print the n of sources in each condition
        for region in ['Visual', 'Other']:
            for side in ['Left', 'Right']:
                print(region, side, 'n =',
                      len(summary.loc[(summary['Cortical Region'] ==
                                       region) &
                                      (summary['Hemisphere'] == side)]))

        # Do pairwise mann whitney tests, and then bonferroni correct them
        # There are six valid comparisons between the four conditions 4*3/2
        nComps = 6
        comp = 0
        print('Bonferroni corrected p values from Mann-Whitney U Test.')
        c1 = []
        c2 = []
        for ii, region1 in enumerate(regions):
            for jj, region2 in enumerate(regions):
                for kk, area1 in enumerate(areas):
                    for ll, area2 in enumerate(areas):
                        if area1 == area2 and region1 == region2:
                            continue
                        if area1 + region1 in c2 and area2 + region2 in c1:
                            continue
                        c1.append(area1 + region1)
                        c2.append(area2 + region2)
                        comp += 1
                        sx = ((summary['Cortical Region'] == region1) &
                              (summary['Hemisphere'] == area1))
                        sy = ((summary['Cortical Region'] == region2) &
                              (summary['Hemisphere'] == area2))
                        x = summary.where(sx)['OSI'].as_matrix()
                        y = summary.where(sy)['OSI'].as_matrix()
                        p = stats.mannwhitneyu(x[np.isfinite(x)],
                                               y[np.isfinite(y)]).pvalue
                        p *= nComps
                        stars = ''
                        if p < 0.0001:
                            stars = '****'
                        elif p < 0.001:
                            stars = '***'
                        elif p < 0.01:
                            stars = '**'
                        elif p < 0.05:
                            stars = '*'
                        print(stars, p, region1, area1, 'vs.', region2, area2)
        if comp != nComps:
            raise ValueError('Unexpected number of comparisons!')
        sns.despine()
        p_vals.append(p)
        plt.tight_layout()
        return summary, p_vals
