from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import warnings
import numpy as np
import time
import os

import cosmos.imaging.atlas_registration as reg
import cosmos.imaging.cell_selection as utils
from cosmos.lib.transparent_imshow.transparent_imshow import transp_imshow


class CellPlotter:

    """A class to make various plots of traces
    and corresponding spatial footprints.
    """

    def __init__(self, traces, traces_raw, footprints, base_image,
                 spikes=None, errbar=None, date=None, name=None,
                 fig_save_path=None, suffix=None, cmap=None):
        self.traces = traces  # [neuron x T]
        self.traces_raw = traces_raw  # [neuron x T]
        self.footprints = footprints  # [X x Y x T]
        self.base_image = base_image  # [X x Y]
        self.spikes = spikes  # [ neuron x T]

        # [neuron x T]. Provide if wish to plot error bar on traces.
        self.errbar = errbar

        # Date of the traces
        self.date = str(date)

        # Name of session for traces
        self.name = str(name)

        # Where to save plots (won't save if None)
        self.fig_save_path = fig_save_path

        # The suffix for all plot save-outs
        self.suffix = suffix

        # i.e. plt.cm.hsv. Color trace/cells according to colormap
        self.cmap = cmap

        # Set this with set_highlighted_neurons.
        self.neuron_ind = np.arange(0, 10)

        # Do not set a unique alpha value for the highlighted neurons
        self.neuron_ind_alpha = None

    def set_highlighted_neurons(self, neuron_ind, alpha=None):
        """
        :param neuron_ind: an np.array with indices of neurons to highlight
                           (i.e. generated using np.arange)
        :param alpha: an np.array of length neuron_ind containing the desired
                      alpha value for each highlighted neuron
        """
        if self.traces is not None:
            if np.max(neuron_ind) > self.traces.shape[0]:
                warnings.warn('Some indices specified for ' +
                              'highlighting are invalid!')

        neuron_ind = neuron_ind[np.where(neuron_ind < self.traces.shape[0])]
        self.neuron_ind = neuron_ind

        self.neuron_ind_alpha = None
        if alpha is not None:
            if len(alpha) == len(neuron_ind):
                self.neuron_ind_alpha = alpha
            else:
                warnings.warn('Ignoring alpha vector ' +
                              'because it is of incorrect length!')

    def plot_traces(self, n_timepoints=2000, ax=None, save_plot=True,
                    event_frames=None, dt=None, do_zscore=True):
        """
        :param save_plot: bool. Save out the plot to self.fig_save_path
        :param ax: optionally provide a plot axis for the figure
        :param n_timepoints: number of frames to plot.
        :param event_frames: array or list of frames to draw vertical lines.
        :param dt: time, in seconds, corresponding to a single frame
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

        if dt is None:
            dt = 1

        for iter, idx in enumerate(self.neuron_ind):
            nt = n_timepoints

            t = dt * np.arange(nt)
            if do_zscore:
                zscale = 4
                scale_factor = np.std(self.traces_raw[idx, :nt])*zscale
                normed_trace = self.traces[idx, :nt]/scale_factor
                normed_raw_trace = self.traces_raw[idx, :nt]/scale_factor
            else:
                normed_trace, scale_factor = self._normalize(
                    self.traces[idx, :nt])
                normed_raw_trace, _ = self._normalize(
                    self.traces_raw[idx, :nt])

            if self.cmap is not None:
                linecolor = 'k'
                facecolor = self.cmap(iter / float(len(self.neuron_ind)))
            else:
                linecolor = 'k'
                facecolor = (0, 0, 1)

            plt.plot(t, normed_raw_trace + iter, color=linecolor, linewidth=1)
            # plt.plot(t, normed_trace + iter, color='r', linewidth=1)
            plt.plot(t, normed_trace + iter, color=facecolor, linewidth=1)

            # plt.plot(t, normed_trace + iter, color=linecolor, linewidth=1)
            if self.errbar is not None:
                plt.fill_between(t, normed_trace + iter -
                                 self.errbar[idx, :nt] / scale_factor,
                                 normed_trace + iter +
                                 self.errbar[idx, :nt] / scale_factor,
                                 facecolor=facecolor, alpha=0.3)
            if self.spikes is not None:
                plt.plot(t, self._normalize(self.spikes[idx, :nt],
                                            percentile=False) + iter, 'r')
        plt.ylim([0, len(self.neuron_ind)])
        ax.get_yaxis().set_ticks(np.arange(len(self.neuron_ind)))
        if do_zscore:
            plt.title('num z-units per row: {}'.format(zscale))

        if event_frames is not None:
            for xc in event_frames:
                if xc < n_timepoints:
                    plt.axvline(
                        x=xc * dt, color=(.3, .3, .3, .8), linestyle='--')

        if self.fig_save_path is not None:
            if self.suffix is not None:
                if save_plot:
                    fig_save_dir = os.path.join(
                        self.fig_save_path, self.date, self.name)
                    if not os.path.isdir(fig_save_dir):
                        os.makedirs(fig_save_dir)
                    plt.savefig(
                        os.path.join(fig_save_dir, 'traces_' + self.suffix))

    def plot_contours(self, highlight_neurons=False,
                      display_numbers=False,
                      ax=None, edge_color=(1, 0, 0, 1),
                      highlight_color=(0, 0, 1, 1),
                      cmap='gray', maxthr=0.8,
                      atlas_outline=None,
                      just_show_highlighted=False,
                      contour_linewidth=1, rotate_image=True,
                      color_list=None,
                      show_footprints=False,
                      no_borders=False):
        """

        :param highlight_neurons: bool. Whether to highlight neurons
                                  that have been specified using
                                  set_highlighted_neurons()
        :param display_numbers: bool. Whether to label contours
                                with the ID number of the neuron.
        :param ax: (optional). Plot to a specific axes.
        :param edge_color: Color of contour outline. i.e. (1, 0, 0 ,1)
        :param highlight_color: Color of filled in contours, i.e. (1, 0, 0, 1)
        :param cmap: the colormap for the background image.
        :param maxthr: float. Threshold for drawing contours.
        :param atlas_outline: array. If provided, then overlays atlas outline.
        :param just_show_highlighted: bool. Just show the highlight neurons.
                                      This is generally significantly faster
                                      at plotting.
        :param contour_linewidth: int. Set the linewidth
        :param color_list: Optionally provide a list of colors for each source.
        :param show_footprints: bool. If True then display the
                                footprint corresponding to each source
                                in addition to the contour of that footprint.
        :return:
        """

        print(show_footprints)
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

        start_time = time.time()

        ff = self.footprints
        fs = ff.shape

        if highlight_neurons or just_show_highlighted:
            hn = self.neuron_ind
        else:
            hn = None

        if self.neuron_ind_alpha is not None and highlight_neurons:
            if color_list is not None:
                chosen_colors = True
            else:
                chosen_colors = False
                color_list = []
            for ind, alpha in enumerate(self.neuron_ind_alpha):
                if chosen_colors:
                    # print('Colors selected by user.')
                    pass
                elif highlight_color is not None:
                    color_curr = np.array(highlight_color, dtype=float)
                    color_curr[3] = alpha
                    color_list.append(color_curr)
                elif self.cmap is not None:
                    color_curr = np.array(self.cmap(ind / float(len(hn))),
                                          dtype=float)
                    color_curr[3] = alpha
                    color_list.append(color_curr)
                else:
                    color_curr = np.array((1, 0, 0, 1))
                    # color_curr = np.array((1, 0.56, 0, 1))
                    color_curr[3] = alpha
                    color_list.append(color_curr)
            highlight_color = color_list
        elif highlight_neurons and self.cmap is not None:
            highlight_color = self.cmap

        if just_show_highlighted:
            ff = np.squeeze(ff)
            ff_h = np.squeeze(ff[:, :, hn])
            print(ff_h.shape)
            fs_h = ff_h.shape
            flat_footprints = np.reshape(
                ff_h, (fs_h[0] * fs_h[1], fs_h[2]), order='F')
        else:
            flat_footprints = np.reshape(ff, (fs[0] * fs[1], fs[2]), order='F')

        coordinates = utils.plot_contours(flat_footprints, self.base_image,
                                          display_numbers=display_numbers,
                                          maxthr=maxthr, thr_method='max',
                                          colors=edge_color, cmap=cmap,
                                          highlight_neurons=hn,
                                          highlight_color=highlight_color,
                                          just_show_highlighted=(
                                            just_show_highlighted),
                                          contour_linewidth=contour_linewidth,
                                          rotate_vertical=rotate_image,
                                          swap_dim=False,
                                          show_footprints=show_footprints)

        if not no_borders:
            plt.xlabel('ncells = ' + str(fs[2]))
            plt.title(self.name)
            # ax.get_xaxis().set_ticks([])
            # ax.get_yaxis().set_ticks([])
        else:
            plt.axis('off')

        vmin, vmax = plt.gca().get_images()[0].get_clim()
        if atlas_outline is not None:
            vmin, vmax = plt.gca().get_images()[0].get_clim()
            overlay = reg.overlay_atlas_outline(atlas_outline,
                                                self.base_image)
            if rotate_image:
                overlay = rotate(overlay, -90)
        else:
            if rotate_image:
                overlay = rotate(self.base_image, -90)
            else:
                overlay = self.base_image
        plt.imshow(overlay, cmap='gray', clim=[vmin, vmax], aspect='equal')

        if show_footprints:
            for i in range(ff_h.shape[2]):
                if rotate_image:
                    ff_show = rotate(ff_h[:, :, i], -90)
                else:
                    ff_show = ff_h[:, :, i]
                transp_imshow(ff_show,
                              cmap='Blues', gam=1)

        if self.fig_save_path is not None:
            if self.suffix is not None:
                fig_save_dir = os.path.join(
                    self.fig_save_path, self.date, self.name)
                if not os.path.isdir(fig_save_dir):
                    os.makedirs(fig_save_dir)
                print('Saving to ' + os.path.join(
                    fig_save_dir, 'contours_' + self.suffix))
                fname = os.path.join(
                    fig_save_dir, 'contours_' + 'atlas'
                    + str(int(atlas_outline is not None)) + '_' + self.suffix)
                plt.savefig(
                        fname,
                        bbox_inches='tight',
                        pad_inches=0.0)

        print("---Plotting contours: %s seconds ---" %
              (time.time() - start_time))

        return coordinates

    def _normalize(self, trace, percentile=True):
        """
        Normalize a fluorescence trace by its max or its 99th percentile.
        :return trace: normalized trace
        :return scale_factor: normalization scale factor (e.g. for errorbars)
        """
        trace = trace - np.percentile(trace, 15)
        scale_factor = 1
        if np.percentile(trace, 99) > 0:
            if percentile:
                scale_factor = np.percentile(trace, 99)
            else:
                scale_factor = np.max(trace)
            trace = trace / scale_factor
        trace = trace - np.percentile(trace, 15)
        return trace, scale_factor
