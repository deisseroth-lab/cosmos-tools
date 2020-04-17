import os
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
import warnings
import scipy.sparse
from scipy.signal import savgol_filter
import h5py

import bokeh
import bokeh.models as models
import bokeh.plotting as bpl
from bokeh.models import CustomJS, ColumnDataSource, Range1d
from bokeh.plotting import figure, show
from bokeh.io import export_svgs
from bokeh.io import output_file, show, output_notebook, push_notebook

import cosmos.imaging.atlas_registration as reg
import cosmos.imaging.cell_selection as utils
import cosmos.traces.cell_plotter as cell_plotter
import cosmos.traces.trace_analysis_utils as trace_utils
from cosmos.imaging.cell_sorter import CellSorter

from IPython.display import clear_output
from IPython.core.debugger import set_trace


class CosmosDataset:
    """Loads, plots, and merges output of CNMFe
    for a single COSMOS dataset (which contains
    'top' and 'bot' focus video stacks.)
    """

    def __init__(self, base_path, dataset_info, fig_save_path):
        self.dataset_info = dataset_info  # A dict containing: 'date', 'name'.
        self.base_path = base_path  # Top dir where CNMFe out.mat is found.
        self.fig_save_path = fig_save_path  # Top dir for saving figures.
        self.date = dataset_info['date']
        self.name = dataset_info['name']
        self.data = self._load_dataset()

        # Variables to be filled during processing.
        self.tform = None
        self.footprints_aligned = dict()
        self.mean_frames_aligned = dict()
        self.keypoints_aligned = dict()
        self.cm_aligned = dict()
        self.roi_map_aligned = dict()
        self.corr_frames_aligned = dict()
        self.areas = dict()
        self.tseries = dict()
        self.spikes = dict()
        self.tseries_raw = dict()
        self.atlas_info = dict()
        self.footprints_unalign = None
        self.cm_unalign = None
        self.keypoints_unalign = None
        self.mean_frames_unalign = None

        # For 'bot_focus' and 'top_focus', which cells are included
        # after merging. Assumes that this is using culled traces.
        self.included = dict()

    def cull(self, do_load=True, do_auto=False, do_manual=True, which_key=None,
             only_show_non_culled=True):
        """
        Select good traces.
        Has an automatic culler (biased towards fewer false negatives),
        and a GUI interface for manual culling.
        The automated classifier selects neurons that have a symmetric shape,
        and where the deconvolved trace is a good match to the raw trace.

        :param do_load: Load previously saved record of good cells.
        :param do_auto: Run automatic classifier (takes a little while).
        :param do_manual: Run GUI. If neither do_load or do_auto is true,
                          then, it will show all neurons.
        :param which_key: If only one of the datasets in self.data
                          should be run. (i.e. 'top_focus', 'bot_focus',
                          'two_photon', or 'single_focus').
        :param only_show_non_culled: Bool.
            If True, then does not display neurons that have been previously
            culled (i.e. either from auto-culling or a previous run of manual
                    culling).

        :return None. Saves keep_cells.npz file with 'keep_cells'
                      - the indices of the good cells - to the same
                      directory that the original CNMF output mat file
                      is saved. If manual culling has been performed,
                      then also saves a manual_backup.npz file
                      (just in case future auto-culling overwrites
                      the main file).
        """
        for key, d in self.data.items():
            if not d:
                continue
            if which_key is not None:
                if key != which_key:
                    continue

            results = d['results']
            base_im = results['Cn']
            base_im = base_im/np.amax(base_im)

            vid_path = d['path']
            vid_dir = vid_path.rsplit("/", 1)[0]
            save_path = os.path.join(vid_dir, 'keep_cells.npz')

            ncells = results['A'].shape[1]
            footprints = np.array(results['A'][:, :].todense())
            footprints = np.reshape(
                footprints, (base_im.shape[0], base_im.shape[1], ncells),
                order='F')

            keep_cells = None

            if do_load:
                if os.path.isfile(save_path):
                    f = np.load(save_path)
                    keep_cells = f['keep_cells']
                else:
                    print('Cannot load: ' + save_path)
                    print('Auto-culling...')
                    do_auto = True

            if do_auto:
                # corr_thresh = 0.85
                # aspect_ratio_thresh = 1.8
                corr_thresh = 0.7
                print('Autoculling with corr_thresh='+str(corr_thresh))
                aspect_ratio_thresh = None
                keep_cells, corrs, aspect_ratios = utils.classify_neurons(
                    results, corr_thresh, aspect_ratio_thresh)
                np.savez(save_path, keep_cells=keep_cells)

            if do_manual:
                # Type 'k' to keep a cell or 'd' to delete.
                # YOU MUST TYPE 's' TO SAVE YOUR SELECTIONS.
                clear_output()
                ncells = footprints.shape[2]
                traces = results['C']
                traces_raw = results['C_raw']
                if keep_cells is None:
                    print('Displaying all cells (with no culling).')
                    keep_cells = 1 * np.ones((ncells, 1))  # Passed to GUI.

                use_class = True
                if not use_class:
                    pass
                    # Deprecated.
                    # utils.cell_select_GUI(footprints, traces, traces_raw,
                    #                       base_im, save_path, keep_cells,
                    #                       do_only_auto_culled)
                else:
                    CS = CellSorter(footprints, traces, traces_raw,
                                    base_im, save_path, keep_cells,
                                    show_only_kept_cells=only_show_non_culled)

    def plot_cells_premerge(self, cell_ids, do_traces=True,
                            do_contours=True, use_culled_cells=True,
                            highlight_neurons=True, which_key=None,
                            n_timepoints=800):
        """
        Instantiates a CellPlotter class to plot traces
        and contours for cells in the original dataset,
        before any merging across focal planes has occurred.

        :param cell_ids: np.array of cell_ids to plot
                         (i.e. np.arange(0, 20))
        :param do_traces: bool. Plot traces.
        :param do_contours: bool. Plot cell ROI contours.
        :param use_culled_cells: bool. If false, use all cells,
                                 not just the good ones selected
                                 with self.cull()
        :param display_numbers: bool. Display ROI numbers on the
                                contour plot.
        :param which_key: If only one of the datasets in self.data
                          should be plotted. (i.e. 'top_focus',
                          'bot_focus', 'two_photon', or 'single_focus').
        """

        for key, d in self.data.items():
            if not d:
                continue
            if which_key is not None:
                if key != which_key:
                    continue

            results = d['results']
            fpath = d['path']

            if use_culled_cells:
                f = np.load(
                    os.path.join(fpath.rsplit("/", 1)[0], 'keep_cells.npz'))
                which_cells = np.where(f['keep_cells'] > 0)[0]
            else:
                which_cells = np.arange(results['C'].shape[0])

            # Gather traces and footprints.

            # Inferred spikes
            S = np.array(results['S'].todense())[which_cells, :]

            # Denoised fluorescence
            C = np.array(results['C'])[which_cells, :]

            # Raw fluorescence
            F = np.array(results['C_raw'])[which_cells, :]
            base_im = results['meanFrame']
            ff = np.asarray(results['A'].todense())
            ff = np.reshape(
                ff, (base_im.shape[0], base_im.shape[1], ff.shape[1]),
                order='F')
            ff = ff[:, :, which_cells]

            # Initialize CellPlotter class with data.
            CP = cell_plotter.CellPlotter(
                C, F, ff, base_im, date=self.date,
                name=self.name, fig_save_path=self.fig_save_path,
                suffix='premerge_'+key+'.pdf')
            CP.set_highlighted_neurons(cell_ids)
            fig = plt.figure(figsize=(20, 10))
            if do_traces:
                CP.plot_traces(
                    n_timepoints=n_timepoints, ax=plt.subplot(121),
                    save_plot=False)
            if do_contours:
                CP.plot_contours(highlight_neurons=highlight_neurons,
                                 display_numbers=highlight_neurons,
                                 ax=plt.subplot(122),
                                 rotate_image=False)

    def align_planes(self, use_culled_cells=True, do_debug=False):
        """
        Aligns 'top_focus' and 'bot_focus' videos together.
        Alignment transform is generated based on manually
        selected keypoints (i.e. from get_alignment_keypoints()
        function in frameProcessor.py), which are saved out in
        the same location as the CNMFe out.mat results file.

        Fills the following class variables:
            tform: the learned alignment transform, for use in skimage
            keypoints_aligned: the aligned xy coords of the 2 keypoints
            cm_aligned: the yx coords of the centers of mass of each roi
            areas: the area of each roi
            mean_frames_aligned: the aligned mean video frames
            corr_frames_aligned: the aligned correlation images (from cnmf)
            footprints_aligned: the aligned [X x Y x T] stack of
                                neural footprint maps
            roi_map_aligned: image where the value at each cm of
                             of each roi corresponds to the area of
                             that roi.
        """

        do_alignment = (len(self.data['bot_focus']) > 0 and
                        len(self.data['top_focus']) > 0)

        if do_alignment:
            keypoints_unalign = dict()
            mean_frames_unalign = dict()
            footprints_unalign = dict()
            cm_unalign = dict()
            corr_frames_unalign = dict()

            # Load and gather data.
            for key in ['bot_focus', 'top_focus']:
                fpath = self.data[key]['path']
                results = self.data[key]['results']

                f = np.load(
                    os.path.join(fpath.rsplit("/", 1)[0], 'keypoints.npz'))
                keypoints_unalign[key] = f['coords'][0:2, :]

                if key == 'top_focus':
                    for npzkey in f.keys():
                        self.atlas_info[npzkey] = f[npzkey]

                mean_frames_unalign[key] = results['meanFrame']
                corr_frames_unalign[key] = results['Cn']

                if use_culled_cells:
                    f = np.load(
                        os.path.join(
                            fpath.rsplit("/", 1)[0], 'keep_cells.npz'))
                    which_cells = np.where(f['keep_cells'] > 0)[0]
                else:
                    which_cells = np.arange(results['C'].shape[0])

                print('Loading footprints.')
                footprints = np.array(results['A'][:, which_cells].todense())
                print('Reshaping footprints.')
                footprints_unalign[key] = np.reshape(
                    footprints, (mean_frames_unalign[key].shape[0],
                                 mean_frames_unalign[key].shape[1],
                                 footprints.shape[1]), order='F')

                cm_unalign[key], self.areas[key] = utils.get_cm(
                    footprints_unalign[key])

                self.tseries[key] = np.array(
                    results['C'])[which_cells, :]
                self.spikes[key] = np.array(
                    results['S'].todense())[which_cells, :]
                self.tseries_raw[key] = np.array(
                    results['C_raw'])[which_cells, :]

            # Now, do alignments.
            print('Obtaining alignment transform.')
            self.tform = utils.fit_transform(keypoints_unalign['top_focus'],
                                             keypoints_unalign['bot_focus'])
            aligned_keypoint = utils.transform_points(
                self.tform, keypoints_unalign['bot_focus'])
            self.keypoints_aligned['top_focus'] = keypoints_unalign[
                'top_focus']
            self.keypoints_aligned['bot_focus'] = aligned_keypoint

            print('Aligning mean frames.')
            aligned_frame = utils.align_image(
                self.tform, mean_frames_unalign['bot_focus'])
            aligned_frames = utils.crop_images(
                mean_frames_unalign['top_focus'], aligned_frame)
            self.mean_frames_aligned['top_focus'] = aligned_frames[0]
            self.mean_frames_aligned['bot_focus'] = aligned_frames[1]

            print('Cropping atlas.')
            aligned_atlases = utils.crop_images(self.atlas_info['img'],
                                                aligned_frames[0])
            self.atlas_info['img'] = aligned_atlases[0]
            aligned_atlases = utils.crop_images(
                self.atlas_info['aligned_atlas_outline'],
                aligned_frames[0])
            self.atlas_info['aligned_atlas_outline'] = aligned_atlases[0]

            print('Aligning corr frames.')
            aligned_frame = utils.align_image(
                self.tform, corr_frames_unalign['bot_focus'])
            aligned_frames = utils.crop_images(
                corr_frames_unalign['top_focus'], aligned_frame)
            self.corr_frames_aligned['top_focus'] = aligned_frames[0]
            self.corr_frames_aligned['bot_focus'] = aligned_frames[1]

            print('Aligning stacks.')
            aligned_footprint = utils.align_stacks(
                self.tform, footprints_unalign['bot_focus'])
            aligned_footprints = utils.crop_stacks(
                footprints_unalign['top_focus'], aligned_footprint)
            self.footprints_aligned['top_focus'] = aligned_footprints[0]
            self.footprints_aligned['bot_focus'] = aligned_footprints[1]

            self.footprints_unalign = footprints_unalign

            aligned_cm = np.fliplr(utils.transform_points(
                self.tform, np.fliplr(cm_unalign['bot_focus'])))
            self.cm_aligned['top_focus'] = cm_unalign['top_focus']
            self.cm_aligned['bot_focus'] = aligned_cm

            self.roi_map_aligned['top_focus'] = utils.get_roi_map(
                self.cm_aligned['top_focus'], self.areas['top_focus'],
                self.mean_frames_aligned['top_focus'].shape)
            self.roi_map_aligned['bot_focus'] = utils.get_roi_map(
                self.cm_aligned['bot_focus'], self.areas['bot_focus'],
                self.mean_frames_aligned['bot_focus'].shape)

            # Potentially optional save-out of unaligned variables,
            # for testing purposes.
            self.cm_unalign = cm_unalign
            self.footprints_unalign = footprints_unalign
            self.keypoints_unalign = keypoints_unalign
            self.mean_frames_unalign = mean_frames_unalign

            if do_debug:
                print("Keypoints_aligned should match between top and bottom")
                print(self.keypoints_aligned)
                print("Here are the unaligned keypoints")
                print(self.keypoints_unalign)

                plt.figure()
                plt.imshow(self.mean_frames_aligned['top_focus'])
                plt.imshow(self.mean_frames_aligned['bot_focus'])

    def merge_planes(self, do_debug=False, do_plots=False,
                     manual_keypoints=None):
        """
        Merge neuron rois from 'top_focus'
        and 'bot_focus' stacks.

        :param manual_keypoints: (for advanced user). This allows you to
                                provide a tuple of two sets of xy coordinates,
                                for the top and bottom hemisphere,
                                which will manually define the keypoints
                                for fitting the classification line for
                                merging the top and bot focus images.

        #TODO: If 'single_focus' stack is provided,
        then save out all traces from the single
        video, as if they were merged.

        The strategy for merging is to find a dividing line
        on each hemisphere, such that the ROI sizes are smaller
        on one side for 'top_focus' and smaller on the other
        side for 'bot_focus'.

        Fills self.included class variable.
        """

        do_merge = (len(self.data['bot_focus']) > 0 and
                    len(self.data['top_focus']) > 0)

        if do_merge:
            left_classifier, right_classifier, blocks_inds = \
                    self.get_classification_lines(
                        nblocks=10, do_plot_classification_lines=do_plots,
                        do_plot_blocks_raw=do_debug,
                        do_plot_blocks_means=do_debug,
                        manual_keypoints=manual_keypoints)

            included = dict()
            for key in ['top_focus', 'bot_focus']:
                is_inside = (key == 'top_focus')
                cm = self.cm_aligned[key]
                included[key] = self._classify_points(
                    cm[:, 1], cm[:, 0], left_classifier,
                    right_classifier, is_inside)
            self.included = included

            do_plot_separate = do_plots
            if do_plot_separate:
                for key in ['top_focus', 'bot_focus']:
                    cm = self.cm_aligned[key]
                    area = self.areas[key]
                    incl = included[key]
                    p1 = figure(plot_width=600, plot_height=600)
                    c = p1.circle(
                        x=cm[:, 1], y=cm[:, 0],
                        radius=area[:, 0]/300.0, color='blue')
                    c = p1.circle(
                        x=np.squeeze(cm[incl, 1]), y=np.squeeze(cm[incl, 0]),
                        radius=np.squeeze(area[incl, 0])/300.0,
                        color='firebrick')
                    xx = blocks_inds[key][1:]
                    yy_plot_left = left_classifier.predict(xx[:, np.newaxis])
                    yy_plot_right = right_classifier.predict(xx[:, np.newaxis])
                    lfit = p1.line(y=yy_plot_left,
                                   x=xx, color='green', line_width=3)
                    rfit = p1.line(y=yy_plot_right,
                                   x=xx, color='green', line_width=3)
                    show(p1)
                    p1.output_backend = "svg"
                    export_svgs(
                        p1, filename=os.path.join(self.fig_save_path,
                                                  self.date, self.name,
                                                  "roi_"+key+".svg"))

            do_plot_merged = do_plots
            if do_plot_merged:
                p1 = figure(plot_width=600, plot_height=600)
                for key in ['top_focus', 'bot_focus']:
                    cm = self.cm_aligned[key]
                    incl = included[key]
                    area = self.areas[key]
                    c = p1.circle(x=np.squeeze(cm[incl, 1]),
                                  y=np.squeeze(cm[incl, 0]),
                                  radius=np.squeeze(area[incl, 0]/300.0),
                                  color='firebrick')

                p1.title.text = 'Merged'
                show(p1)
                p1.output_backend = "svg"
                export_svgs(p1, filename=os.path.join(
                    self.fig_save_path, self.date, self.name,
                    "roi_"+"merged"+".svg"))

    def get_classification_lines(self, nblocks=10,
                                 do_plot_blocks_raw=False,
                                 do_plot_blocks_means=False,
                                 do_plot_classification_lines=True,
                                 manual_keypoints=None):
        """
        Computes classification lines for the left
        and right hemispheres of the brain, such that
        the ROI sizes are smaller on one side 'top_focus'
        and smaller on the other side for 'bot_focus'.

        Because of the specific geometry of the COSMOS
        preparation, where the brain is smoothly curved,
        we know that there should be two total classification
        lines: the 'bot_focus' image will be in focus on
        the lateral left and right sides, and the 'top_focus'
        will be in focus in the medial region.

        We determine the position of the classification lines
        by breaking each image up into strips, or blocks, that
        from the left to right side of the brain, and are stacked
        from posterior to anterior. Within each block, we compute
        a smoothed mean ROI size at each lateral position.
        We then determine the lateral position at which the mean
        ROI size for the 'top_focus' and 'bot_focus' images cross
        one another. This position is used as the classifier position
        within in that block.
        Given the classifier position for each block, we fit a smoothed
        and quadratically interpolated line that can then be used to
        classify all ROIs.

        The function returns the interpolated lines (as sklearn model
        objects) for the right and left hemispheres.

        ## Change 2/15/18: Do not use the two endpoints for fitting the
                           polynomial.

        :return left_classifier: sklearn model object for classification
                                 line on the left hemisphere.
        :return right_classifier: sklearn model object for classification
                                 line on the right hemisphere.
        """
        do_classification = (len(self.data['bot_focus']) > 0 and
                             len(self.data['top_focus']) > 0)

        if not do_classification:
            raise ValueError('get_classification_lines() was called on a '
                             'CosmosDataset object that does not contain'
                             'either a bot_focus or top_focus dataset.')

        all_blocks, blocks_inds = self._get_block_projections(
            nblocks, ['top_focus', 'bot_focus'], self.roi_map_aligned)
        print(blocks_inds)
        do_plot_blocks_raw = False
        if do_plot_blocks_raw:
            for i in range(nblocks):
                plt.figure()
                plt.plot(all_blocks['top_focus'][:, i], 'bo')
                plt.plot(all_blocks['bot_focus'][:, i], 'ro')
                plt.ylim([0, 700])

        # Find classification positions for each block.
        left_inds = []  # Left classification position for each block.
        right_inds = []  # Right classification position for each block.
        if manual_keypoints is None:
            do_median = False
            for i in range(nblocks):
                block_means = dict()
                for key in ['top_focus', 'bot_focus']:
                    block_means[key] = trace_utils.moving_avg(
                        all_blocks[key][:, i], do_median)
                L = min([len(val) for (key, val) in block_means.items()])
                for key in ['top_focus', 'bot_focus']:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",  np.RankWarning)
                        block_means[key] = savgol_filter(
                            block_means[key][:L], 21, 3)

                # Find intersection points where ROI size starts to become
                # bigger for the bot_focus vs. top_focus images.
                doOrig = False  # Use the simpler, less-robust original method.
                if doOrig:
                    inds = np.where(
                        block_means['bot_focus'] > block_means['top_focus'])[0]
                    if len(inds) == 0:
                        inds = np.array([0, L-1])
                    inds = inds.astype(int)

                    left_ind = inds[0]
                    right_ind = inds[-1]
                else:
                    block_means['bot_focus'] = self._interp_nans(
                        block_means['bot_focus'])
                    block_means['top_focus'] = self._interp_nans(
                        block_means['top_focus'])

                    intersections = np.where(
                        np.diff(np.sign(block_means['bot_focus']
                                        - block_means['top_focus'])))[0]
                    d = 20
                    intersections = intersections[
                        np.where(np.logical_and(intersections > d,
                                 intersections < L-d))[0]]

                    if np.isnan(block_means['bot_focus']).all():
                        left_crossing_inds = np.array([])
                        right_crossing_inds = np.array([])
                    elif np.isnan(block_means['top_focus']).all():
                        left_crossing_inds = np.array([])
                        right_crossing_inds = np.array([])
                    else:
                        left_crossing_inds = np.where(np.logical_and(
                            block_means['bot_focus'][intersections+d] >
                            block_means['top_focus'][intersections+d],
                            block_means['bot_focus'][intersections-d] <
                            block_means['top_focus'][intersections-d]))[0]
                        right_crossing_inds = np.where(np.logical_and(
                            block_means['bot_focus'][intersections+d] <
                            block_means['top_focus'][intersections+d],
                            block_means['bot_focus'][intersections-d] >
                            block_means['top_focus'][intersections-d]))[0]

                    if not left_crossing_inds.size:
                        left_ind = np.nan  # 0
                    else:
                        left_ind = np.amin(intersections[left_crossing_inds])
                        if left_ind > L/2:
                            left_ind = int(L/2)

                    if not right_crossing_inds.size:
                        right_ind = np.nan  # L-1
                    else:
                        right_ind = np.amax(intersections[right_crossing_inds])
                        if right_ind < L/2:
                            right_ind = int(L/2)

                left_inds.append(left_ind)
                right_inds.append(right_ind)

                if do_plot_blocks_means:
                    plt.figure()
                    mtop = block_means['top_focus']
                    mbot = block_means['bot_focus']
                    plt.plot(mtop, 'b-')
                    plt.plot(mbot, 'r-')
                    if ~np.isnan(left_inds[i]):
                        plt.plot(left_inds[i], mtop[left_inds[i]], 'go')
                    if ~np.isnan(right_inds[i]):
                        plt.plot(right_inds[i], mtop[right_inds[i]], 'ro')

            left_inds = self._interp_nans(np.array(left_inds)).astype(int)
            right_inds = self._interp_nans(np.array(right_inds)).astype(int)

            xx = blocks_inds['top_focus'][3:-1]
            yy = np.array(left_inds[3:-1])
            yy_plot_left, classifier_left = utils.fit_polynomial(
                xx, yy, xx, 2)

            yy = np.array(right_inds[3:-1])
            yy_plot_right, classifier_right = utils.fit_polynomial(
                xx, yy, xx, 2)

            if do_plot_classification_lines:
                for key in ['top_focus', 'bot_focus']:
                    cm = self.cm_aligned[key]
                    area = self.areas[key]
                    p1 = figure(plot_width=600, plot_height=600)
                    c = p1.circle(x=cm[:, 1], y=cm[:, 0],
                                  radius=area[:, 0] / 300.0)
                    ll = p1.line(y=left_inds[1:], x=blocks_inds[key][1:],
                                 color='orange')
                    r = p1.line(y=right_inds[1:], x=blocks_inds[key][1:],
                                color='orange')
                    lfit = p1.line(y=yy_plot_left, x=xx, color='green',
                                   line_width=3)
                    rfit = p1.line(y=yy_plot_right, x=xx, color='green',
                                   line_width=3)
                    #                 show(p1)
                    p1.output_backend = "svg"
                    export_svgs(
                        p1, filename=os.path.join(self.fig_save_path,
                                                  self.date, self.name,
                                                  'classification_lines_' +
                                                  key + '.svg'))
                    show(p1)
        else:
            print('Using manual keypoints')
            right_inds = np.array(manual_keypoints[0])
            left_inds = np.array(manual_keypoints[1])

            left_x = left_inds[:, 0]
            left_y = left_inds[:, 1]

            right_x = right_inds[:, 0]
            right_y = right_inds[:, 1]

            xx = left_x
            yy = left_y
            yy_plot_left, classifier_left = utils.fit_polynomial(xx, yy, xx, 2)

            xx = right_x
            yy = right_y
            yy_plot_right, classifier_right = utils.fit_polynomial(
                xx, yy, xx, 2)

            if do_plot_classification_lines:
                for key in ['top_focus', 'bot_focus']:
                    cm = self.cm_aligned[key]
                    area = self.areas[key]
                    p1 = figure(plot_width=600, plot_height=600)
                    c = p1.circle(x=cm[:, 1], y=cm[:, 0],
                                  radius=area[:, 0] / 300.0)
                    ll = p1.line(y=left_y, x=left_x,
                                 color='orange')
                    r = p1.line(y=right_y, x=right_x,
                                color='orange')
                    lfit = p1.line(y=yy_plot_left, x=xx, color='green',
                                   line_width=3)
                    rfit = p1.line(y=yy_plot_right, x=xx, color='green',
                                   line_width=3)
                    #                 show(p1)
                    p1.output_backend = "svg"
                    export_svgs(
                        p1, filename=os.path.join(self.fig_save_path,
                                                  self.date, self.name,
                                                  'classification_lines_' +
                                                  key + '.svg'))
                    show(p1)

        return classifier_left, classifier_right, blocks_inds

    def save_merged(self):
        """
        Saves an h5 file containing the merged
        traces and footprints, as well as the aligned atlas.
        If working with a two_photon or single_focus
        dataset, will directly save the traces and
        footprints from the single video (but in
        the same format as with the merged equivalent.)
        """

        do_merge = (len(self.data['bot_focus']) > 0 and
                    len(self.data['top_focus']) > 0)

        if do_merge:
            incl_t = self.included['top_focus'][0]
            incl_b = self.included['bot_focus'][0]
            save_focmask = False
            if save_focmask:
                focmask_path = os.path.join(
                    self.base_path, self.date, self.name, 'focus_incl_masks')
                np.savez(focmask_path, incl_top=incl_t, incl_bot=incl_b)
                return

            merged_footprints = np.concatenate(
                (self.footprints_aligned['top_focus'][:, :, incl_t],
                 self.footprints_aligned['bot_focus'][:, :, incl_b]),
                axis=2)

            merged_cm = np.vstack((self.cm_aligned['top_focus'][incl_t, :],
                                   self.cm_aligned['bot_focus'][incl_b, :]))

            merged_areas = np.vstack((self.areas['top_focus'][incl_t, :],
                                      self.areas['bot_focus'][incl_b, :]))

            merged_tseries = np.vstack(
                (self.tseries['top_focus'][incl_t, :],
                 self.tseries['bot_focus'][incl_b, :]))

            merged_spikes = np.vstack(
                (self.spikes['top_focus'][incl_t, :],
                 self.spikes['bot_focus'][incl_b, :]))

            merged_tseries_raw = np.vstack(
                (self.tseries_raw['top_focus'][incl_t, :],
                 self.tseries_raw['bot_focus'][incl_b, :]))

            merged_mean_frames = np.concatenate(
                (np.expand_dims(self.mean_frames_aligned['top_focus'], 2),
                 np.expand_dims(self.mean_frames_aligned['bot_focus'], 2)),
                axis=2)

            merged_corr_frames = np.concatenate(
                (np.expand_dims(self.corr_frames_aligned['top_focus'], 2),
                 np.expand_dims(self.corr_frames_aligned['bot_focus'], 2)),
                axis=2)

            merged_focus_id = np.vstack(
                (np.zeros((len(incl_t), 1)), np.ones((len(incl_b), 1))))

            merged_atlas_info = self.atlas_info
        else:
            if len(self.data['single_focus']) > 0:
                raise('single_focus Not yet implemented')
                pass

        try:
            led_frames = np.load(
                os.path.join(self.base_path,
                             self.date, self.name, 'led_frames.npz'))['inds']
        except Exception:
            print('Could not load led_frames.')
            led_frames = np.array([])

        save_path = os.path.join(self.base_path, self.date, self.name,
                                 self.date + '-' + self.name +
                                 '-merged_traces_unculled.h5')

        with h5py.File(save_path, "w") as hf:
            hf.create_dataset("footprints", data=merged_footprints,
                              compression="gzip", compression_opts=9)
            hf.create_dataset("cm", data=merged_cm)
            hf.create_dataset("areas", data=merged_areas)
            hf.create_dataset("tseries", data=merged_tseries)
            hf.create_dataset("tseries_raw", data=merged_tseries_raw)
            hf.create_dataset("spikes", data=merged_spikes)
            hf.create_dataset("mean_frames", data=merged_mean_frames)
            hf.create_dataset("corr_frames", data=merged_corr_frames)
            hf.create_dataset("focus_id", data=merged_focus_id)
            hf.create_dataset("date", data=self.date)
            hf.create_dataset("name", data=self.name)
            hf.create_dataset("led_frames", data=led_frames)

            atlas_group = hf.create_group("atlas_info")
            atlas_group.create_dataset(
                "aligned_atlas_outline",
                data=merged_atlas_info['aligned_atlas_outline'])
            atlas_group.create_dataset("atlas_coords",
                                       data=merged_atlas_info['atlas_coords'])
            atlas_group.create_dataset("img",
                                       data=merged_atlas_info['img'])
            atlas_group.create_dataset("atlas",
                                       data=merged_atlas_info['atlas'])
            atlas_group.create_dataset("coords",
                                       data=merged_atlas_info['coords'])

        print('Saved to: ' + save_path)
        return save_path

    def plot_cells_postmerge(self, cell_ids, do_traces=True,
                             do_contours=True, do_atlas_overlay=True,
                             highlight_neurons=True, n_timepoints=800,
                             do_events=True, do_seconds=True,
                             load_culled=False):
        """
        Loads merged traces and footprints, and plots traces
        and contours, highlighting those specified in cell_ids.
        :param cell_ids: np.array (i.e  np.arange(90, 93) )
        :param do_traces: bool. Plot traces for specified cell_ids.
        :param do_contours: bool. Plot contours.
        :param do_atlas_overlay: bool. Overlay aligned atlas on plot.
        :param highlight_neurons: bool. Highlight specified cell_ids in the
                                    contour plot.
        :param n_timepoints: int. How many frames to plot in trace plot.
        :param do_events: bool. Plot vertical lines at led_frame times.
        :param do_seconds: bool. Make x-axis in seconds (vs frames).
        """
        if not load_culled:
            h5_path = os.path.join(
                self.base_path, self.date, self.name,
                self.date + '-' + self.name +
                '-merged_traces_unculled.h5')
        else:
            h5_path = os.path.join(
                self.base_path, self.date, self.name,
                self.date + '-' + self.name +
                '-merged_traces.h5')

        with h5py.File(h5_path, "r") as hf:
            'spikes', 'tseries', 'tseries_raw',
            S = np.array(hf['spikes'])
            C = np.array(hf['tseries'])
            F = np.array(hf['tseries_raw'])
            ff = np.array(hf['footprints'])
            base_im = np.squeeze(np.array(hf['mean_frames'])[:, :, 0])
            led_frames = np.array(hf['led_frames'])

            if 'atlas_info' in hf.keys():
                atlas_info = hf['atlas_info']
                aligned_atlas_outline = np.array(
                    atlas_info['aligned_atlas_outline'])
                atlas = np.array(atlas_info['atlas'])
                atlas_coords = np.array(atlas_info['atlas_coords'])
                img_coords = np.array(atlas_info['coords'])
                img = np.array(atlas_info['img'])
            else:
                print('atlas_info not a member of: ' + h5_path)
                do_atlas_overlay = False

        CP = cell_plotter.CellPlotter(
                C, F, ff, base_im, date=self.date, name=self.name,
                fig_save_path=self.fig_save_path, suffix='postmerge.pdf')
        CP.set_highlighted_neurons(cell_ids)
        fig = plt.figure(figsize=(20, 10))
        if do_traces:
            if do_seconds:
                dt = 1.0/34
            else:
                dt = 1
            CP.plot_traces(n_timepoints=n_timepoints,
                           ax=plt.subplot(121),
                           save_plot=False,
                           event_frames=led_frames,
                           dt=dt)
            if do_seconds:
                plt.xlabel('Time [s]')
            else:
                plt.xlabel('Frame')

        if do_contours:
            CP.plot_contours(highlight_neurons=highlight_neurons,
                             display_numbers=highlight_neurons,
                             ax=plt.subplot(122),
                             maxthr=0.8,
                             rotate_image=False)

        if do_atlas_overlay:
            vmin, vmax = plt.gca().get_images()[0].get_clim()
            overlay = reg.overlay_atlas_outline(aligned_atlas_outline,
                                                img)
            plt.imshow(overlay, cmap='gray', clim=[vmin, vmax])

    def cull_postmerge(self, do_load=True, do_auto=False, do_manual=True,
                       which_key=None, nt=1e4, only_show_non_culled=True):
        """
        Select good traces in merged data.
        Has an automatic culler (biased towards fewer false negatives),
        and a GUI interface for manual culling.
        The automated classifier selects neurons that have a symmetric shape,
        and where the deconvolved trace is a good match to the raw trace.

        :param do_load: Load previously saved record of good cells.
        :param do_auto: Run automatic classifier (takes a little while).
        :param do_manual: Run GUI. If neither do_load or do_auto is true,
                          then, it will show all neurons.
        :param which_key: If only one of the datasets in self.data
                          should be run. (i.e. 'top_focus', 'bot_focus',
                          'two_photon', or 'single_focus').
        :param nt: How many timepoints to display in the traces.
                          Fewer timepoints leads to faster loading and display.
        :param only_show_non_culled: Bool. If True, then does not show neurons
                                     that have been previously culled (either
                                     from auto-culling or previous run of
                                     manual culling).

        :return None. Saves keep_cells.npz file with 'keep_cells'
                      - the indices of the good cells - to the same
                      directory that the original CNMF output mat file
                      is saved. If manual culling has been performed,
                      then also saves a manual_backup.npz file
                      (just in case future auto-culling overwrites
                      the main file).
        """

        # Load data.
        h5_path = os.path.join(self.base_path, self.date, self.name,
                               self.date + '-' + self.name +
                               '-merged_traces_unculled.h5')

        with h5py.File(h5_path, "r") as hf:
            footprints = np.array(hf['footprints'])
            cm = np.array(hf['cm'])
            areas = np.array(hf['areas'])
            tseries = np.array(hf['tseries'])
            tseries_raw = np.array(hf['tseries_raw'])
            spikes = np.array(hf['spikes'])
            mean_frames = np.array(hf['mean_frames'])
            corr_frames = np.array(hf['corr_frames'])
            focus_id = np.array(hf['focus_id'])
            date = hf['date']
            name = hf['name']
            led_frames = np.array(hf['led_frames'])

            if 'atlas_info' in hf.keys():
                atlas_info = hf['atlas_info']
                aligned_atlas_outline = np.array(
                    atlas_info['aligned_atlas_outline'])
                atlas = np.array(atlas_info['atlas'])
                atlas_coords = np.array(atlas_info['atlas_coords'])
                img_coords = np.array(atlas_info['coords'])
                img = np.array(atlas_info['img'])

        # Cull.
        results = {}
        results['C'] = tseries
        results['C_raw'] = tseries_raw
        results['A'] = footprints
        results['Cn'] = np.squeeze(corr_frames[:, :, 0])

        base_im = results['Cn']
        base_im = base_im / np.amax(base_im)

        vid_path = h5_path
        vid_dir = vid_path.rsplit("/", 1)[0]
        save_path = os.path.join(vid_dir, 'keep_cells_merged.npz')

        ncells = results['C'].shape[0]

        keep_cells = None

        if do_load:
            if os.path.isfile(save_path):
                f = np.load(save_path)
                keep_cells = f['keep_cells']
            else:
                print('Cannot load: ' + save_path)
                print('Auto-culling...')
                do_auto = True

        if do_auto:
            corr_thresh = 0.5
            aspect_ratio_thresh = None
            keep_cells, corrs, aspect_ratios = utils.classify_neurons(
                results,
                corr_thresh,
                aspect_ratio_thresh)
            np.savez(save_path, keep_cells=keep_cells)

        if do_manual:
            # Type 'k' to keep a cell or 'd' to delete.
            # YOU MUST TYPE 's' TO SAVE YOUR SELECTIONS.
            clear_output()
            ncells = footprints.shape[2]
            nt = int(nt)
            traces = results['C'][:, 1:nt]
            traces_raw = results['C_raw'][:, 1:nt]
            print('Only non-previously culled: '+str(only_show_non_culled))
            if keep_cells is None:
                print('Displaying all cells (with no culling).')
                keep_cells = 1 * np.ones(
                    (ncells, 1))  # Passed in by reference to GUI.

            CS = CellSorter(footprints, traces, traces_raw,
                            base_im, save_path, keep_cells,
                            show_only_kept_cells=only_show_non_culled)

    def save_cull_postmerge(self):
        """
        Saves a new h5 file containing the merged
        traces and footprints, as well as the aligned atlas.
        This new file will cull neurons based on the
        save keep_cells_merged.npz output of
        cull_postmerge().
        """

        # Load data.
        h5_path = os.path.join(self.base_path, self.date, self.name,
                               self.date + '-' + self.name +
                               '-merged_traces_unculled.h5')

        with h5py.File(h5_path, "r") as hf:
            footprints = np.array(hf['footprints'])
            cm = np.array(hf['cm'])
            areas = np.array(hf['areas'])
            tseries = np.array(hf['tseries'])
            tseries_raw = np.array(hf['tseries_raw'])
            spikes = np.array(hf['spikes'])
            mean_frames = np.array(hf['mean_frames'])
            corr_frames = np.array(hf['corr_frames'])
            focus_id = np.array(hf['focus_id'])
            date = str(np.array(hf['date']))
            name = str(np.array(hf['name']))
            led_frames = np.array(hf['led_frames'])

            if 'atlas_info' in hf.keys():
                atlas_info = hf['atlas_info']
                aligned_atlas_outline = np.array(
                    atlas_info['aligned_atlas_outline'])
                atlas = np.array(atlas_info['atlas'])
                atlas_coords = np.array(atlas_info['atlas_coords'])
                img_coords = np.array(atlas_info['coords'])
                img = np.array(atlas_info['img'])

        # Load keep_cells file.
        vid_path = h5_path
        vid_dir = vid_path.rsplit("/", 1)[0]
        keep_cells_path = os.path.join(vid_dir, 'keep_cells_merged.npz')
        if os.path.isfile(keep_cells_path):
            f = np.load(keep_cells_path)
            keep_cells = f['keep_cells']
        else:
            keep_cells = np.ones((tseries.shape[0], 1))
        keep_cells = keep_cells.astype(bool)
        keep_cells = np.where(keep_cells)[0]

        # Save merged and culled traces to a new file.
        save_path = os.path.join(self.base_path, self.date, self.name,
                                 self.date + '-' + self.name +
                                 '-merged_traces.h5')

        with h5py.File(save_path, "w") as hf:
            hf.create_dataset("footprints",
                              data=footprints[:, :, keep_cells],
                              compression="gzip", compression_opts=9)
            hf.create_dataset("cm",
                              data=cm[keep_cells, :])
            hf.create_dataset("areas",
                              data=areas[keep_cells, :])
            hf.create_dataset("tseries",
                              data=tseries[keep_cells, :])
            hf.create_dataset("tseries_raw",
                              data=tseries_raw[keep_cells, :])
            hf.create_dataset("spikes",
                              data=spikes[keep_cells, :])
            hf.create_dataset("mean_frames",
                              data=mean_frames)
            hf.create_dataset("corr_frames",
                              data=corr_frames)
            hf.create_dataset("focus_id",
                              data=focus_id[keep_cells, :])
            hf.create_dataset("date",
                              data=date)
            hf.create_dataset("name",
                              data=name)
            hf.create_dataset("led_frames",
                              data=led_frames)

            atlas_group = hf.create_group("atlas_info")
            atlas_group.create_dataset("aligned_atlas_outline",
                                       data=aligned_atlas_outline)
            atlas_group.create_dataset("atlas_coords",
                                       data=atlas_coords)
            atlas_group.create_dataset("img",
                                       data=img)
            atlas_group.create_dataset("atlas",
                                       data=atlas)
            atlas_group.create_dataset("coords",
                                       data=img_coords)

        print('Saved to: ' + save_path)
        return save_path

    @staticmethod
    def align_atlas_postmerge(h5_path):
        """
        Enables you to realign the atlas on a merged
        traces file.
        Note: THIS CANNOT BE RUN IN AN IPYTHON NOTEBOOK.
        It uses a gui interface, so you need to run it
        in the terminal or an IDE such as pycharm.

        Saves a new h5 file containing the merged
        traces and footprints, as well as the aligned atlas.
        """

        # Load data.
        print(h5_path)
        with h5py.File(h5_path, "r") as hf:
            footprints = np.array(hf['footprints'])
            cm = np.array(hf['cm'])
            areas = np.array(hf['areas'])
            tseries = np.array(hf['tseries'])
            tseries_raw = np.array(hf['tseries_raw'])
            spikes = np.array(hf['spikes'])
            mean_frames = np.array(hf['mean_frames'])
            corr_frames = np.array(hf['corr_frames'])
            focus_id = np.array(hf['focus_id'])
            date = str(np.array(hf['date']))
            name = str(np.array(hf['name']))
            led_frames = np.array(hf['led_frames'])

            if 'atlas_info' in hf.keys():
                atlas_info = hf['atlas_info']
                aligned_atlas_outline = np.array(
                    atlas_info['aligned_atlas_outline'])
                atlas = np.array(atlas_info['atlas'])
                atlas_coords = np.array(atlas_info['atlas_coords'])
                img_coords = np.array(atlas_info['coords'])
                img = np.array(atlas_info['img'])

        print('Aligning.')
        # Align atlas.
        atlas_out = reg.run_align_atlas_gui(img, h5_path)
        atlas_coords = atlas_out[0]
        img_coords = atlas_out[1]
        aligned_atlas_outline = atlas_out[2]
        atlas = atlas_out[3]

        # Save merged and culled traces to a new file.
        save_path = h5_path

        print('Saving.')
        with h5py.File(save_path, "w") as hf:
            hf.create_dataset("footprints",
                              data=footprints,
                              compression="gzip", compression_opts=9)
            hf.create_dataset("cm",
                              data=cm)
            hf.create_dataset("areas",
                              data=areas)
            hf.create_dataset("tseries",
                              data=tseries)
            hf.create_dataset("tseries_raw",
                              data=tseries_raw)
            hf.create_dataset("spikes",
                              data=spikes)
            hf.create_dataset("mean_frames",
                              data=mean_frames)
            hf.create_dataset("corr_frames",
                              data=corr_frames)
            hf.create_dataset("focus_id",
                              data=focus_id)
            hf.create_dataset("date",
                              data=date)
            hf.create_dataset("name",
                              data=name)
            hf.create_dataset("led_frames",
                              data=led_frames)

            atlas_group = hf.create_group("atlas_info")
            atlas_group.create_dataset("aligned_atlas_outline",
                                       data=aligned_atlas_outline)
            atlas_group.create_dataset("atlas_coords",
                                       data=atlas_coords)
            atlas_group.create_dataset("img",
                                       data=img)
            atlas_group.create_dataset("atlas",
                                       data=atlas)
            atlas_group.create_dataset("coords",
                                       data=img_coords)

        print('Saved to: ' + save_path)
        return save_path

    def _load_dataset(self):
        """
        Returns a dict containing results from CNMF.
        Possible keys are:
            'top_focus', 'bot_focus', 'two_photon', 'single_focus'
        Keys are determined based on the name of the saved out cnmf file.

        Note: The most relevant keys in each loaded out.mat file are:
            - `A` -- Sparse matrix of spatial filters/footprints
            - `S` -- Deconvolved spike trains estimated for each neuron
            - `C` -- Denoised calcium signals computed for each neuron
            - `C_raw` -- Raw calcium signals extracted from each neuron
            - `file` -- File that was analyzed to generate this results file
        """
        data = dict()

        print('Loading...')
        data['top_focus'] = self._load_dataset_mat(target_name='top_out.mat')
        data['bot_focus'] = self._load_dataset_mat(target_name='bot_out.mat')
        data['two_photon'] = self._load_dataset_mat(target_name='two_out.mat')
        data['single_focus'] = self._load_dataset_mat(
            target_name='single_out.mat')

        return data

    def _load_dataset_mat(self, target_name='out.mat'):
        """
        Loads and returns the first matching file name it finds.
        Directory structure: base_path/date/name/etc..../out.mat

        :param datasets: a list of dictionaries containing
                         keys 'date' and 'name'
        :param base_path: base path to find datasets in
        :param target_name: name of analysis mat files to find
        :return: list of mat file paths, list of loaded mat files.
        """

        data = dict()

        # Get the date and experiment name for the mat files we want to load
        dataset_path = os.path.join(self.base_path,
                                    self.date,
                                    self.name)

        # Load all the patches from that experiment
        for root, dirs, files in os.walk(dataset_path):
            if target_name in files:
                print(dataset_path + '/...' + target_name)
                mat_path = os.path.join(root, target_name)
                data['path'] = mat_path
                data['results'] = sio.loadmat(mat_path)
                return data

        return data

    def _get_block_projections(self, nblocks, keys, img_dict):
        """
        Splits an image (i.e. an roi_map) into blocks
        that extend from one lateral side to the other,
        and computes the nanmean projection along
        the other dimension.

        :param nblocks: int. Number of blocks to split the image into.
        :param img_dict: dict. For each key, contains an image
                               to be split into blocks. i.e. roi_map_aligned.
        :param keys: list. Keys in the img_dict to use.
                           I.e. ['top_focus', 'bot_focus']

        :returns all_blocks: dict. Keys are same as those passed
                             in to the function. Entries are the
                             mean projection for each block.
        :returns blocks_inds: dict. Keys are same as those passed
                             in to the function. Entries are the
                             anterior-posterior coordinates of
                             each block.
        """

        all_blocks = dict()
        blocks_inds = dict()

        nblocks = 10
        for key in keys:
            r_map = img_dict[key]
            r_map[r_map == 0] = np.nan

            ny = r_map.shape[0]
            nx = r_map.shape[1]
            blocks = np.zeros((ny, nblocks))
            blocks_ind = np.array(range(nblocks))*(nx/nblocks)
            blocks_inds[key] = blocks_ind
            for i in range(nblocks):
                block = r_map[:, int(i*nx/nblocks):int((i+1)*nx/nblocks)]
                # I expect to see RuntimeWarnings in this block.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    blocks[:, i] = np.nanmean(block, axis=1)

            all_blocks[key] = blocks

        return all_blocks, blocks_inds

    def _classify_points(self, xcm, ycm, classifier_left,
                         classifier_right, is_inside):
        """
        Given the coordinates of the center of mass of each roi,
        and sklearn model objects that specify the classifers for
        the left and right classification lines, determines whether
        each ROI is to be included in the merged dataset.

        :params xcm: np.array with x coordinates of ROI centers-of-mass
        :params ycm: same, but with y coordinates
        :params classifier_left: left sklearn model that defines a
                                 classification line
        :params classifier_right: same, but for the right line.
        :params is_inside: bool. Whether points to be included will be between
                the two classification lines, or outside the two lines.
        :returns included: np.array with indices of ROIs to be included.
        """
        xcm = xcm[:, np.newaxis]
        yfit_left = classifier_left.predict(xcm)
        yfit_right = classifier_right.predict(xcm)

        if is_inside:
            included = np.where((ycm > yfit_left) & (ycm < yfit_right))
        else:
            included = np.where((ycm < yfit_left) | (ycm > yfit_right))

        return included

    def _nan_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices)
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    def _interp_nans(self, y):
        """
        Interpolates nan values based on the endpoints
        on either side of the sequence of nans.
        """
        nans, x = self._nan_helper(y)
        if any(nans) and not all(nans):
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    def _shift(self, xs, n):
        """
        Shift an array by n indices. Positive is to the right.
        """
        if n >= 0:
            return np.concatenate((np.full(n, np.nan), xs[:-n]))
        else:
            return np.concatenate((xs[-n:], np.full(-n, np.nan)))
