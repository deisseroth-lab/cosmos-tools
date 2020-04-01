import os
import h5py
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import warnings
from scipy.ndimage.filters import gaussian_filter1d
import pickle

from cosmos.behavior.bpod_dataset import BpodDataset
import cosmos.imaging.atlas_registration as reg
import cosmos.imaging.img_io as cio
import cosmos.traces.trace_analysis_utils as utils


class CosmosTraces:
    """
    Open a COSMOS dataset containing traces.
    """

    def __init__(self, dataset_dict, behavior_plots=True,
                 ttl_plots=False, do_reshape_to_traces=True, do_region_plots=False,
                 use_parent=True, min_area_count=50, s_kernel_size=3,
                 nt_per_trial=7, ntrials=None):

        """
        :param dataset_dict: Can contain the following keys:
            'data_root': path to date-named folders that contain expt folders
            'name': mouse name
            'date': date of experiment
            'fig_save_dir': path to save out figures (optional)
            'behavior_dir': path containing bpod behavior files (optional)
            'ttl_file': name of pickle file containing TTL pulses (optional)
            'bpod_file': name of hdf5 behavior file (optional)
        :param behavior_plots: Make behavior plots? Default: True
        :param ttl_plots: Make TTL pulse plots? Default: False
        :param use_parent: Use higherlevel atlas annotations? Default: True
        :param min_area_count: Min number of ROIs in a region. Default: 20
        :param nt_per_trial: Min number of seconds of each trial.
        """

        self.root_path = dataset_dict['data_root']
        self.name = dataset_dict['name']
        self.date = dataset_dict['date']

        if 'regressors_name' in dataset_dict.keys():
            self.regressors_name = dataset_dict['regressors_name']
        else:
            self.regressors_name = None

        self.loaded_behavior = False
        self.loaded_ttl = False

        # Path to save figures.
        if 'fig_save_dir' not in dataset_dict.keys():
            dataset_dict['fig_save_dir'] = self.root_path
        if 'behavior_dir' not in dataset_dict.keys():
            dataset_dict['behavior_dir'] = os.path.join(self.root_path,
                                                        'behavior')
        self.fig_save_path = os.path.join(dataset_dict['fig_save_dir'],
                                          dataset_dict['date'],
                                          dataset_dict['name'])

        # Load up the output from CosmosDataset.saveMerged() (an HDF5 file).
        print('(1/3) Loading trace data.')
        h5_path = os.path.join(self.root_path, dataset_dict['date'],
                               dataset_dict['name'],
                               dataset_dict['date'] + '-' +
                               dataset_dict['name'] + '-merged_traces.h5')
        self._load_traces(h5_path)
        if ntrials is None:
            #ntrials = len(self.led_frames) - 2 ### USE THIS FOR COVARIATE LOADING
            ntrials = len(self.led_frames) - 1
        self.ntrials = ntrials

        # Load up the behavior file (a mat file).
        if 'bpod_file' in dataset_dict and dataset_dict['bpod_file']:
            print('(2/3) Loading behavior data.')
            bpod_data_fullpath = os.path.join(dataset_dict['behavior_dir'],
                                              dataset_dict['bpod_file'])
            self._load_behavior(bpod_data_fullpath, plot=behavior_plots, ntrials=ntrials)
            self.loaded_behavior = True
        else:
            self.bd = None

        # Load ttl file (a pickle file).
        if 'ttl_file' in dataset_dict and dataset_dict['ttl_file']:
            print('(3/3) Loading TTL data.')
            ttl_path = os.path.join(self.root_path, dataset_dict['date'],
                                    dataset_dict['name'],
                                    dataset_dict['ttl_file'])
            self.ttl_times, self.trial_onset_frames = cio.load_ttl_times(
                ttl_path, debug_plot=ttl_plots)
            self.loaded_ttl = True

        # Determine frame timing for the dataset.
        if self.loaded_behavior:
            self.fps = np.nanmean((self.led_frames[1:20]-self.led_frames[0]) /
                                  self.bd.trial_start_times[1:20])
            self.dt = 1.0/self.fps
            self.event_frames = self.fps*np.array([0,
                                                   self.bd.stimulus_times[0],
                                                   self.bd.stimulus_times[0]+1.5])
        else:
            # warnings.warn('Bpod file not loaded: hardcoding dt.')
            print('Bpod file not loaded: hardcoding dt.')
            self.dt = 0.034
            self.fps = 1/self.dt
            self.event_frames = []

        # Check validity of dataset.
        self.validate_dataset()

        # Do further processing/organization of traces.
        print('Processing traces.')
        atlas_assignments = self._assign_cells_to_regions(do_plot=do_region_plots,
                                                          use_parent=use_parent)

        self.cells_in_region = atlas_assignments[0]
        self.region_of_cell = atlas_assignments[1]
        self.hemisphere_of_cell = self._assign_hemisphere_of_cell(
                                                    do_plot=False)
        self.regions = atlas_assignments[2]

        self.nt_per_trial = nt_per_trial
        if do_reshape_to_traces:
            self._reshape_traces_to_trials(self.dt, nt=self.nt_per_trial,
                                           ntrials=self.ntrials) # Sets Ct,Ft,St,Tt
            self.St_smooth = gaussian_filter1d(self.St, s_kernel_size,
                                               axis=1, mode='constant')

        # Save a list of brain regions with at least threshold traces
        populated_areas = []
        for area in self.regions.keys():
            if (len(self.cells_in_region[self.regions[area]]) >
                    min_area_count):
                populated_areas.append(area)
        self.populated_areas = populated_areas

        self.covars = None

    def validate_dataset(self):

        # We can be off by one trial (in case we stopped things out of order?)
        max_offset = 1

        # These numbers should be the same:
        print('LED trials:', len(self.led_frames))
        if self.loaded_behavior and self.loaded_ttl:
            print('TTL trials:', len(self.ttl_times['trials']),
                  'Bpod trials:', self.bd.ntrials)
            nt = len(self.ttl_times['trials'])
            lt = len(self.led_frames)
            if (np.abs(lt - nt) > max_offset or (
                    np.abs(lt - self.bd.ntrials) > max_offset)):
                    raise ValueError('Inconsistent number of trials detected!')
        else:
            print('Warning: trial onsets only determined with LED times!')

    def centroids_on_atlas(self, cell_weights, cell_ids,
                           atlas_outline=None, max_radius=100,
                           set_alpha=False, set_radius=True,
                           highlight_inds=None):
        """
        Wrapper function for utils.centroid_on_atlas that takes care
        of passing in relevant class variables.

        Plots the centroid of each cell atop an atlas outline,
        where the size and color of each cell marker is determined
        by a cell_weights vector.

        :param cell_weights: ndarray [ncells]. Weight corresponding
                             to each cell.
        :param cell_ids: ndarray [ncells]. Indices of the cells to be shown.
        :param atlas_outline: Provide if preloading the atlas
                              using reg.load_atlas() (may be useful
                              if calling this function many times
                              i.e. to generate a movie).
                              Otherwise, will do automatically.
        :param max_radius: Maximum radius of plotted circles
        :return:
        """
        if atlas_outline is None:
            _, _, atlas_outline = reg.load_atlas()

        utils.centroids_on_atlas(cell_weights, cell_ids,
                                 self.centroids, self.atlas_tform,
                                 atlas_outline=atlas_outline,
                                 max_radius=max_radius,
                                 set_alpha=set_alpha,
                                 set_radius=set_radius,
                                 highlight_inds=highlight_inds)


    def plot_variance_shaded_traces(self, avgs, errs,
                                    cell_inds=None,
                                    footprints=None,
                                    ordering=None,
                                    do_normalize=True,
                                    title=None,
                                    alpha=None):
        """
        Wrapper function for utils.plot_variance_shaded_traces
        that takes care of passing in the relevant class variables.

        :param avgs: [cells x frames] array of mean trace
        :param errs: [cells x frames] array of variance across trials
        :param cell_inds: np.array containing indices of cells to plot
                         (after ordering). i.e. np.arange(0, 40)
        :param footprints: [X x Y x cells] footprints.
                           If not None, will plot contours corresponding
                           to plotted cells.
        :param ordering: None for no ordering.
                         'peak' for ordering by peak time.
                         'var' for ordering by variance.
                         If an np.array is is provided,
                         then order according to that.
                         See utils.plot_variance_shaded_traces for options.
        :param do_normalize:  Normalize each trace by max.
        :param title: (optional) Title for the plot.
        :param alpha: (optional) np.array. assign an alpha value to each
                      plotted contour.
        :return:
        """

        bpod_data = self.bd
        dt = self.dt
        if footprints is not None:
            atlas_outline = self.atlas_outline
            mean_image = self.mean_image
        else:
            atlas_outline = None
            mean_image = None

        utils.plot_variance_shaded_traces(avgs, errs,
                                          cell_inds=cell_inds,
                                          footprints=footprints,
                                          atlas_outline=atlas_outline,
                                          mean_image=mean_image,
                                          bpod_data=bpod_data,
                                          ordering=ordering,
                                          do_normalize=do_normalize,
                                          title=title,
                                          dt=dt,
                                          alpha=alpha)

    def plot_cell_across_trial_types(self, cell,
                                     frame_range, range_mean, range_sem,
                                     trial_sets, names, colors, pvals):
        """
        Wrapper function for utils.plot_cell_across_trial_types that takes
        care of passing in relevant class variables.
        Plots response of a specified cell to different trial conditions.

        :param cell: int. overall index of cell.
        :param frame_range: list. [start_frame, end_frame]
        :param range_mean: [ncells x ntrial_types]
        :param range_sem: [ncells x ntrial_types]
        :param trial_sets: tuple of bool arrays. (type1, type2, type3, ...)
                           where type1 is a boolean np.array of length ntrials
                           showing whether each trial is part of a trial type.
        :param names: tuple of strings. (type1_name, type2_name, ...)
        :param colors: tuple of strings. Color for plotting each trial type.
        :param pvals: [ncells], pvalue computed using
                      get_trial_type_selective_cells().
        :return:
        """

        utils.plot_cell_across_trial_types(cell,
                                           self.Ft, #self.St_smooth,
                                           self.footprints,
                                           self.Tt, self.fps, self.mean_image,
                                           self.atlas_outline,
                                           frame_range, range_mean, range_sem,
                                           trial_sets, names, colors, pvals)

    def plot_raster_by_region(self, traces=None,
                              nframes=3000,
                              startframe=0,
                              which_regions=['MO', 'PTLp', 'RSP', 'SSp', 'VIS'],
                              event_frames=None):
        """
        Wrapper function for utils.plot_raster_by_region to
        incorporate all class variables.

        Raster plot (each row is a the trace of a cell) of the specified
        traces matrix, ordered such that cells are grouped by region.

        :param traces: np.array. [ncells x nt] matrix to plot.
        :param nframes: int. number of frames to plot.
        :param startframe: int. first frames to plot.
        :param which_regions: list. abbreviations of regions to include.
        :param event_frames: (optional). List of frames at which to plot
                             a vertical line.
        """
        if traces is None:
            traces = self.C

        utils.plot_raster_by_region(self.cells_in_region,
                                    self.regions,
                                    self.hemisphere_of_cell,
                                    self.dt,
                                    traces=traces,
                                    nframes=nframes,
                                    startframe=startframe,
                                    which_regions=which_regions,
                                    event_frames=event_frames)

    def load_covariates(self, do_plot=False, do_debug=False):
        if self.regressors_name is not None:
            self.covars = {}
            covariate_dir = os.path.join(self.root_path, 'behavior_videos',
                                         self.date, self.regressors_name)

            ### Load previously saved out regressors.
            upper_camera_path = os.path.join(covariate_dir,
                                             self.date + '_'
                                             + self.regressors_name
                                             + '_upper_camera_regressors.pkl')
            with open(upper_camera_path, 'rb') as f:
                upper_camera_data = pickle.load(f)

            lower_camera_path = os.path.join(covariate_dir,
                                             self.date + '_'
                                             + self.regressors_name
                                             + '_lower_camera_regressors.pkl')
            with open(lower_camera_path, 'rb') as f:
                lower_camera_data = pickle.load(f)

            ### Rescale regressors and shape them organize them by trials.
            for key in ['left_whisker_energy', 'right_whisker_energy']:
                r_covariate, r_led_frames = utils.rescale_covariates(
                                                lower_camera_data[key],
                                                lower_camera_data['led_frames'],
                                                self.led_frames, do_debug=do_debug)
                covar_trials = utils.reshape_to_trials(r_covariate,
                                                       r_led_frames,
                                                       nt=self.nt_per_trial,
                                                       dt=self.dt,
                                                       ntrials=self.ntrials)
                self.covars[key] = covar_trials

            for key in ['upper_body_motion_energy', 'upper_face_motion_energy']:
                r_covariate, r_led_frames = utils.rescale_covariates(
                                                    upper_camera_data[key],
                                                    upper_camera_data['led_frames'],
                                                    self.led_frames, do_debug=do_debug)
                covar_trials = utils.reshape_to_trials(r_covariate,
                                                       r_led_frames,
                                                       nt=self.nt_per_trial,
                                                       dt=self.dt,
                                                       ntrials=self.ntrials)
                self.covars[key] = covar_trials

            self.lower_behavior_camera_led_frames = lower_camera_data['led_frames']
            self.upper_behavior_camera_led_frames = upper_camera_data['led_frames']

            if do_plot:
                for key in self.covars.keys():
                    plt.figure()
                    plt.title(key)
                    plt.imshow(self.covars[key][0, :, :].T)
        else:
            print('No regressors specified for this dataset: {}/{}'.format(self.date, self.name))

    # Private functions.
    def _load_traces(self, traces_hdf5):
        with h5py.File(traces_hdf5, "r") as hf:

            # Save traces
            self.S = np.array(hf['spikes'])
            self.C = np.array(hf['tseries'])
            self.F = np.array(hf['tseries_raw'])
            
            # Save information about traces
            self.footprints = np.array(hf['footprints'])
            self.mean_image = np.squeeze(
                np.array(hf['mean_frames'])[:, :, 0])
            self.mean_timecourse = np.squeeze(
                np.array(hf['mean_frames'])[:, :, 1])
            self.led_frames = np.array(hf['led_frames'])
            self.centroids = np.array(hf['cm'])
            self.ncells = self.centroids.shape[0]

            # Deal with the atlas if necessary
            if 'atlas_info' in hf.keys():

                # Save atlas values
                atlas_info = hf['atlas_info']
                self.atlas = np.array(atlas_info['atlas'])
                self.atlas_outline = np.array(
                    atlas_info['aligned_atlas_outline'])

                # Load annotations and recompute atlas alignment transform.
                atlas_coords = np.array(atlas_info['atlas_coords'])
                self.img_coords = np.array(atlas_info['coords'])
                _, self.atlas_annotations, _ = reg.load_atlas()
                self.atlas_tform = reg.fit_atlas_transform(self.img_coords,
                                                           atlas_coords)

    def _load_behavior(self, behavior_path, plot=True, ntrials=None):
        self.bd = BpodDataset(behavior_path, self.fig_save_path, ntrials=ntrials)

        if plot:
            self.bd.plot_spout_selectivity(min_trial=10)
            self.bd.plot_lick_times()
            self.bd.plot_success()

    # def _load_behavior(self, behavior_path, plot=True):
    #     self.bd = BpodDataset(behavior_path, self.fig_save_path)
    #
    #     if plot:
    #         self.bd.plot_spout_selectivity(min_trial=10)
    #         self.bd.plot_lick_times()
    #         self.bd.plot_success()

    def _reshape_traces_to_trials(self, dt, nt, ntrials=None):
        """
        Reshapes traces to [cells x time x trials],
        and saves to new class variables.

        :param nt: Minimum seconds of each trial
        :param dt: Time in seconds of each frame.
        :return: sets self.Ct: [cells x time x trial] Smoothed traces.
                      self.Ft: [cells x time x trial] Raw traces.
                      self.St: [cells x time x trial] Deconvolved spikes.
                      self.Tt: [time] Time in seconds within trial.
        """
        self.Ct = utils.reshape_to_trials(self.C, self.led_frames,
                                          nt, dt, ntrials=ntrials)
        self.Ft = utils.reshape_to_trials(self.F, self.led_frames,
                                          nt, dt, ntrials=ntrials)
        St = utils.reshape_to_trials(self.S, self.led_frames,
                                     nt, dt, ntrials=ntrials)
        self.St = St
        # self.St = (St > 0).astype(float) ### COMMENTED THIS OUT ON 2019.07.22
        self.Tt = np.arange(self.Ct.shape[1]) * dt

    def _assign_hemisphere_of_cell(self, do_plot=True):
        """
        Assigns a label to each cell indicating whether
        it is on the right or left hemisphere of the brain.

        :return hemisphere_of_cell: binary np.array.
                                    TODO: Determine whether
                                    1 indicates right or
                                    left hemisphere.
        """
        # Find midline based on keypoints used for atlas alignment.
        x_coords = self.img_coords[:, 0]
        y_coords = self.img_coords[:, 1]
        print(x_coords)
        print(y_coords)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        xhat = self.centroids[:, 1]
        yhat = self.centroids[:, 0]
        yclass = m * xhat + c

        # Classify cells based on whether above or below midline.
        left_h_idx = np.where(yhat >= yclass)[0]
        right_h_idx = np.where(yhat < yclass)[0]
        if do_plot:
            plt.plot(x_coords, y_coords, 'k')
            for c, idx in zip(['r', 'b'], [left_h_idx, right_h_idx]):
                plt.plot(self.centroids[idx, 1], self.centroids[idx, 0],
                         c + '.')

        hemisphere_of_cell = np.zeros((self.centroids.shape[0], 1))
        hemisphere_of_cell[right_h_idx] = 1

        return hemisphere_of_cell

    def _assign_cells_to_regions(self, do_plot=True, use_parent=True):
        """
        Assigns cells (and their traces) to
        cortical regions based on the atlas and the xy coordinates
        of the center of mass of each cell.

        Relies on self.cm, self.tform, self.atlas, self.annotations
        having been previously loaded.

        Returns an array with the atlas region (i.e. ID number)
        corresponding to each cell.
         Given an array of xy_coordinates (i.e. the center of mass
        of each cell's ROI), return an array with the atlas region
        corresponding to each cell, as well as a dict which contains
        an entry for each region indicating the cells that are in that
        region.
        :param use_parent:
        :return: atlas assignments. A tuple which contains two objects,
                 atlas_assignments[0]: cells_in_region:
                                       a dict which contains an entry for
                                       each region, indicating the cells
                                       that are in that region.
                 atlas_assignments[1]: region_of_cells:  an array with the
                                       atlas region ID for each cells.
                 atlas_assignments[2]: regions: a dict that maps region
                                       acronym to region id.
                                       Can get cells in a region i.e. by
                                       calling cells_in_region[regions['AUD']].
        """
        c_in_r, r_of_c = reg.assign_cells_to_regions(np.fliplr(self.centroids),
                                                     self.atlas_tform,
                                                     self.atlas,
                                                     self.atlas_annotations,
                                                     get_parent=use_parent,
                                                     do_debug=False)
        regions = dict()
        for i in c_in_r.keys():
            name = self.atlas_annotations[str(i)]['acronym'].decode("utf-8")
            regions[name] = i

        if do_plot:
            nregions = len(c_in_r.keys())
            if use_parent:
                plt.figure(figsize=(10, 10))
            else:
                plt.figure(figsize=(15, 30))
            for ind, region in enumerate(c_in_r.keys()):
                plt.subplot(np.ceil(nregions/3.0), 3, ind+1)
                plt.imshow(reg.overlay_atlas_outline(self.atlas_outline,
                                                     self.mean_image),
                           cmap='gray')
                plt.axis('off')
                plt.title(str(self.atlas_annotations[str(region)]['acronym']) +
                          " \n ncells:" + str(len(c_in_r[region])))
                for cell in c_in_r[region]:
                    plt.plot(self.centroids[cell, 1],
                             self.centroids[cell, 0],
                             'ro', markersize=3)

        atlas_assignments = (c_in_r, r_of_c, regions)
        return atlas_assignments

    def check_ttl_pulses(self, frames_to_plot=5000, end_offset=1000):
        # Check if the TTL pulses line up
        plt.figure(figsize=(10, 2))
        for xx in [1, 2]:
            plt.subplot(1, 2, xx)
            plt.plot(self.led_frames,
                     [0.9]*len(self.led_frames), '.', label='led')
            plt.plot(self.trial_onset_frames,
                     [1]*len(self.trial_onset_frames), '.', label='ttl')
            plt.ylim([.4, 1.5])
            if xx == 1:
                plt.xlim([0, frames_to_plot])
                plt.title('start')
            else:
                plt.xlim([np.max(self.led_frames) - frames_to_plot,
                         np.max(self.led_frames)+end_offset])
                plt.title('end')
                plt.legend()
            plt.xlabel('frame')
            plt.ylabel('AU')
