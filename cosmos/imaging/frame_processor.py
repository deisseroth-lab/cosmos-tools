import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import time
import tifffile
import peakutils  # can install with conda.
from subprocess import call
import scipy.signal
from scipy.ndimage.filters import gaussian_filter
import sys

import cosmos.imaging.img_io as iio
import cosmos.imaging.atlas_registration as reg


class FrameProcessor:
    """A processor for extracting neural traces from raw
    frames of a single experimental dataset.

    Attributes:
        raw_data_dir: str. Directory where all raw data is stored.
        processed_data_dir: str. Directory where all processed data is stored.
        date: str. The date of the experiment.
        name: str. The name of the dataset.
        mean_img: ndarray. The mean image across all frames of the dataset.
        crop_coords: ndarray. Rectangle coordinates for each crop region.
        mean_img_cropped: ndarray. Mean image for each cropped region.
        led_frame_inds: ndarray. Indices of frames with trial-start indicating LED flash.
        alignment_keypoints: ndarray. Keypoints for aligning cropped regions to each other.

    NOTE: For ome.tif, this only seems to work with python2.7
    """

    def __init__(self, raw_data_dir, processed_data_dir, dataset):
        """
        :param raw_data_dir: Computer-specific directory where raw data is stored.
        :param processed_data_dir: Directory where processed data is stored.
        :param dataset: A dict containing the 'date' and 'name' of a dataset.
                        Dataset raw files will be found at the location:
                            raw_data_dir/date/name
                        Processed files will be stored at the location:
                            processed_data_dir/date/name
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.date = dataset['date']
        self.name = dataset['name']
        self.mean_img = None
        self.crop_coords = None
        self.mean_img_cropped = None
        self.led_frame_inds = None
        self.alignment_keypoints = None
        self.vid_names = ['top', 'bot']
        # self.vid_names = ['top']
        self.nroi = len(self.vid_names)


        ### Protected attributes.
        self._data_path = os.path.join(self.raw_data_dir, self.date, self.name)
        self._out_path = os.path.join(self.processed_data_dir, self.date, self.name)
        self._roi_coords_file = os.path.join(self._out_path,'roi_coords.npz')
        self._led_frame_file = os.path.join(self._out_path,'led_frames.npz')
        self._keypoints_file = os.path.join(self._out_path)
        self._shifts_file = os.path.join(self._out_path,'shifts.npz')

    def select_crop_rois(self):
        """
        Provides a GUI interface for selecting a provided
        number of square regions of interest (ROIs).
        Saves the coordinates for cropping to a cropInfo.npz
        file for each ROI, as well as an overall file
        allCropCoords.npz for subsequent image plotting.
        Also saves out an image of the ROIs overlaid on a mean image.
        :return: crop_coords: Crop coordinates for each ROI.
        """

        # Load up the raw data from individual .tif files
        print('Loading: ', self._data_path)
        stack = iio.load_raw_data(self._data_path, sub_seq=range(5), print_freq=1)

        # Manually select ROIs
        status = 'not done'
        mean_img = np.log10(np.mean(stack, 0)+1)
        # loc_names = [str(i) for i in range(self.nroi)]
        loc_names = self.vid_names
        output_folder = self._out_path
        while status:
            slices = self.__get_roi(mean_img, self.vid_names, output_folder)
            if (sys.version_info > (3, 0)):
                status = input('Press enter if ROIs look good. Otherwise say no:')
            else:
                status = raw_input('Press enter if ROIs look good. Otherwise say no:')
        np.savez(self._roi_coords_file, coords=slices, loc_names=loc_names)

        fig = plt.figure()
        for idx, slice_ in enumerate(slices):
            ax = fig.add_subplot(1, self.nroi, idx + 1)
            ax.imshow(mean_img[slice_[0], slice_[1]])
            # plt.title(loc_names[idx] + ' ' + self.vid_names[idx])
            plt.title(self.vid_names[idx])
        plt.savefig(self._out_path + '/roi_summary_indiv.pdf')

        # with open(self._roi_coords_file, 'w') as text_file:
        #     text_file.write(str(slices))

    def crop_stack(self, do_remove_led_frames=False, n_frames=None,
                   do_motion_correct=True, LED_buffer=2, led_std=3):
        """
        Loads the crop rois that were saved out using select_crop_rois().
        Loads the raw video. Saves out to a bigtiff file.
        Note: to import a bigtiff into ImageJ. Use File->Import->BioFormats
        :param do_remove_led_frames: bool. If true, will find indices
                    of frames where LED turns on. When saving out video
                    these frames will be replaced by neighboring frames.
                    Additionally, will save out a file that contains
                    the LED frame times.
        :param LED_buffer: number of frames before and after LED frame to remove.
        :param led_std: float how many standard deviations above baseline to use
                        as threshold for finding an led frame.

        :return: Nothing.
        """
        if os.path.isfile(self._roi_coords_file):
            coord_file = np.load(self._roi_coords_file)
        else:
            raise FileNotFoundError('ROI coordinates file has not been generated.')
        slices = coord_file['coords']
        # loc_names = coord_file['loc_names']
        loc_names = self.vid_names
        print(slices)

        # Load up the raw data from individual .tif files
        stack = iio.load_raw_data(self._data_path, sub_seq=None, print_freq=100)

        if do_remove_led_frames:
            print('Finding LED frames.')
            ### Find indices of LED frames.
            ledx = slice(10,100)
            ledy = slice(700,900)
            avg_trace = np.squeeze(np.mean(np.mean(stack[:, ledx, ledy], 2), 1))
            # Only look at a segment in the middle - in case light was
            # turned off at the beginning or end of acquisition.
            trace_segment = avg_trace[int(len(avg_trace)/4):int(len(avg_trace)/2)]
            # threshed_avg = (avg_trace > (np.std(trace_segment)*0.75 + \
            #                              np.percentile(trace_segment, 10))).astype(int)
            threshed_avg = (avg_trace > (np.std(trace_segment)*led_std + # Changed this from scale=2 on 20190710
                                         np.percentile(trace_segment, 10))).astype(int)
            led_peak_frames = peakutils.indexes(threshed_avg, min_dist=30)
            np.savez(self._led_frame_file, inds=led_peak_frames)

            ### Replace LED frames with preceding frame.
            for ii, frame in enumerate(led_peak_frames):
                # stack[frame-3:frame+3, :,:] = np.tile(stack[frame-4,:,:], (6, 1, 1))

                # stack[frame-1,:,:] = (stack[frame-2,:,:]*3.0/4 \
                #                      + stack[frame+2,:,:]*1.0/4)
                # stack[frame,:,:] = (stack[frame-2,:,:]*2.0/4 \
                #                    + stack[frame+2,:,:]*2.0/4)
                # stack[frame+1,:,:] = (stack[frame-2,:,:]*1.0/4 \
                #                      + stack[frame+2,:,:]*3.0/4)

                nf = LED_buffer ## nframes before and after LED ### NORMALLY set to 2
                ind = 1.0
                for ff in np.arange(-nf+1, nf):
                    stack[frame+ff, :, :] = (stack[frame-nf,:,:]*(nf*2.0-ind)/(nf*2.0) ### THERE WAS A BUG HERE!
                                             + stack[frame+nf, :, :]*(ind)/(nf*2.0))
                    ind = ind + 1
            avg_trace_new = np.squeeze(np.mean(np.mean(stack[:, ledx, ledy], 2), 1))

            ### Plot average trace before and after removal.
            fig = plt.figure()
            p1, = plt.plot(avg_trace)
            plt.plot(led_peak_frames, avg_trace[led_peak_frames], 'ro')
            p2, = plt.plot(avg_trace_new, 'g')
            plt.legend([p1, p2], ['orig', 'filtered'])
            plt.savefig(self._out_path + '/led_peaks.pdf')

        ### Save out each ROI to a separate bigtiff file.
        stack_depth = np.shape(stack)[0]
        if n_frames is not None:
            stack_depth = min(n_frames, stack_depth)
            ### THIS NEEDS TO BE TESTED.

        for name, slice_ in zip(loc_names, slices):
            print('Saving out ROI: ' + name)
            roi_dir = os.path.join(self._out_path, str(name))
            if not os.path.exists(roi_dir):
                os.makedirs(roi_dir)
            roi_path = os.path.join(roi_dir, str(name)+'.tif')
            print('Saving ' + roi_path)
            t0 = time.time()
            with tifffile.TiffWriter(roi_path, bigtiff=True) as tif:
                template = stack[0, slice_[0], slice_[1]]
                for idx in range(stack_depth):
                    frame = stack[idx, slice_[0], slice_[1]]
                    if do_motion_correct:
                        shiftx, shifty = self.get_averaged_motion(template, frame)
                        shifted_frame = self.shift_frame(frame, shiftx, shifty)
                    else:
                        shifted_frame = frame

                    tif.save(shifted_frame)

            print('Saved. Took ', time.time() - t0, ' seconds.')

    @staticmethod
    def shift_frame(frame, shiftx, shifty):
        """
        Shift an image (i.e. frame of a video) according
        shiftx and shifty.
        :param frame: [time x X x Y]
        :param shiftx: [time]. Output from get_motion()
        :param shifty: [time]. Output from get_motion()
        :return: shifted_stack - motion corrected stack.
        """
        frame = np.roll(frame, -int(np.floor(shifty)), axis=0)
        frame = np.roll(frame, -int(np.floor(shiftx)), axis=1)
        return frame

    def get_averaged_motion(self, template, frame):
        """
        Get the linear motion shift across the video
        averaged across a few cropped regions.

        :param template:
        :param frame: Frame that will be shifted to match template
        :return: shiftsx, shiftsy
        """

        stack = np.stack((template, frame), axis=0)

        # Compute shifts in x and y directions
        crop1 = stack[:, 200:300, 200:300]
        crop2 = stack[:, 200:300, 400:500]
        crop3 = stack[:, 400:500, 200:300]
        crop4 = stack[:, 400:500, 400:500]
        # crops = (crop1, crop2, crop3, crop4)
        crops = (crop1, crop3) # It is faster and less noisy to only use two crop regions.

        shiftsx = np.zeros((len(crops), stack.shape[0]))
        shiftsy = np.zeros((len(crops), stack.shape[0]))

        for crop_ind, crop in enumerate(crops):
            # print('Crop #{}'.format(crop_ind))
            shiftx, shifty, template_crop, target_crop = self.get_motion(crop)

            shiftsx[crop_ind, :] = shiftx[:, 0]
            shiftsy[crop_ind, :] = shifty[:, 0]

        shiftx = np.mean(shiftsx[:, 1])
        shifty = np.mean(shiftsy[:, 1])
        return shiftx, shifty

    @staticmethod
    def get_motion(crop):
        """
        Get the linear motion shift across the video
        for a single cropped region.
        Uses autocorrelation based approach.

        :param crop: A cropped video [time x X x Y]
        :return: shiftx, shifty: the pixel shift at each
                 time step of the video, relative to the first
                 frame, in the x and y directions.
        """

        template = crop[0, :, :].astype(float)
        template = template / np.max(template)
        template = gaussian_filter(template, sigma=1) \
                   - gaussian_filter(template, sigma=2)

        nframes = crop.shape[0]
        if nframes < 1:
            raise(ValueError, 'In get_motion(), crop.shape[0] must be > 0. Shape is currently: {}'.format(crop.shape))

        x_c = np.zeros((nframes, 1))
        y_c = np.zeros((nframes, 1))

        target = np.zeros(template.shape)
        for ind in range(nframes):
            target = crop[ind, :, :].astype(float)
            target = target / np.max(target)

            target = gaussian_filter(target, sigma=1) \
                     - gaussian_filter(target, sigma=2)

            a = scipy.signal.correlate(target, template, mode='same')
            y_c[ind], x_c[ind] = np.unravel_index(a.argmax(), a.shape)

        shiftx = x_c - x_c[0]
        shifty = y_c - y_c[0]

        return shiftx, shifty, template, target

    def plot_motion(self):
        """
        Plots a metric of motion across the time series.
        Uses spatial correlation of a set of cropped regions.
        Uses first frame as template, to which subsequent frames are compared.
        :return:
        """
        for name in ['top']:
            roi_dir = os.path.join(self._out_path, str(name))
            roi_path = os.path.join(roi_dir, str(name) + '.tif')

            try:
                ff = np.load(self._led_frame_file)
                led_frame_inds = ff['inds']
                # sub_seq = range(0, np.max(led_frame_inds), 2) # Load every other frame (faster).
                sub_seq = None
            except:
                sub_seq = None

            vid = iio.load_raw_data(roi_dir, sub_seq=sub_seq, print_freq=100)

            # Crop out subregions.
            crop1 = vid[:, 200:300, 200:300]
            crop2 = vid[:, 200:300, 400:500]
            crop3 = vid[:, 400:500, 200:300]
            crop4 = vid[:, 400:500, 400:500]
            crops = (crop1, crop2, crop3, crop4)

            shifts = np.zeros((len(crops), vid.shape[0]))

            for crop_ind, crop in enumerate(crops):
                print('Crop #{}'.format(crop_ind))
                shiftx, shifty, template, target = self.get_motion(crop)

                shift = np.sqrt((shiftx**2)+(shifty**2))
                shifts[crop_ind, :] = shift[:, 0]

                plt.figure()
                plt.plot(shift)
                plt.ylabel('Pixel shift')
                plt.xlabel('Time [frames]')
                plt.title('Crop {}'.format(crop_ind))
                plt.savefig(self._out_path + '/shift_crop{}.pdf'.format(crop_ind))

                plt.figure(100, figsize=(len(crops)*5, 5))
                plt.subplot(1, len(crops), crop_ind+1)
                plt.imshow(template)
                plt.title(str(crop_ind))
                plt.suptitle('Templates')

                plt.figure(101, figsize=(len(crops)*5, 5))
                plt.subplot(1, len(crops), crop_ind+1)
                plt.imshow(target)
                plt.title(str(crop_ind))
                plt.suptitle('Targets')

        print('Done saving crops')
        plt.figure()
        plt.plot(shifts.T, alpha=0.5)
        plt.plot(np.mean(shifts, axis=0), 'k')
        plt.ylabel('Pixel shift')
        plt.xlabel('Time [frames]')
        plt.title('Average pixel shift')
        print('Plotting average shifts.')
        plt.savefig(self._out_path + '/shift_average.pdf')

        plt.figure(100)
        plt.savefig(self._out_path + '/shift_templates.png')

        plt.figure(101)
        plt.savefig(self._out_path + '/shift_targets.png')

        np.savez(self._shifts_file, inds=shifts)

    def run_cnmfe(self,
                  matlab_path='~/Matlab/bin/matlab -softwareopengl',
                  nthreads=3):
        """
        Call matlab version of CNMF-e on each of the cropped ROIs.
        :return: void.
        """
        cnmfe_function = 'run_cnmfe_cosmos'

        # input_folders = [os.path.join(self._out_path, str(x), str(x) + '.tif') \
        #                  for x in range(self.nroi)]
        input_folders = [os.path.join(self._out_path, str(x), str(x) + '.tif') \
                         for x in self.vid_names]

        for idx, inp in enumerate(input_folders):
            out_name = self.vid_names[idx]
            print('Processing ', idx)
            print('in:  ', inp)

            # Start MATLAB and call the cnmf-e function.
            # call(matlab_path + ' -nodesktop -nosplash -r \"' + cnmfe_function +
                 # '(\'' + inp + '\'); exit;\"', shell=True)
            # print(matlab_path + ' -nodesktop -nosplash -r \"' + cnmfe_function +
            #      '(\'' + inp + '\', \'' + out_name + '\'); exit;\"')
            # cmd = (matlab_path + ' -softwareopengl -nodesktop -nosplash -r \"' + cnmfe_function +
            #      '(\'' + inp + '\', \'' + out_name + '\', \'num_threads\', ' + str(nthreads) + '); exit;\"')
            cmd = (matlab_path + ' -softwareopengl -nodesktop -nosplash -r \"' + cnmfe_function +
                 '(\'' + inp + '\', \'' + out_name + '\', ' + str(nthreads) + '); exit;\"')

            call(cmd, shell=True)
            # call(matlab_path + ' -softwareopengl -nodesktop -nosplash -r \"' + cnmfe_function +
            #      '(\'' + inp + '\', \'' + out_name + '\'); exit;\"', shell=True)

    def get_alignment_keypoints(self):
        """
        This function is DEPRECATED - you should use atlas_align instead.
        Provides GUI for selecting alignment
        keypoints for an image.
        Saves out the results to directories
        specified in the FrameProcessor constructor.
        :return: None
        """
        raise('Use atlas_align() instead of this get_alignment_keypoints().')

        if os.path.isfile(self._roi_coords_file):
            coord_file = np.load(self._roi_coords_file)
        else:
            raise FileNotFoundError('ROI coordinates file has not been generated.')
        slices = coord_file['coords']
        # loc_names = coord_file['loc_names']

        # for name in loc_names:
        for name in self.vid_names:
            roi_dir = os.path.join(self._out_path, str(name))
            roi_path = os.path.join(roi_dir, str(name) + '.tif')

            vid = iio.load_raw_data(roi_dir, sub_seq=range(10), print_freq=100)

            patch_coords = []

            print(vid.shape)
            mean_img = np.squeeze(np.mean(vid, axis=0))
            plt.figure(figsize=(30, 30))
            plt.imshow(mean_img)
            plt.title('Click on anterior midline')
            c = plt.ginput(n=1)
            plt.plot(c[0][0], c[0][1], 'ro')
            patch_coords.extend(c)

            plt.title('Click on posterior midline (lambda)')
            c = plt.ginput(n=1)
            plt.plot(c[0][0], c[0][1], 'ro')
            patch_coords.extend(c)

            plt.title('Click on right anterior/lateral corner')
            c = plt.ginput(n=1)
            plt.plot(c[0][0], c[0][1], 'ro')
            patch_coords.extend(c)

            plt.title('Click on left anterior/lateral corner')
            c = plt.ginput(n=1)
            plt.plot(c[0][0], c[0][1], 'ro')
            patch_coords.extend(c)

            plt.close()

            ### Save out selections
            patch_coords_array = np.zeros((len(patch_coords), 2))
            for ci, c in enumerate(patch_coords):
                patch_coords_array[ci, 0] = np.round(c[0])
                patch_coords_array[ci, 1] = np.round(c[1])

            print(patch_coords_array)
            keypoints_dir = os.path.join(self._keypoints_file, name, \
                                         str(name)+'_source_extraction')
            save_fname =os.path.join(keypoints_dir, 'keypoints.npz')
            np.savez(save_fname, coords=patch_coords_array)

            plt.figure()
            plt.imshow(mean_img)
            for ci, c in enumerate(patch_coords):
                print(c)
                plt.plot(c[0], c[1], 'ro')
            plt.savefig(os.path.join(keypoints_dir, 'keypoints.png'))

    def atlas_align(self):
        """
        Provides a limited GUI for aligning image to atlas, as well
        as selecting keypoints to align the two sub-images.
        Saves out the results to directories
        specified in the FrameProcessor constructor.
        # :param atlas_path: Full path to atlas_top_projection.mat file containing
        #                    output from process_atlas_script.m
        :return: Saves out keypoints.npz containing aligned atlas and keypoints
        """
        print("Atlas alignment.")
        if (sys.version_info > (3, 0)):
            text = input('Press enter to begin quick atlas alignment. Do not worry about precision, this can be refined later: ')
        else:
            text = raw_input('Press enter to begin quick atlas alignment. Do not worry about precision, this can be refined later: ')

        if os.path.isfile(self._roi_coords_file):
            coord_file = np.load(self._roi_coords_file)
        else:
            raise FileNotFoundError('ROI coordinates file has not been generated.')

        for name in self.vid_names:
            plt.close('all')
            ### Load image.
            roi_dir = os.path.join(self._out_path, str(name))
            roi_path = os.path.join(roi_dir, str(name) + '.tif')
            #vid = iio.load_raw_data(roi_dir, sub_seq=None, print_freq=100)
            vid = iio.load_raw_data(roi_dir, sub_seq=range(1), print_freq=100)
            # img = np.mean(vid, axis=0)
            img = vid

            keypoint_positions = ['Anterior midline',
                                  'Posterior midline']
                                  # 'Right anterolateral corner',
                                  # 'Left anterolateral corner']

            ### Load atlas.
            atlas, annotations, atlas_outline = reg.load_atlas()
            do_manual_atlas_keypoint=False
            if do_manual_atlas_keypoint:
                atlas_coords = []
                plt.figure(figsize=(30, 30))
                plt.imshow(atlas_outline)
                for t in keypoint_positions:
                    plt.title('Click on ' + t)
                    c = plt.ginput(n=1)
                    plt.plot(c[0][0], c[0][1], 'ro')
                    atlas_coords.extend(c)
            else:
                atlas_coords = [(98, 227), #(83, 227)
                                (348, 227)]
                                # (83, 303),
                                # (83, 151)] ### These numbers were taken from an initial manual selection.
                plt.figure(figsize=(30,30))
                plt.imshow(atlas_outline)
                for cc in atlas_coords:
                    plt.plot(cc[0], cc[1], 'ro')

            ### Convert selected keypoints to array.
            atlas_coords_array = np.zeros((len(atlas_coords), 2))
            for ci, c in enumerate(atlas_coords):
                atlas_coords_array[ci, 0] = np.round(c[0])
                atlas_coords_array[ci, 1] = np.round(c[1])

            while 1:
                plt.figure()
                plt.imshow(img)
                img_coords = []
                do_manual_patch_keypoint = True
                if do_manual_patch_keypoint:
                    plt.figure(figsize=(30, 30))
                    plt.imshow(img)
                    for t in keypoint_positions:
                        plt.title('Click on ' + t)
                        c = plt.ginput(n=1)
                        plt.plot(c[0][0], c[0][1], 'ro')
                        img_coords.extend(c)
                else:
                    ### These numbers are just for fast debugging.
                    img_coords = [(26, 297),
                                  (430, 314)]
                                  # (26, 187),
                                  # (17, 420)]
                    plt.figure(figsize=(30, 30))
                    plt.imshow(img)
                    for cc in img_coords:
                        plt.plot(cc[0], cc[1], 'ro')

                plt.close('all')

                ### Convert selected keypoints to array.
                img_coords_array = np.zeros((len(img_coords), 2))
                for ci, c in enumerate(img_coords):
                    img_coords_array[ci, 0] = np.round(c[0])
                    img_coords_array[ci, 1] = np.round(c[1])

                aligned_atlas_outline, aligned_img, tform = reg.align_atlas_to_image(atlas_outline, img,
                                                                                     atlas_coords_array[0:2, :],
                                                                                     img_coords_array[0:2, :],
                                                                                     do_debug=False
                                                                                     )

                ### Overlay atlas on image for checking that things look good.
                overlay = reg.overlay_atlas_outline(aligned_atlas_outline, img)
                plt.figure(figsize=(20,20))
                plt.imshow(overlay)
                plt.title('Check that things look good, and close this window manually.')
                plt.show()

                if (sys.version_info > (3, 0)):
                    text = input('Look good? [y] or [n]')
                else:
                    text = raw_input('Look good? [y] or [n]')

                print(text)
                if text == 'y':
                    break

            ### Save out selections
            keypoints_dir = os.path.join(self._keypoints_file, name, \
                                         str(name) + '_source_extraction')
            save_fname = os.path.join(keypoints_dir, 'keypoints.npz')
            print('Saving keypoints and aligned atlas to: ' + save_fname)
            np.savez(save_fname,
                     coords=img_coords_array,
                     atlas_coords=atlas_coords,
                     atlas=atlas,
                     img=aligned_img,
                     aligned_atlas_outline=aligned_atlas_outline)

            plt.figure()
            plt.imshow(img)
            for ci, c in enumerate(img_coords):
                print(c)
                plt.plot(c[0], c[1], 'ro')
            plt.savefig(os.path.join(keypoints_dir, 'keypoints.png'))

            plt.figure()
            plt.imshow(overlay)
            plt.savefig(os.path.join(keypoints_dir, 'overlay.png'))

    @staticmethod
    def __get_roi(img, loc_names, output_folder):
        """
        The GUI interface for selecting regions of interest
        in a provided image.
        :param img: ndarray. The image to be used for selecting ROIs.
        :param loc_names: list. Name describing each ROI. (i.e. ['top', 'bot'])
        :param output_folder: str. Location to save out coordinates.
        :return:
        """

        if not os.path.isdir(output_folder): os.makedirs(output_folder)
        nroi = len(loc_names)
        colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
        plt.rcParams['image.cmap'] = 'gray'

        # ROI selection callback.
        def onselect(eclick, erelease):
            print(loc_names[idx] + ' ROI selected.' +
                  ' Change or close window to continue.')

        # Make a new window to choose each ROI.
        selectors = []
        for idx in range(nroi):
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(1, 1, 1)

            ax.imshow(img)
            selector = RectangleSelector(ax, onselect, drawtype='box',
                                         interactive=True)
            plt.title('Choose ' + loc_names[idx] + ' ROI.')
            plt.show()
            selectors.append(selector)

        # Plot all selections.
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img)
        for idx, selector in enumerate(selectors):
            rect = plt.Rectangle([selector.corners[0][0], selector.corners[1][0]],
                                 selector.corners[0][2] - selector.corners[0][0],
                                 selector.corners[1][2] - selector.corners[1][0],
                                 fill=None, color=colors[idx])
            ax.add_patch(rect)
        plt.savefig(output_folder + '/roi_summary.pdf')
        plt.show()

        slices = []
        for selector in selectors:
            x0 = int(selector.corners[1][0]) - 1
            x0 = 0 if x0 < 0 else x0
            x = slice(x0, int(selector.corners[1][2]))
            x0 = int(selector.corners[0][0]) - 1
            x0 = 0 if x0 < 0 else x0
            y = slice(x0, int(selector.corners[0][2]))
            slices.append([x, y])
        return slices
