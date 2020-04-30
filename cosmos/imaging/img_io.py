#!/usr/bin/python
""""
A module for input and output of image stack files.
"""

import skimage.io
import glob
import numpy as np
import os
import time
import pickle
from collections import defaultdict

# Begin video imports
import pims
import moviepy.editor as mpy
import tifffile
from PIL import Image
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from moviepy.editor import *
# End video imports


def load_raw_data(input_folder, sub_seq=None, scale=1, print_freq=1000):
    """
    Loads either multipage tiff or set of tif files
    located in the input folder.
    :param input_folder: str. Folder in which tiff files(s) are located.
    :param sub_seq:list. Indices of files to lead (i.e. range(10))
    :param scale: scalar. Scale the dimensions of the image.
    :param print_freq: Print loading progress after every 'print_freq' number
                       of frames.
    :return: image stack

    NOTE: There was originally an issue where, for importing ome.tif,
          this only seems to work with python 2.7!!!
    """

    # Get list of tif files
    tif_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.tif'):
            tif_files.append(file)
    tif_files.sort()

    start_load = time.time()
    block_time = time.time()

    is_multipage_seq = False
    if ".ome.tif" in tif_files[0]:
        is_multipage_seq = True

    if len(tif_files) == 1 and not is_multipage_seq:
        print('There is a single tif file (presumably multi-page/bigtiff). ')
        with tifffile.TiffFile(os.path.join(input_folder,
                               tif_files[0])) as tif:
            print(sub_seq)
            stack = tif.asarray(key=sub_seq)
            # stack = tif.asarray()
    else:
        if is_multipage_seq:
            print('Loading multi-page sequence of ome.tif.')
            tif_order = []
            # Properly order the files
            for ind, tf in enumerate(tif_files):
                id = tf.split('_')[-1].split('.ome.tif')[0]
                if id == 'Pos0':
                    tif_order.append(0)
                else:
                    tif_order.append(int(id))

            tif_files = [x for _, x in sorted(zip(tif_order, tif_files))]
            # Deal wih sub_seq
            for ind, tf in enumerate(tif_files):
                f = os.path.join(input_folder, tf)
                with tifffile.TiffFile(f) as tif:
                    print('Loading substack #', str(ind))
                    substack = tif.asarray(key=sub_seq)
                    print(substack.shape)
                if ind == 0:
                    # The entire tif stack is loaded from the first filename.
                    stack = substack
                    break
                else:
                    pass

            # Load the files and concatenate them
            # Deal with subseq.
        else:
            # Read them into a np array (requires holding whole stack in mem!)
            print('Loading sequence of individual tifs.')

            if sub_seq is not None:
                tif_files = tif_files[sub_seq[0]:sub_seq[-1] + 1]

            stack = []
            for idx, tiff_in in enumerate(tif_files):
                if idx % print_freq == 0:
                    fraction = (idx + 1) / len(tif_files)
                    new_block_time = time.time()
                    print('Reading frame ', idx, ' fraction = ', fraction,
                          ' t = ', new_block_time - block_time)
                    block_time = new_block_time
                with tifffile.TiffFile(os.path.join(input_folder,
                                       tiff_in)) as tif:
                    if idx == 0:
                        im = tif.asarray()
                        stack = np.zeros((len(tif_files), im.shape[0],
                                         im.shape[1]), dtype=np.uint16)
                    stack[idx, :, :] = tif.asarray()

    print('Total time for load: ', time.time() - start_load)

    return stack


def load_tiff_stack(dirname):
    print("\t--> Loading video: " + dirname)
    imseq = pims.ImageSequence(os.path.join(dirname, '*.tif'))
    imgs = np.zeros((imseq.frame_shape[0], imseq.frame_shape[1], len(imseq)))
    for i in range(len(imseq)):
        imgs[:, :, i] = imseq.get_frame(i)

    return imgs


def load_video(path_to_data):
    """
    Load a video represented as a stack of tiffs or a multipage tiff.
    :param path_to_data: either a path to a folder containing a stack of tiffs,
                         or the full path to a multipage tiff.
    :return: a video as a 3D numpy array.
    """
    print("\t--> Loading video: " + path_to_data)

    if path_to_data[-4:] == 'tiff' or path_to_data[-3:] == 'tif':
        # Multipage tiff
        vid = skimage.io.imread(path_to_data)
        vid = np.swapaxes(np.swapaxes(vid, 0, 2), 0, 1)

    else:
        # Separate tiffs
        fnames = glob.glob(os.path.join(path_to_data, '*.tif*'))
        vid = np.asarray(skimage.io.imread_collection(fnames))
        vid = np.swapaxes(np.swapaxes(vid, 0, 2), 0, 1)

    return vid


def save_to_mp4(vid, save_path):
    pass


def save_video(vid, save_path, fps=30, clim=None,
               cmap="gray", new_method=True):
    """
    Save out a video?
    TODO Document this function
    :param vid:
    :param save_path:
    :return:
    """
    print("\t--> Saving video: " + save_path)

    def frame_to_npimage(frame):
        coefs = np.tile(frame, [3, 1, 1])
        coefs = (coefs - clim[0])/(clim[1] - clim[0])
        coefs[np.where(coefs > 1)] = 1
        coefs[np.where(coefs < 0)] = 0
        coefs = 255*coefs.swapaxes(0, 2).swapaxes(0, 1)
        return coefs

    def make_frame(t):
        return frame_to_npimage(vid[:, :, t])

    if new_method:
        animation = mpy.VideoClip(make_frame, duration=vid.shape[2])
        animation.write_videofile(save_path, fps=fps)
    else:
        method == ''
        ffmpeg_writer = manimation.writers['ffmpeg']
        metadata = dict(title='Title!', artist='Matplotlib',
                        comment=save_path)
        writer = ffmpeg_writer(bitrate=500)
        fig = plt.figure()
        if clim is None:
            clim = [np.min(vid), np.max(vid)]

        plotted = plt.imshow(vid[:, :, 0], clim=clim, cmap=cmap)
        nframes = vid.shape[2]
        with writer.saving(fig, save_path, nframes):
            for i in range(nframes):
                plotted.set_data(vid[:, :, i])
                writer.grab_frame()


def save_hdf5(vid, save_path):
    """

    :param vid:
    :param save_path:
    :return:
    """
    # TO DO TO DO
    raise NotImplementedError('Method not yet implemented!')


def load_ttl_file(ttl_path):
    """
    Load up a TTL file from a desired path. Return raw TTL traces.
    :param ttl_path: string containing path to a pkl file containing TTL pulses
    :return: dict containing raw TTL traces from all channels
    """
    ttl_file = defaultdict(list)
    with open(ttl_path, "rb") as f:
        while True:
            try:
                datum = pickle.load(f)
                for key in datum.keys():
                    ttl_file[key].append(datum[key])
            except EOFError:
                break

    for key in ttl_file.keys():
        ttl_file[key] = np.concatenate(ttl_file[key])

    return ttl_file


def load_ttl_times(ttl_path, thresh=None, sampling_rate=20000,
                   close_samples=10, debug_plot=False):
    """
    Parse and load TTL times from a provided path to a TTL file.
    :param ttl_path: string containing path to a pkl file containing TTL pulses
    :param thresh: (optional) channel names and TTL detection thresholds.
    :param sampling_rate: (optional) TTL pulse sampling rate in Hz
    :param close_samples: (optional) pulses detected in sequential
                                     samples are discarded.
    :param debug_plot: (optional) make debug plots?
    :return: dict containing parsed TTL event times (in samples)
    """

    # Lookup table for saving signals
    lut = {'Dev1/ai0': 'imaging',
           'Dev1/ai1': 'behavior_lower',
           'Dev1/ai2': 'trials',
           'Dev1/ai3': 'visual_stimulus',
           'Dev1/ai4': 'behavior_upper'}

    # Thresholds for detecting signals
    if thresh is None:
        thresh = {'Dev1/ai0': 6,
                  'Dev1/ai1': 3,
                  'Dev1/ai2': 6,
                  'Dev1/ai3': 6,
                  'Dev1/ai4': 3}
    else:
        thresh = {'Dev1/ai0': thresh,
                  'Dev1/ai1': thresh,
                  'Dev1/ai2': thresh,
                  'Dev1/ai3': thresh,
                  'Dev1/ai4': thresh}

    # Polarity on different TTL signals
    sign = {'Dev1/ai0': -1,
            'Dev1/ai1': -1,
            'Dev1/ai2': 1,
            'Dev1/ai3': -1,
            'Dev1/ai4': -1}

    # Load the TTL file
    ttl_file = load_ttl_file(ttl_path)

    # Convert the data into numpy arrays
    for key in ttl_file.keys():
        if np.ndim(ttl_file[key]) > 1:
            ttl_file[key] = np.concatenate(ttl_file[key])
        else:
            ttl_file[key] = np.array(ttl_file[key])

    # Detect TTL events
    ttl_times = {}
    for idx, k in enumerate(ttl_file.keys()):
        sd_thresh = thresh[k]
        dt = np.diff(sign[k]*ttl_file[k])
        ttl_times[lut[k]] = np.where(dt > sd_thresh*np.std(dt))[0]
        close_idx = np.where(np.diff(ttl_times[lut[k]]) < close_samples)[0]
        ttl_times[lut[k]] = np.delete(ttl_times[lut[k]], close_idx)
        rate = np.median(np.diff(ttl_times[lut[k]] / sampling_rate)[0])
        print(k, lut[k], round(rate, 3), 's,', round(1/rate, 3), 'Hz')

    # Get trial onset times in frames
    trial_onset_frames = np.zeros(len(ttl_times['trials']))
    for idx, trial_onset in enumerate(ttl_times['trials']):
        df = np.abs(trial_onset - ttl_times['imaging'])
        trial_onset_frames[idx] = np.argmin(df)

    # Plot pulses to see if things make sense
    if debug_plot:
        plt.figure(figsize=(8, 8))
        for ai, key in enumerate(ttl_times.keys()):
            plt.subplot(2, 3, ai+1)
            plt.plot(1/(np.diff(ttl_times[key])/sampling_rate), '.')
            plt.ylabel('Hz')
            plt.xlabel('Events')
            plt.title(key)
        plt.tight_layout()

        plt.figure()
        off = 1000
        offset = [0, -5, -20, -25, -45]
        x = ttl_times['trials'][10]
        for idx, k in enumerate(ttl_file.keys()):
            plt.plot(sign[k]*ttl_file[k][x-off:x+off]+offset[idx])
        in_rng = np.intersect1d(np.where(ttl_times['imaging'] >= x-off)[0],
                                np.where(ttl_times['imaging'] < x+off)[0])
        ind = ttl_times['imaging'][in_rng]-(x-off)
        plt.plot(ind, [-5]*len(ind), 'r.', markersize=10)

    return ttl_times, trial_onset_frames
