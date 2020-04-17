from matplotlib import pyplot as plt
from matplotlib import animation, rc
import seaborn as sns
import os
import numpy as np
import matplotlib
import tifffile
from PIL import Image
import time
from matplotlib import gridspec
from subprocess import call

# import cv2

import sys
sys.path.append("..")
from util import VideoSlider  # noqa


# This script makes a video of the behavior
# and neural video side by side.
# Unclear if this still works - needs to be cleaned
# up (20180321).
def load_raw_data(input_folder, scale=1, n_frames=None):
    # Get list of tif files
    tif_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.tif'):
            tif_files.append(file)
    tif_files.sort()
    if n_frames is not None:
        tif_files = tif_files[0:n_frames]

    if len(tif_files) == 1:
        with tifffile.TiffFile(
                os.path.join(input_folder, tif_files[0])) as tif:
            stack = tif.asarray()
    else:
        # Read them into a np array (requires holding whole stack in memory!)
        n_frame_print = 100
        for idx, tiff_in in enumerate(tif_files):
            if idx % n_frame_print == 0:
                fraction = (idx+1)/len(tif_files)
                print('Reading frame ', idx, ' fraction = ', fraction)
            with Image.open(input_folder + '/' + tiff_in) as im:
                im = im.resize([int(im.size[0]*scale), int(im.size[1]*scale)])
                if idx == 0:
                    print('Allocating array')
                    whole_stack = np.zeros(
                        (len(tif_files), im.size[1], im.size[0]))
                    print('Done allocating array')
                else:
                    whole_stack[idx, :, :] = np.array(im)
    return whole_stack


def get_peak_dip_indices(trace, stdev):
    trial_onsets = np.where(trace > stdev * np.std(trace))[0]
    trial_light_off = np.where(trace < -stdev * np.std(trace))[0]
    trial_onsets = np.delete(
        trial_onsets, np.where(np.diff(trial_onsets) == 1))
    trial_light_off = np.delete(
        trial_light_off, np.where(np.diff(trial_light_off) == 1))

    return trial_onsets, trial_light_off


if __name__ == '__main__':

    base_dir = "/home/izkula/Data/data/"

    date = '20171115'
    sessname = 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session3_1'
    tonguename = 'cuxai148m1-20171115-150411-camera'
    neural_fname = os.path.join(base_dir, date, sessname, "Pos0")
    tongue_fname = os.path.join(base_dir, date, tonguename + '.tif')

    # Load neural video
    neural_stack = load_raw_data(neural_fname, 1, n_frames=None)

    # Load tongue video
    with tifffile.TiffFile(tongue_fname) as tiff:
        tongue_stack = tiff.asarray()

    # Get trial onset frames (neural)
    neural_mean = np.mean(
        np.mean(neural_stack[:, 1:200, 1:200], axis=1), axis=1)
    dm = np.diff(neural_mean)
    neural_trial_onsets, neural_trial_light_off = get_peak_dip_indices(dm, 5)

    # Get trial onset frames (tongue)
    rgb = np.sum(np.sum(tongue_stack, axis=1), axis=1)
    dg = np.diff(np.array(rgb[:, 1], dtype=float))
    tongue_trial_onsets, tongue_trial_light_off = get_peak_dip_indices(dg, 10)

    # Extract video for a single trial
    trials = [0, 3, 5, 13]
    trials = [1]

    for nskipframes in [1]:
        for trial in trials:
            neural_trial = neural_stack[
                neural_trial_onsets[
                    trial]:neural_trial_onsets[trial+1]+30, :, :]
            tongue_trial = tongue_stack[
                tongue_trial_onsets[
                    trial]:tongue_trial_onsets[trial+1]+30, :, :]

            # Get min-subtracted neural video (maybe use the second-to-min)
            minsub_trial = neural_trial - np.min(neural_trial, axis=0)

            # Determine upsample rate and generate paired video
            tto = tongue_trial_onsets
            nto = neural_trial_onsets
            index_scale = (
                tto[trial+1]-tto[trial])*1.0/(nto[trial+1]-nto[trial])

            # Save frames to a folder
            imdir = os.path.join("/home/izkula/COSMOS_results/",
                                 date, sessname,
                                 'trial_' + str(trial), str(nskipframes))
            try:
                os.makedirs(imdir)
            except FileExistsError:
                print(imdir + ' already exists.')

            fig = plt.figure(figsize=(10, 10))
            iter = 0
            for kk in range(0, np.shape(tongue_trial)[0], nskipframes):
                print(kk)
                gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
                ax0 = plt.subplot(gs[0])
                scaled_ind = min(
                    int(np.round(kk / index_scale)),
                    np.shape(minsub_trial)[0]-1)
                ax0.imshow(
                    minsub_trial[scaled_ind, :, :],
                    clim=[0, 2500], cmap='Greys')
                ax0.axes.get_xaxis().set_visible(False)
                ax0.axes.get_yaxis().set_visible(False)

                ax1 = plt.subplot(gs[1])
                ax1.imshow(tongue_trial[kk, :, :], cmap='Greys')
                ax1.axes.get_xaxis().set_visible(False)
                ax1.axes.get_yaxis().set_visible(False)

                plt.tight_layout()
                plt.savefig(os.path.join(imdir, str(iter).zfill(7)+'.png'),
                            bbox_inches='tight')
                iter = iter + 1

            call(["ffmpeg", "-r", "30", "-i",
                  imdir+"/%07d.png",
                  "-vcodec", "libx264", "-pix_fmt", "yuv420p", imdir+".mp4"])
