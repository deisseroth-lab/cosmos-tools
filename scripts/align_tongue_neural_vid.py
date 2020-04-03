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
from util import VideoSlider


### This script makes a video of the behavior
### and neural video side by side.
### Unclear if this still works - needs to be cleaned
### up (20180321).

def load_raw_data(input_folder, scale=1, n_frames = None):
    # Get list of tif files
    tif_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.tif'):
            tif_files.append(file)
    tif_files.sort()
    if n_frames is not None:
        tif_files = tif_files[0:n_frames]

    if len(tif_files) == 1:
        with tifffile.TiffFile(os.path.join(input_folder, tif_files[0])) as tif:
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
                    whole_stack = np.zeros((len(tif_files), im.size[1], im.size[0]))
                    print('Done allocating array')
                else:
                    whole_stack[idx,:,:] = np.array(im)

            # stack = np.reshape(whole_stack, [len(whole_stack), im.size[0], im.size[1]])
    return whole_stack

def get_peak_dip_indices(trace, stdev):
    trial_onsets = np.where(trace > stdev * np.std(trace))[0]
    trial_light_off = np.where(trace < -stdev * np.std(trace))[0]
    trial_onsets = np.delete(trial_onsets, np.where(np.diff(trial_onsets) == 1))
    trial_light_off = np.delete(trial_light_off, np.where(np.diff(trial_light_off) == 1))

    return trial_onsets, trial_light_off

if __name__ == '__main__':

    base_dir = "/home/izkula/Data/data/"

    date = '20171115'
    # sessname = 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session2_1'
    # tonguename = 'cuxai148m1-20171115-144534-camera'
    sessname = 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session3_1'
    tonguename = 'cuxai148m1-20171115-150411-camera'
    neural_fname =  os.path.join(base_dir, date, sessname, "Pos0")
    tongue_fname =  os.path.join(base_dir, date, tonguename + '.tif')


    ### Load neural video
    neural_stack = load_raw_data(neural_fname, 1, n_frames=None)

    ### Load tongue video
    with tifffile.TiffFile(tongue_fname) as tiff:
        tongue_stack = tiff.asarray()
    # VideoSlider.vslide(np.swapaxes(tongue_stack[:, :, :, 1], 0, 2))

    ### Get trial onset frames (neural)
    neural_mean = np.mean(np.mean(neural_stack[:, 1:200, 1:200], axis=1), axis=1)
    dm = np.diff(neural_mean)
    neural_trial_onsets, neural_trial_light_off = get_peak_dip_indices(dm, 5)
    # VideoSlider.vslide(np.swapaxes(neural_stack[neural_trial_onsets[2]:neural_trial_onsets[2]+20, :, :], 0, 2))

    ### Get trial onset frames (tongue)
    rgb = np.sum(np.sum(tongue_stack, axis=1), axis=1)
    dg = np.diff(np.array(rgb[:, 1], dtype=float))
    tongue_trial_onsets, tongue_trial_light_off = get_peak_dip_indices(dg, 10)

    ### Extract video for a single trial
    trials =  [0, 3, 5, 13]
    trials = [1]
    # trials = [27, 28]
    # trials = [30, 50]

    for nskipframes in [1]:
        for trial in trials:
            neural_trial = neural_stack[neural_trial_onsets[trial]:neural_trial_onsets[trial+1]+30,:,:]
            tongue_trial = tongue_stack[tongue_trial_onsets[trial]:tongue_trial_onsets[trial+1]+30,:,:]

            ### Get min-subtracted neural video (maybe use the second-to-min)
            minsub_trial = neural_trial - np.min(neural_trial, axis=0)

            ### Determine upsample rate and generate paired video
            # n_tongue = np.shape(tongue_trial)[0]
            # n_neural = np.shape(neural_trial)[0]
            index_scale = (tongue_trial_onsets[trial+1]-tongue_trial_onsets[trial])*1.0/(neural_trial_onsets[trial+1]-neural_trial_onsets[trial])
            # index_scale = n_tongue*1.0/n_neural

            #### Save frames to a folder
            # nskipframes = 5 ### Only include every nskipframes in the video

            # imdir = os.path.join("/home/izkula/Data/Results/", date, sessname, 'trial_'+ str(trial), str(nskipframes))
            imdir = os.path.join("/home/izkula/COSMOS_results/", date, sessname, 'trial_'+ str(trial), str(nskipframes))
            try:
                os.makedirs(imdir)
            except:
                print(imdir + ' already exists.')

            fig = plt.figure(figsize=(10, 10))
            iter = 0
            for kk in range(0,np.shape(tongue_trial)[0],nskipframes):
                print kk
                gs = gridspec.GridSpec(1, 2, width_ratios=[3,1])
                ax0 = plt.subplot(gs[0])
                scaled_ind = min(int(np.round(kk / index_scale)), np.shape(minsub_trial)[0]-1)
                # ax0.imshow(minsub_trial[scaled_ind, :, :], clim=[0, 500], cmap='Greys')
                ax0.imshow(minsub_trial[scaled_ind, :, :], clim=[0, 2500], cmap='Greys')
                ax0.axes.get_xaxis().set_visible(False)
                ax0.axes.get_yaxis().set_visible(False)

                ax1 = plt.subplot(gs[1])
                ax1.imshow(tongue_trial[kk, :, :], cmap='Greys')
                ax1.axes.get_xaxis().set_visible(False)
                ax1.axes.get_yaxis().set_visible(False)

                plt.tight_layout()
                plt.savefig(os.path.join(imdir, str(iter).zfill(7)+'.png'), bbox_inches='tight')
                iter = iter + 1

            call(["ffmpeg", "-r", "30", "-i", imdir+"/%07d.png","-vcodec", "libx264", "-pix_fmt", "yuv420p", imdir+".mp4"])



    #### then call ffmpeg to convert to a video
    # import matplotlib.animation as manimation
    #
    # fig = plt.figure()
    # moviewriter = manimation.FFMpegFileWriter()
    # moviewriter.setup(fig, '/home/izkula/src/COSMOS/cosmos/analysis_scripts/test.mp4', dpi=100)
    # for kk in range(10):
    #     plt.subplot(121)
    #     plt.imshow(tongue_trial[kk, :, :, 1], cmap='Greys')
    #     plt.subplot(122)
    #     scaled_ind = int(np.round(kk / index_scale))
    #     plt.imshow(minsub_trial[kk, :, :], clim=[0, 1000], cmap='Greys')
    #     # plt.imshow(minsub_trial[scaled_ind, :, :], clim=[0, 1000], cmap='Greys') #### USE THIS THE OTHER IS FOR DEBUGGING
    #     moviewriter.grab_frame()
    # moviewriter.finish()
    #
    # # metadata = dict(title=neural_fname, artist='Matplotlib',
    # #                 comment='Movie support!')
    # # writer = FFMpegWriter(fps=15, metadata=metadata)
    # #
    # #
    # #
    #
    #         # time.sleep(0.03)
    #
    #
    # import matplotlib.animation as animation
    # fig2 = plt.figure()
    #
    # x = np.arange(-9, 10)
    # y = np.arange(-9, 10).reshape(-1, 1)
    # base = np.hypot(x, y)
    # ims = []
    # for add in np.arange(15):
    #     ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))
    #
    # im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
    #                                    blit=True)
    # im_ani.save('im.mp4', writer=writer)
