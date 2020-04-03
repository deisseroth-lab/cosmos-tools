#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Manually cut out 2 lenslet array image RIOs and
run CNMFe to extract traces.

Created on Jan 29, 2018

@author: izkula
"""
import time

import cosmos.params.dataset_params as params
from cosmos.imaging.frame_processor import FrameProcessor

if __name__ == '__main__':

    workstation = 'cosmosdata'  # 'cosmosdata', 'analysis2'
    if workstation == 'analysis2':
        # raw_data_dir = "/home/izkula/Data/data/"
        raw_data_dir = "/media/optodata/tam/"
    elif workstation == 'cosmosdata':
        #raw_data_dir = "/home/user/optodata/tam/"
        processed_data_dir = "/hdd1/Data/processedData/"
        # raw_data_dir = '/run/user/1000/gvfs/smb-share:server=171.65.102.187,share=data2/tam'
        #raw_data_dir = '/home/user/optodata2/tam'
        raw_data_dir = "/hdd1/Data/" ### REMOVE THIS IN A BIT!!

    print(raw_data_dir, processed_data_dir)

    ###########################################
    # 1. CHANGE PATHS ABOVE
    # 2. ADD PATHS TO DATASET_PARAMS.PY
    # 3. CHANGE FLAGS BELOW (RUN ONE AT A TIME)
    # 4. CHOOSE VENV: 'source activate cosmos2'
    # 5. RUN SCRIPT WITHOUT ARGUMENTS
    ##########################################
    do_select_crop_rois = False
    do_process_stacks = True
    do_run_cnmfe = False
    do_atlas_align = False

    for i in range(len(params.DATASETS)):
        d = params.DATASETS[i]
        print(d)

        f = FrameProcessor(raw_data_dir=raw_data_dir,
                           processed_data_dir=processed_data_dir,
                           dataset=d)

        f.vid_names = ['top', 'bot']
        # f.vid_names = ['bot']
        # f.vid_names = ['top']

        start = time.time()
        if do_select_crop_rois:  ###
            f.select_crop_rois()
        if do_process_stacks:
            f.crop_stack(do_remove_led_frames=False, do_motion_correct=True, LED_buffer=4) #2019.09.24 JK FIXME -- THERE ARE NO LED FRAMES IN THE CURRENT DATAESET FIXME FIXME FIXME FIXME
            f.plot_motion()
        if do_run_cnmfe:
            if workstation == 'analysis2':
                f.run_cnmfe(matlab_path='/home/izkula/Software/Matlab2017b/bin/matlab', nthreads=6)
            elif workstation == 'cosmosdata':
                f.run_cnmfe(matlab_path='/home/user/software/MATLAB/R2018a/bin/matlab', nthreads=6)
        if do_atlas_align:
            f.atlas_align()

        end = time.time()
        print("Elapse time: " + str(end - start))
