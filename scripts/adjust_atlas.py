#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Provides an interface for aligning atlas
after traces have been merged, and resaving out
a new merged-trace file.

Created on May 10, 2018

@author: izkula
"""

import time
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py

import cosmos.params.trace_analyze_params as params
from cosmos.traces.cosmos_traces import CosmosTraces
from cosmos.imaging.cosmos_dataset import CosmosDataset

# Note: If you screw up an alignment, the original merged-traces file should be
# in processedData folder (on the computer where trace extraction occured).

if __name__ == '__main__':
    # data_dir = "/home/izkula/Dropbox/cosmos_data/"
    data_dir = "/home/user/Dropbox/cosmos_data/"

    do_adhoc = False

    if do_adhoc:
        # Visual dataset 8
        # h5_path = (dd + '20180213/' +
        #           'm72_vis_stim_2/20180213-m72_vis_stim_2-merged_traces.h5')
        # Visual dataset 12
        # h5_path = (dd + '20180522/' +
        #           'cux2ai148m194_visual_stim_1/' +
        # '20180522-cux2ai148m194_visual_stim_1-merged_traces.h5')
        # Visual dataset 13
        # h5_path = (dd + '20180522/' +
        #           'cux2ai148m943_visual_stim_1/' +
        # 20180522-cux2ai148m943_visual_stim_1-merged_traces.h5')
        # Visual dataset 14
        # h5_path = (dd + '20180523/' +
        #           'cux2ai148m945_visual_stim_1/' +
        # 20180523-cux2ai148m945_visual_stim_1-merged_traces.h5')
        # Visual dataset 15
        h5_path = (dd + '20180523/' +
                   'cux2ai148m192_vis_stim_1/' +
                   '20180523-cux2ai148m192_vis_stim_1-merged_traces.h5')
    else:
        dataset_id = 38

        dataset = params.DATASETS[dataset_id]
        dataset['data_root'] = data_dir
        h5_path = os.path.join(data_dir, dataset['date'], dataset['name'],
                               dataset['date'] + '-' + dataset['name'] +
                               '-merged_traces.h5')

    CosmosDataset.align_atlas_postmerge(h5_path)
