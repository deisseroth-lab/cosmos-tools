import numpy as np
"""
Parameter file containing information for datasets to be processed in intrinsic_imaging_alignment notebook.
Instructions:
    1) NEVER modify the numeric key of an existing entry, always append a new entry. This could screw up
       other people's scripts.
    2) For an entry, include at minimum the following keys: 
        'base': directory in which scaled video is saved.
        'im_path': name of scaled movie. 
        'hz': frame rate of scaled movie. 
        'ndirs': number of moving bar orientations
        'stim_length': in seconds. time for bar to move across screen
        'blank_length': in seconds. time of blank screen between trials.
        'sd_thresh': threshold for finding led frames.
        'start_frame': how many frames to cut off of the beginning.
        'cull_first_pulse': Ignore the first led pulse (because doesn't correspond to trial)
        'filter_data': Where to smooth the data. This flag not currently used?
        'info': Additional information about the trial. 
"""

DATASETS = {
    0: {'base': '/hdd1/Data/20180701/thy1gc_bars8_iso0-4_20hz_40trials_grating10_blank1/', 
        'im_path': 'thy1gc_bars8_iso0-4_20hz_40trials_grating10_blank1_1-1.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 300,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Test with Thy1Gcamp6s mouse. You can definitely see the relevant signal.'
       }, 
    
    1: {'base': '/media/Data/data/20180703/m72_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1_scaled/', 
        'im_path': 'm72_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Initial 150 trial recording for m72. Conclusion: Worked well for orientation 2.'
       }, 
    
    2: {'base': '/media/Data/data/20180703/m943_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1_scaled/', 
        'im_path': 'm943_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Initial 150 trial recording for m943. Conclusion: Orientations 1, 3 are good, weirdly. But not 0/2.'
       }, 
    
    3: {'base': '/media/Data/data/20180703/m194_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1_scaled/', 
        'im_path': 'm194_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Initial 150 trial recording for m194. Conclusion: Worked ok.'
       },

    4: {'base': '/media/Data/data/20180703/m945_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1_scaled/', 
        'im_path': 'm945_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1_scaled.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Initial 150 trial recording for m945. Conclusion: Worked well for orientation 2.'
       }, 
    
    5: {'base': '/media/Data/data/20180704/m943_bars8_700nm_iso0-5_20hz_200trials_ori180_grating10_blank1_1_1-1_scaled/', 
        'im_path': 'm943_bars8_700nm_iso0-5_20hz_200trials_ori180_grating10_blank1_1_1-1_scaled.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 1,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Recaptured more data of the key orientation for m943.'
       },    
    6: {'base': '/hdd1/Data/20180829/m4116_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1_scaled/', 
        'im_path': 'm4116_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_1-1_scaled.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Initial 150 trial recording for m4116.'
       },    
    7: {'base': '/media/Data/data/20181205/rasgrfm4263_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_scaled/',
        'im_path': 'rasgrfm4263_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1-1.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length':10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'rasgrfm4263. Note, you swapped orientations 0 and 2 (so that you record the better orientation first) Video looks good for orientation 0.'
       },
    
    7: {'base': '/hdd1/Data/20190523/cux2m4293_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_scaled/', 
        'im_path': 'cux2m4293_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1_scaled.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Initial 150 trial recording for m4293.'
       },   
    
    8: {'base': '/hdd1/Data/20190524/cux2m4293_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1-1_scaled/', 
        'im_path': 'cux2m4293_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1-1_scaled.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': 'Second 150 trial recording for m4293.'
       },   
    
        9: {'base': '/hdd1/Data/20190614/cux25456_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1-1_scaled/', 
        'im_path': 'cux25456_bars8_700nm_iso0-5_20hz_150trials_grating10_blank1_1-1_scaled.tif',
        'led_px': np.array([150, 30]),
        'hz': 10,
        'ndirs': 4,
        'stim_length': 10,
        'blank_length': 1,
        'sd_thresh': 1,
        'start_frame': 115,
        'cull_first_pulse': True,
        'filter_data': False,
        'info': '.'
       },   
    
}