
"""
Parameter file containing information for datasets to be processed in trace_analyze_params_ik notebook.
Instructions:
    1) NEVER modify the numeric key of an existing entry, always append a new entry. This could screw up
       other people's scripts.
    2) For an entry, include at minimum the following keys: 'date', 'name', 'bpod_file', 'info'
"""
DATASETS = {
    1: {'date': '20180212', 'name': 'm72_COSMOSShapeMultiGNG_1',
        'bpod_file': 'm72/COSMOSShapeMultiGNG/Session Data/m72_COSMOSShapeMultiGNG_20180212_193940.mat',
        'info': 'This was imported on 2/15/18. First one that looks great!',
        },

    2: {'date': '20180219', 'name': 'cux2m72_COSMOSTrainMultiGNG_day5_1',
        'bpod_file': 'cux2m72/COSMOSTrainMultiGNG/Session Data/cux2m72_COSMOSTrainMultiGNG_20180219_160349.mat',
        'info': 'This behavior data was used for Tims 2018 lab meeting and neural data for COSYNE 2018.',
        },

    3: {'date': '20180227', 'name': 'cux2m72_COSMOSTrainMultiBlockGNG_1',
        'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180227_132125.mat',
        'info': 'Fairly good 4-spout behavioral data (between trials 50-175). Need to import this neural data (as of 20180308)',
        },

    4: {'date': '20180228', 'name': 'cux2m72_COSMOSTrainMultiBlockGNG_1',
        'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180228_142118.mat',
        'info': 'Fairly good 4-spout behavioral data (between trials 110-200). Need to import neural data (as of 20180308)',
        },

    5: {'date': '20180226', 'name': 'cux2m72_COSMOSTrainMultiBlockGNG_1',
        'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180226_210817.mat',
        'info': 'Does not have imaging data.',
        },

    6: {'date': '20180314', 'name': 'vtapfcm36_COSMOSTrainMultiBlockGNG_1',
        'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180226_210817.mat',
        'info': 'Does not have imaging data.',
        },

    7: {'date': '20180401', 'name': 'cux2ai148m72_COSMOSTrainMultiBlockGNG_1',
        'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180401_134852.mat',
        #'ttl_file': 'cux2m72-20180401-134904-TTL.pkl',
        'regressors_name': 'cux2m72-20180401-134904',
        'info': '***Imported this 4/2/18. Very good behavior. 1240 sources.',
        },
    8: {'date': '20180213', 'name': 'm72_vis_stim_2',
        'bpod_file': None,
        'info': 'HORIZONTAL Visual dataset, good quality, from m72 (used at COSYNE).',
        },
    9: {'date': '20180404', 'name': 'cux2ai148m72_visual_stim_1',
        'bpod_file': None,
        'info': 'VERTICAL Visual dataset, good quality, from m72.',
        },
    10: {'date': '20180404', 'name': 'cux2ai148m945_visual_stim_1',
         'bpod_file': None,
         'info': 'VERTICAL Visual dataset, decent quality, from m945.',
         },
    11: {'date': '20180420', 'name': 'cux2ai148m943_COSMOSTrainMultiBlockGNG_1',
         'bpod_file':'cux2m943/COSMOSTrainMultiBlockGNG/Session Data/cux2m943_COSMOSTrainMultiBlockGNG_20180420_130634.mat',
         'regressors_name':'cux2m943-20180420-130639',
         #'ttl_file':'cux2m943-20180420-130639-TTL.pkl',
         'info':'***This has good behavior. 1084 sources'
        },
    12: {'date': '20180522', 'name': 'cux2ai148m194_visual_stim_1',
         'bpod_file': None,
         'info': 'HORIZONTAL Visual dataset, best quality, from m194.',
         },
    13: {'date': '20180522', 'name': 'cux2ai148m943_visual_stim_1',
         'bpod_file': None,
         'info': 'HORIZONTAL Visual dataset, ok quality, from m943.',
         },
    14: {'date': '20180523', 'name': 'cux2ai148m945_visual_stim_1',
         'bpod_file': None,
         'info': 'HORIZONTAL Visual dataset, decent quality, from m945.',
         },
    15: {'date': '20180523', 'name': 'cux2ai148m192_vis_stim_1',
         'bpod_file': None,
         'info': 'HORIZONTAL Visual dataset, decent quality, from m192.',
         },
    16: {'date': '20180520', 'name': 'cux2ai148m194_COSMOSTrainMultiBlockGNG_1',
         'bpod_file':'cux2m194/COSMOSTrainMultiBlockGNG/Session Data/cux2m194_COSMOSTrainMultiBlockGNG_20180520_161430.mat',
         'info':'***First day of training, but over 2000 recovered sources!', ### At one point this was dataset 12...
        },
    17: {'date': '20180514', 'name': 'cux2ai148m72_COSMOSTrainMultiBlockGNG_1',
         'bpod_file':'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180514_161411.mat',
         'info':'**?*Good behavior. But window was becoming less good by this point. 723 sources', ### At one point, this was dataset 13... ### NEED TO FIX THIS, doesn't import?
        },
    18: {'date': '20180626', 'name': 'cux2ai148m945_COSMOSTrainMultiBlockGNG_1',
         'bpod_file':'cux2m945/COSMOSTrainMultiBlockGNG/Session Data/cux2m945_COSMOSTrainMultiBlockGNG_20180626_132933.mat',
         'regressors_name':'cux2m945-20180626-132942',
         'info':'***Reasonable behavior. Around 1000 neurons.', 
        },
    19: {'date': '20180709', 'name': 'cux2ai148m194_COSMOSTrainMultiBlockGNG_2',
         'bpod_file':'cux2m194/COSMOSTrainMultiBlockGNG/Session Data/cux2m194_COSMOSTrainMultiBlockGNG_20180709_134919.mat',
         'regressors_name':'cux2m194-20180709-134705',
         'info':'***Reasonable behavior. Around 1400 neurons.', 
        },
    21: {'date': '20180707', 'name': 'cux2ai148m194_COSMOSTrainMultiBlockGNG_1', 
         'bpod_file': 'cux2m194/COSMOSTrainMultiBlockGNG/Session Data/cux2m194_COSMOSTrainMultiBlockGNG_20180707_200946.mat', 
         'info': 'Fine, no motion. Blue light turned off at end, be careful - trim it?. SNR is poor.', # NEEDS WORK
        },
    22: {'date': '20180514', 'name':'cux2ai148m72_COSMOSTrainMultiBlockGNG_2',
         'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180514_161957.mat', 
         'info': '***Near perfect accuracy, window maybe not as good? SNR not great. 766 cells. Top no movement really. same with bottom. looks great',
        },
    23: {'date': '20180511', 'name':'cux2ai148m72_COSMOSTrainMultiBlockGNG_1',
         'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180511_153239.mat', 
         'info': 'Top no movement really, great. Bot seems to have more movement. On the edge of usable. Started with a lot of cells but due to motion ended up with only 597 (started with almost 3000 rois)'
        },
    24: {'date': '20180430', 'name': 'cux2ai148m943_COSMOSTrainMultiBlockGNG_1', 
         'bpod_file': 'cux2m943/COSMOSTrainMultiBlockGNG/Session Data/cux2m943_COSMOSTrainMultiBlockGNG_20180430_160711.mat',
         'info': 'Top has some jiggle at end, bot has no movement, imported looked okay, 84%, 692 rois. Not many posterior.',
        },    
    25: {'date': '20180424', 'name': 'cux2ai148m943_COSMOSTrainMultiBlockGNG_1', 
         'bpod_file': 'cux2m943/COSMOSTrainMultiBlockGNG/Session Data/cux2m943_COSMOSTrainMultiBlockGNG_20180424_114718.mat',
         'info': 'peak 94% # No movement, one jiggle in middle of top but ok, bot has a little more movement, 829 rois',
        },
#     26: {'date': '20180629', 'name':'cux2ai148m945_COSMOSTrainMultiBlockGNG_1', 
#          'bpod_file': None,
#          'info': 'Top has some jumps (maybe 2-3 pixels radius, seems like something was twisting? --- this dataset looks pretty crappy, not many good cells, a lot of artifactual cells with discontinuities - 320 rois'},
    27: {'date': '20180624', 'name': 'cux2ai148m945_COSMOSTrainMultiBlockGNG_1', 
         'bpod_file': 'cux2m945/COSMOSTrainMultiBlockGNG/Session Data/cux2m945_COSMOSTrainMultiBlockGNG_20180624_165052.mat',
         'info': 'peak 81% # Top essentially no movement. Bot also good. 1100 rois',
        },
    28: {'date': '20180205', 'name': 'COSMOSShapeMultiGNG-m72_1',
         'bpod_file': 'm72/COSMOSShapeMultiGNG/Session Data/m72_COSMOSShapeMultiGNG_20180205_160116.mat',
         'info': 'justOdors 10k frames. No movement,good. Around 600 cells'
        },
    29: {'date': '20180214', 'name': 'm140_initial_odor_exposure__1',
         'bpod_file': 'm140/COSMOSShapeMultiGNG/Session Data/m140_COSMOSShapeMultiGNG_20180214_164532.mat',
         'info': 'justOdors 10k frames. Only about 350 cells.'
        },
    30: {'date': '20180328', 'name':'cux2ai148m943_COSMOSShapeMultiGNG_justOdor_1',
         'bpod_file':'cux2ai148m943/COSMOSShapeMultiGNG/Session Data/cux2ai148m943_COSMOSShapeMultiGNG_20180328_140647.mat',
#          'bpod_file':'cux2ai148m943/COSMOSShapeMultiGNG/Session Data/cux2ai148m943_COSMOSShapeMultiGNG_20180328_141718.mat
         'info':'justOdors 10k frames. About 800 cells.'
        },
    31: {'date':'20180328', 'name': 'cux2ai148m945_COSMOSShapeMultiGNG_justOdor_1',
         'bpod_file':'cux2ai148m945/COSMOSShapeMultiGNG/Session Data/cux2ai148m945_COSMOSShapeMultiGNG_20180328_153229.mat',
         'info':'justOdor 10k frames. About 700 cells. Pretty good looking.'
        },
    32: {'date':'20181002', 'name': 'cux2m4202-vis-stim_2',
         'bpod_file': None,
         'info':'OEG MOUSE (good), visual stimulation to gratings. HORIZONTAL Visual dataset.'
        },
    33: {'date':'20181002', 'name': 'cux2m4204-vis-stim_3',
         'bpod_file': None,
         'info':'OEG MOUSE (bad), visual stimulation to gratings. HORIZONTAL Visual dataset.'
        },
    34: {'date':'20181002', 'name': 'thy1m740-vis-stim_2',
         'bpod_file': None,
         'info':'THY1 MOUSE (good), visual stimulation to gratings. HORIZONTAL Visual dataset.'
        },
    35: {'date':'20181014', 'name': 'rasgrf2m4263_COSMOSTrainMultiBlockGNG_1',
         'bpod_file':'rasgrfm4263/COSMOSTrainMultiBlockGNG/Session Data/rasgrfm4263_COSMOSTrainMultiBlockGNG_20181014_153314.mat',
         'info': '1414 sources.  ',
        },
    
    36: {'date':'20190522', 'name':'cux2m4293_lsd_1',
          'info':'~2500 sources!',
        },

    37: {'date': '20190604', 'name': 'cux2m4293_oddball_merged_orig_14_then_reversed_14',
         'bpod_file': None, #'cux2m4293/Oddball/Session Data/cux2m4293_Oddball_20190604_213011.mat', ### But also this one, from the second half after the reversal cux2m4293_Oddball_20190604_220612
         'info': 'First test of oddball paradigm.',
         },
    
    38: {'date': '20190502', 'name': 'm4293_1hr_resting_state_3',
         'bpod_file': None,
         'info': 'Spontaneous activity, 2568 sources.',
         },

    39: {'date': '20190521', 'name': 'cux2m4293_saline_day2_1',
         'bpod_file': None,
         'info': 'Culled JK, 20190722.',
         },
}
 
