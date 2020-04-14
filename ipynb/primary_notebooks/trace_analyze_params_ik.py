
"""
Parameter file containing information for datasets to be processed in trace_analyze_params_ik notebook.
Instructions:
    1) NEVER modify the numeric key of an existing entry, always append a new entry. This could screw up
       other people's scripts. 
    2) For an entry, include at minimum the following keys: 'date', 'name', 'bpod_file', 'info'
"""
DATASETS = {    
     '1': {'date': '20180212', 'name': 'm72_COSMOSShapeMultiGNG_1', 
           'bpod_file':'m72/COSMOSShapeMultiGNG/Session Data/m72_COSMOSShapeMultiGNG_20180212_193940.mat',
           'info':'This was imported on 2/15/18. First one that looks great!',
          }, 
    
    '2':{'date': '20180219', 'name':'cux2m72_COSMOSTrainMultiGNG_day5_1', 
         'bpod_file':'cux2m72/COSMOSTrainMultiGNG/Session Data/cux2m72_COSMOSTrainMultiGNG_20180219_160349.mat',
         'info':'This behavior data was used for Tims 2018 lab meeting and neural data for COSYNE 2018.',
        },
    
    '3':{'date': '20180227', 'name':'cux2m72_COSMOSTrainMultiBlockGNG_1', 
         'bpod_file':'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180227_132125.mat',
         'info':'Fairly good 4-spout behavioral data (between trials 50-175). Need to import this neural data (as of 20180308)',
        },
    
    '4':{'date': '20180228', 'name':'cux2m72_COSMOSTrainMultiBlockGNG_1', 
         'bpod_file':'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180228_142118.mat',
         'info':'Fairly good 4-spout behavioral data (between trials 110-200). Need to import neural data (as of 20180308)',
        },
    
    '5':{'date':'20180226', 'name':'cux2m72_COSMOSTrainMultiBlockGNG_1',
         'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180226_210817.mat',
         'info':'Does not have imaging data.',
        },
    
    '6':{'date':'20180314', 'name':'vtapfcm36_COSMOSTrainMultiBlockGNG_1',
         'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180226_210817.mat',
         'info':'Does not have imaging data.',
        },
    
    '7':{'date': '20180401', 'name': 'cux2ai148m72_COSMOSTrainMultiBlockGNG_1',
         'bpod_file': 'cux2m72/COSMOSTrainMultiBlockGNG/Session Data/cux2m72_COSMOSTrainMultiBlockGNG_20180401_134852.mat',
         'info':'***Imported this 4/2/18. Very good behavior.',
        }, 
}
