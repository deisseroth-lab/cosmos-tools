%%%% This script runs CNMF-E on ROIs (that have been processed with
%%%% crop_ROIs function). 

addpath(genpath('~/src/COSMOS/matlab'))
addpath('~/Software/CNMF_E-master')
cnmfe_setup
close all; clear;

baseDir = '/home/izkula/Data/data'
processedDataDir = '/home/izkula/Data/processedData'
saveDir = processedDataDir

doRunCNMFE = true

useProcessedPatches = true;


data = {...
   {'20180102', 'm52_1wk_post_tamox_2', '', [1]};
}


errorLog = {}
for kk = 1:numel(data)
    d = data{kk};

    cropSaveDir =  fullfile(saveDir, d{1}, d{2})
    
    %%% Register the two images?
    
    
    %%% Select regions to process (with ROI poly?)
    
    
    %%% Break them up - or let CNMF do this automatically? That didn't seem
    %%% to work so well, though
    
    
    %%% Run cnmfe in parallel?
    
    
    %%% Gather the results? Merge defocused regions etc. 
    
    
    %%% Fit to atlas
    
    
    

    for patch = d{4}
        if useProcessedPatches
            patchDir = fullfile(cropSaveDir, ['patch-', num2str(patch)], num2str(patch), 'vid.tif')
        else
            patchDir = fullfile(baseDir, d{1}, d{2}, 'vid.tif')
        end
        
         try
            neuron = run_cnmfe(patchDir)
            
         catch
             errorLog{kk} = patchDir;
             disp(['---------->Something happened with ', patchDir])
         end
    end
end