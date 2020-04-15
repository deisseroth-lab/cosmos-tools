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


% data = {...
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session2_1', 'cuxai148m1_COSMOSTrainGNG_20171115_144532', [2]}; %[1, 2, 3]
% %     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session3_1', 'cuxai148m1_COSMOSTrainGNG_20171115_150409', [1,2, 3]}; %[1,2,3]
% %     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session5_1', 'cuxai148m1_COSMOSTrainGNG_20171115_154259', [2]}; %%% roi1 and roi3 yielded no neurons with current settings
% }

data = {...
%    {'20171207', 'm369_f2_1', 'f2', [1]};
%    {'20171207', 'm369_f2.8_1', 'f2.8', [1]};
%    {'20171207', 'm369_f2.8_defocus500um_1', 'f2.8-500um', [1]};
% %    {'20171207', 'm369_f4_1',  'f4', [1]};
% %    {'20171207', 'm369_f4_defocus500um_1',  'f4-500um', [1]};
% %    {'20171207', 'm369_f5.6_1', 'f5.6', [1]};
% %    {'20171207', 'm369_f5.6_defocus500um_1', 'f5.6-500um', [1]};
% %    {'20171207', 'm369_f8_defocus500um_1', 'f8-500um', [1]};
% %    {'20171207', 'm369_f1.4_defocus500um_1', 'f1.4-500um', [1]};
%     {'20171207', 'm369_f8_1', 'f8', [1]};
    {'20171207', 'm369_f2_defocus500um_1', 'f2-500um', [1]}; %%%To import
    {'20171207', 'm369_f1.2_1', 'f1.4', [1]}; %%%To import
}

% data = {...
%     {'20171228', 'f1.4_cux2m1_1', 'f1.4', [1, 2]};
%     {'20171228', 'f2_cux2m1_1', 'f2', [1, 2]};
%     {'20171228', 'f2.8_cux2m1_1', 'f2.8', [1, 2]};
%     {'20171228', 'f4_cux2m1_1', 'f4', [1, 2]};
%     {'20171228', 'f5.6_cux2m1_1', 'f5.6', [1, 2]};
%     {'20171228', 'f8_cux2m1_1', 'f8', [1, 2]};
%     {'20171228', 'v10_cux2m1_1', 'v10', [1, 2]};
%     {'20171228', 'v11_cux2m1_1', 'v11', [1, 2]};
% }

data = {...   
    {'20171230', 'f5.6_cux2m1_1_1', 'f5.6-1', [1, 2], 1};
    {'20171230', 'f1.4_cux2m1_1_1', 'f1.4-1', [1, 2], 1};
    {'20171230', 'f2_cux2m1_1_1', 'f2-1',[1, 2],  1};
    {'20171230', 'f2.8_cux2m1_1_1', 'f2.8-1', [1, 2], 1};
    {'20171230', 'f4_cux2m1_1_1', 'f4-1', [1, 2], 1};
    {'20171230', 'f8_cux2m1_1_1', 'f8-1', [1, 2], 1};
    {'20171230', 'v10_cux2m1_1_1', 'v10-1', [1, 2], 40/50};
    {'20171230', 'f5.6_cux2m1_2_1', 'f5.6-2', [1, 2], 1};
    {'20171230', 'f1.4_cux2m1_2_1', 'f1.4-2', [1, 2], 1};
    {'20171230', 'f2_cux2m1_2_1', 'f2-2',[1, 2],  1};
    {'20171230', 'f2.8_cux2m1_2_1', 'f2.8-2',[1, 2],  1};
    {'20171230', 'f4_cux2m1_2_1', 'f4-2',[1, 2],  1};
    {'20171230', 'f8_cux2m1_2_1', 'f8-2', [1, 2], 1};
    {'20171230', 'v10_cux2m1_2_1', 'v10-2', [1, 2], 40/50};
    {'20171230', 'f5.6_cux2m1_3_1', 'f5.6-3', [1, 2], 1};
    {'20171230', 'f1.4_cux2m1_3_1', 'f1.4-3',[1, 2],  1};
    {'20171230', 'f2_cux2m1_3_1', 'f2-3',[1, 2],  1};
    {'20171230', 'f2.8_cux2m1_3_1', 'f2.8-3',[1, 2],  1};
    {'20171230', 'f4_cux2m1_3_1', 'f4-3', [1, 2], 1};
    {'20171230', 'f8_cux2m1_3_1', 'f8-3', [1, 2], 1};
    {'20171230', 'v10_cux2m1_3_1', 'v10-3',[1, 2],  40/50};
}

data = {...
   {'20180102', 'm52_1wk_post_tamox_2', '', [1, 2]};
}

data = {...
   {'20180108', 'm52_2wk_post_tamox_real_1', '', [1, 2]};
}

data = {...
    {'20180124', 'cux2ai148m72_vis_stim_1', '', [0, 1]};
}

errorLog = {}
tt = tic
for kk = 1:numel(data)
    d = data{kk};

    cropSaveDir =  fullfile(saveDir, d{1}, d{2})

    for patch = d{4}
        if useProcessedPatches
%             patchDir = fullfile(cropSaveDir, ['patch-', num2str(patch)], num2str(patch), 'vid.tif')
            patchDir = fullfile(cropSaveDir, num2str(patch), [num2str(patch), '.tif'])
        else
            patchDir = fullfile(baseDir, d{1}, d{2}, 'vid.tif')
        end
        
         try
            neuron = run_cnmfe(patchDir);
            
            vid = LoadImageStack(neuron.file, 1:10);
%             neuron.P.meanFrame = mean(vid, 3); 
            
            neuron = struct(neuron);
            neuron.meanFrame = mean(vid, 3);    
            tic
            save([neuron.P.folder_analysis, '/out.mat'],'-v6','-struct','neuron');
            toc
            
         catch
             errorLog{kk} = patchDir;
             disp(['---------->Something happened with ', patchDir])
         end
    end
end
toc