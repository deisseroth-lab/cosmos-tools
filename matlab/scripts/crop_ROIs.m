%%% Script to extract ROIs for cnmf trace extraction from COSMOS recording.

addpath(genpath('/home/izkula/src/COSMOS'))


baseDir = '/home/izkula/Data/data'
% baseDir =  '/media/izkula/data5/OEG_computer/'
processedDataDir = '/home/izkula/Data/processedData'
saveDir = processedDataDir

doLoadCropCoords = true
doSaveImROI = false
doProcessVideo = true
doRemoveLEDFrames = true
doCopyBpodFile = false

doSelectAllROIs = true %%% Set this if you want to manually set rectangle size
% data = {...
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session2_1', 'cuxai148m1_COSMOSTrainGNG_20171115_144532'};
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session3_1', 'cuxai148m1_COSMOSTrainGNG_20171115_150409'};
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session5_1', 'cuxai148m1_COSMOSTrainGNG_20171115_154259'};
% }

data = {...   
     {'20171228', 'f5.6_cux2m1_1', ''};
%    {'20171228', 'f1.4_cux2m1_1', ''};
%     {'20171228', 'f2_cux2m1_1', ''};
%     {'20171228', 'f2.8_cux2m1_1', ''};
%     {'20171228', 'f4_cux2m1_1', ''};
%     {'20171228', 'f8_cux2m1_1', ''};
%      {'20171228', 'v10_cux2m1_1', ''};
%     {'20171228', 'v11_cux2m1_1', ''};
}


data = {...   
     {'20171230', 'f5.6_cux2m1_1_1', '', 1};
     {'20171230', 'f1.4_cux2m1_1_1', '', 1};
     {'20171230', 'f2_cux2m1_1_1', '', 1};
    {'20171230', 'f2.8_cux2m1_1_1', '', 1};
    {'20171230', 'f4_cux2m1_1_1', '', 1};
    {'20171230', 'f8_cux2m1_1_1', '', 1};
     {'20171230', 'v10_cux2m1_1_1', '', 40/50};
    {'20171230', 'f5.6_cux2m1_2_1', '', 1};
    {'20171230', 'f1.4_cux2m1_2_1', '', 1};
    {'20171230', 'f2_cux2m1_2_1', '', 1};
    {'20171230', 'f2.8_cux2m1_2_1', '', 1};
    {'20171230', 'f4_cux2m1_2_1', '', 1};
    {'20171230', 'f8_cux2m1_2_1', '', 1};
    {'20171230', 'v10_cux2m1_2_1', '', 40/50};
    {'20171230', 'f5.6_cux2m1_3_1', '', 1};
    {'20171230', 'f1.4_cux2m1_3_1', '', 1};
    {'20171230', 'f2_cux2m1_3_1', '', 1};
    {'20171230', 'f2.8_cux2m1_3_1', '', 1};
    {'20171230', 'f4_cux2m1_3_1', '', 1};
    {'20171230', 'f8_cux2m1_3_1', '', 1};
    {'20171230', 'v10_cux2m1_3_1', '', 40/50};
}

data = {...
   {'20180102', 'm52_1wk_post_tamox_2', '', [1]};
}

%%%% The first good whole brain dataset (just spontaneous),
%%%% Where you got a 1-2k cells that showed importance
%%%% of the multiple focal planes. 
%%%% You were using this for building out trace analysis
%%%% and multi-focal merging. 
data = {...
   {'20180108', 'm52_2wk_post_tamox_real_1', '', [1]};
}

%%%% The best mouse at this point - clear across. 
data = {...
    {'20180124', 'cux2ai148m72_vis_stim_1', '', [40/50]};
    {'20180124', 'cux2ai148m72_vis_stim_2', '', [40/50]};
}


for kk = 1:numel(data)
    d = data{kk};

    cropSaveDir =  fullfile(saveDir, d{1}, d{2}, 'cropCoords')
    if ~exist(cropSaveDir, 'dir'); mkdir(cropSaveDir); end
    
    dirname = fullfile(baseDir, d{1}, d{2});
    
    
    %%%% Load or select ROIs  %%%%%
    if ~doLoadCropCoords
        isDone = false;
        whichRoi = 1;
        allMidline = {};
        allCropCoords = {};
        allMeanImages = {};
        while ~isDone
            nCropROIs = 1;
%             doSelectAllROIs = false
            roiImSaveName = fullfile(cropSaveDir, ['roi', num2str(whichRoi)])
            scaleFactor = d{4};
            [cropCoords, midline, meanImage] = SelectROIs(dirname, nCropROIs, ...
                                                   doSelectAllROIs, roiImSaveName, ...
                                                   scaleFactor);
            allMidline{whichRoi} = midline;
            allCropCoords{whichRoi} = cropCoords;
            allMeanImages{whichRoi} = meanImage;

            str = input('Add another roi? (type y)', 's')
            if strcmp(str, 'y')
                isDone = false;
                whichRoi = whichRoi + 1;
            else
                isDone = true;
            end
        end
        save(fullfile(cropSaveDir, 'allCropCoords.mat'), 'd', 'allCropCoords', 'allMeanImages', 'allMidline');
    else
        load(fullfile(cropSaveDir, 'allCropCoords.mat'), 'allCropCoords', 'allMeanImages', 'allMidline');
    end
    
    %%%% Save out image of rois
    if doSaveImROI
        figure()
        imagesc(log10(allMeanImages{1}))
        colormap(gray)
        hold on
        for ii = 1:numel(allCropCoords)
            cropCoords = allCropCoords{ii};
            rectangle('Position', cropCoords(1,:), 'EdgeColor', 'r', 'LineWidth', 2)
        end
        print('-dpng', fullfile(cropSaveDir, 'allROIs.png'))
    end
    
    %%%% Copy bpod behavior file to processedData dir %%%%%%
    if doCopyBpodFile
        try
            copyfile(fullfile(baseDir, d{1}, [d{3}, '.mat']), fullfile(saveDir, d{1}, d{2}, 'bpod.mat'))
        catch
            disp('Couldnt copy bpod file')
        end
    end
    
    
    %%%% Load and process videos  %%%%%
    if doProcessVideo
        %%%% Load video
        numFrames = 9999;
        startFrame = 1
        frames = [startFrame:numFrames + startFrame];
        vid = LoadImageStack(dirname, frames);
        
        
        %%%% Replace LED frames with the preceding non-LED frame
        if doRemoveLEDFrames
            avg = squeeze(mean(mean(vid,1), 2));
            avg = double(avg > 3*std(avg)+min(avg));
            [pkvals, LEDpeakframes] = findpeaks(avg); 
            for ff = 1:numel(LEDpeakframes)
                frame = LEDpeakframes(ff);
                vid(:,:,frame-1:frame+4) = repmat(vid(:,:,frame-2), 1, 1, 6);
            end
        end
        
        figure, plot(abs(diff(squeeze(mean(mean(vid, 1), 2)))))
        figure, plot(abs(diff(squeeze(mean(mean(vid, 1), 2))))< 1e-10)
        %%%% Save out cropped videos
        for patch = 1:numel(allCropCoords)
            cropCoords = allCropCoords{patch};
           
            cc = round(cropCoords(1,:)); 
            dataCrop = vid(cc(2):cc(2)+cc(4), cc(1):cc(1)+cc(3),:);

            doSaveCroppedVideos = true
            if doSaveCroppedVideos
                dirname = fullfile(processedDataDir, d{1}, d{2}, ['patch-', num2str(patch)]);
                if ~exist(dirname, 'dir'); mkdir(dirname); end
                patchdirname = fullfile(dirname, num2str(patch));
                keepScaling = true
                fnames = SaveMultipageTiff( dataCrop, patchdirname, 'vid', keepScaling);
                if doRemoveLEDFrames
                    save([dirname, '_cropInfo.mat'], 'cropCoords', 'LEDpeakframes');
                    save([dirname, '_LEDpeakframes.mat'], 'LEDpeakframes');
                else
                    save([dirname, '_cropInfo.mat'], 'cropCoords');
                end
            end
        end
        
        
    end
    
end
%%
% 
% 
% %%%% Load image stack
% frames = [startFrame:numFrames + startFrame];
% vid = LoadImageStack(dirname, frames);
% 
% %%%% Select crop coords
% 
% 
% %%%% Replace the LED frames with the preceding frame
% 
% 
% %%%% Save out cropped videos
% 
% if doSaveCroppedVideos
%     dirname = fullfile(processedDataDir, d{1}, d{2}, ['ROIs_', num2str(roiNum)]);
%     if ~exist(dirname, 'dir'); mkdir(dirname); end
%     patchdirname = fullfile(dirname, num2str(patch));
%     fnames = SaveMultipageTiff( dataCrop, patchdirname, 'vid');
%     save([dirname, '_cropCoords.mat'], 'cropCoords');
% end