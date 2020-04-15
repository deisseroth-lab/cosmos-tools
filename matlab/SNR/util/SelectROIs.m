function [cropCoords, midline, meanImage] = SelectROIs(dirname, nCropROIs, ...
                                                   doSelectAllROIs, roiImSaveName, ...
                                                   scaleFactor)
% SelectROIs
% - Loads a few frames from the specified dataset
% - Provides GUI interface for selecting ROIs
% - Saves out image of the ROIs
% Params:
% - dirname: location of the specified video from which to select ROI
% - nROIs: number of ROIs to select
% - doSelectAllROIs: select each ROI individually, as opposed to gridding
%                     between two specified points. 
% - cropSaveDir: directory for saving out an image of the selected ROIs
% - scaleFactor: if you are hardcoding the size of the ROI, this adjusts
% that  by the specified scale factor

    if ~exist('scaleFactor', 'var') || isempty(scaleFactor)
        scaleFactor = 1;
    end
    
    scaleFactor = double(scaleFactor); 

    %%% Load video and select crop coords
    currImages = LoadImageStack(dirname, 1:10);
    meanImage = mean(currImages, 3);
    figure('Position', [100, 100, 2000, 2000]), 
    imagesc(log(meanImage));

   
    useHardcodedSize = true;
    
    cropCoords = zeros(nCropROIs, 4);
    if doSelectAllROIs %%%% Select all of the ROIs individually
        disp(['Select ', num2str(nCropROIs), ' square ROIs (hold shift while dragging for square).']);
        title(['Select ', num2str(nCropROIs), ' square ROIs (hold shift while dragging for square).'])
        for nn = 1:nCropROIs
            cc = getrect;
            cropCoords(nn, :) = cc;
            hold on; rectangle('Position', cc);
        end
    else %%%% Just select the limits of all of the ROIs and interpolate

        title(['Select bottom left edge of left-most rectangle']);
        firstPt = ginput(1) %%% select bottom left edge left most rectangle
        
        if useHardcodedSize
            width = 75*scaleFactor;
            height = 200*scaleFactor;
            cropCoords(:, 1) = firstPt(1);
            cropCoords(:, 2) = firstPt(2) - height;
            cropCoords(:, 3) = width;
            cropCoords(:, 4) = height;
        else
            title(['Select top right edge of right-most rectangle']);
            secondPt = ginput(1); %%% select top right edge of right most rectangle
            width = floor(abs(secondPt(1) - firstPt(1))*0.95/nCropROIs);
            height = floor(abs(firstPt(2) - secondPt(2)));
            cropCoords(:, 1) = linspace(firstPt(1), secondPt(1)-width, nCropROIs);
            cropCoords(:, 2) = firstPt(2) - height;
            cropCoords(:, 3) = width;
            cropCoords(:, 4) = height;
        end
        for nn = 1:nCropROIs
            hold on; rectangle('Position', cropCoords(nn,:));
        end
    end


    title(['Click on midline.'])
    midline = ginput(1);

    title(roiImSaveName)
    print('-dpng', roiImSaveName)
%     print('-dpng', fullfile(cropSaveDir, ['template_', num2str(kk)]))
    gcf(); close