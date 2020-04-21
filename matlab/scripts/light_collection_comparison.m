%%% Light collection comparison:
%%% Takes an ROI from an image and computes the mean intensity. 
%%% Used to create bar chart to plots comparison of light throughput
%%% of optical designs.

addpath(genpath('/home/izkula/src/COSMOS'))
addpath(genpath('~/src/COSMOS/matlab'))
close all; clear;

set(0,'defaultfigurecolor',[1 1 1])
set(0,'defaultAxesFontSize',20)
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 10);

baseDir = '/home/izkula/Data/data/'
saveDir = '/home/izkula/Dropbox/cosmos/trace_analysis/20171228'
plotDir = fullfile(saveDir, 'plots')
if ~exist(saveDir, 'dir'); mkdir(saveDir); end
if ~exist(plotDir, 'dir'); mkdir(plotDir); end


data = {...
    {'20171228', 'f1.4_cux2m1_1', 1};
    {'20171228', 'f1.4_cux2m1_500um_1', 1};
    {'20171228', 'f2_cux2m1_1', 1};
    {'20171228', 'f2_cux2m1_500um_1', 1};
    {'20171228', 'f2.8_cux2m1_1', 1};
    {'20171228', 'f2.8_cux2m1_500um_1', 1};
    {'20171228', 'f4_cux2m1_1', 1};
    {'20171228', 'f4_cux2m1_500um_1', 1};
    {'20171228', 'f5.6_cux2m1_1', 1};
    {'20171228', 'f5.6_cux2m1_500um_1', 1};
    {'20171228', 'f8_cux2m1_1', 1};
    {'20171228', 'f8_cux2m1_500um_1', 1};
    {'20171228', 'v10_cux2m1_1', 40/50};
    {'20171228', 'v10_cux2m1_500um_1', 40/50};
}

meanVals = zeros(1, numel(data));
stdVals = zeros(1, numel(data));
npixelsVals = zeros(1, numel(data));

for kk = 1:numel(data)
    d = data{kk};
   
    cropSaveDir =  fullfile(saveDir, d{1}, d{2}, 'cropCoords')
    if ~exist(cropSaveDir, 'dir'); mkdir(cropSaveDir); end
    
    dirname = fullfile(baseDir, d{1}, d{2});
    
    if ~doLoadCropCoords
        isDone = false;
        whichRoi = 1;
        allMidline = {};
        allCropCoords = {};
        allMeanImages = {};

        nCropROIs = 1;
        doSelectAllROIs = true
        roiImSaveName = fullfile(cropSaveDir, ['roi', num2str(whichRoi)])
        scaleFactor = d{3};
        [cropCoords, midline, meanImage] = SelectROIs(dirname, nCropROIs, ...
                                           doSelectAllROIs, roiImSaveName, ...
                                           scaleFactor);

        allMidline{whichRoi} = midline;
        allCropCoords{whichRoi} = cropCoords;
        allMeanImages{whichRoi} = meanImage;
        save(fullfile(cropSaveDir, 'allCropCoords.mat'), 'd', 'allCropCoords', 'allMeanImages', 'allMidline');
    else
         load(fullfile(cropSaveDir, 'allCropCoords.mat'), 'd', 'allCropCoords', 'allMeanImages', 'allMidline');
    end

                                   
                                   
    %%%% Save out image of rois
    doSaveImROI = true
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
    
    
    numFrames = 200;
    startFrame = 1
    frames = [startFrame:numFrames + startFrame];
    vid = LoadImageStack(dirname, frames);
    
    cc = round(cropCoords(1,:)); 
    dataCrop = vid(cc(2):cc(2)+cc(4), cc(1):cc(1)+cc(3),:);
    
    meanVals(kk) = mean(scaleFactor*dataCrop(:));
    stdVals(kk) = std(scaleFactor*dataCrop(:));
    npixelsVals(kk) = numel(size(dataCrop, 1)*size(dataCrop, 2));
end

%%
labels = {}
iter = 1
for kk = 1:2:numel(data)
    d = data{kk}
    labelFull = d{2};
    labelShort = strsplit(labelFull, '_')
    
    labels{iter} = labelShort{1}
    iter = iter + 1;
end

figure('Position', [100, 100, 400, 300])
for kk = 1:2:numel(data) 
 hold on
 mm = meanVals(kk);
 sem = stdVals(kk)./sqrt(npixelsVals(kk)/numFrames)
 bar(kk+0.1, mm, 'g');
    hErrorbar = errorbar(kk+0.1, mm, sem, sem, '.k');
    set(hErrorbar, 'marker', 'none')
end
for kk = 2:2:numel(data)
 hold on
 mm = meanVals(kk);
 sem = stdVals(kk)./sqrt(npixelsVals(kk)/numFrames)
 bar(kk-0.1, mm, 'c');
    hErrorbar = errorbar(kk-0.1, mm, sem, sem, '.k');
    set(hErrorbar, 'marker', 'none')
end
 
xticks([1:2:numel(data)]+0.5)
ylabel('Arbitrary units')
set(gca,'xticklabel',labels, 'FontSize', 12)
title('Mean light density', 'FontSize', 12)
box on
print('-depsc',fullfile(saveDir, ['light_collection_comparison.eps']))
