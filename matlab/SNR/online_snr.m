%%%% This is the ORIGINAL, LEGACY script to compute the PNR (peak-to-noise ratio) and correlation image 
%%%% for a set of cropped patches of test videos. 

addpath(genpath('~/src/COSMOS/matlab'))
addpath('~/Software/CNMF_E-master')
cnmfe_setup
close all; clear;


%%% Select which operations to run. %%%
doSelectCrop = false
doLoad = true
doComputeSNR = true
doPlot = true
isRotated = false %%% The 20170713 were rotated 
doSelectAllROIs = true

doSaveCroppedVideos = false; %%%% Takes a long time. Don't do this generally.

baseDir = '/home/izkula/Data/data/';
processedDataDir = '/home/izkula/Data/processedData/';
roiNum = 30
saveDir = ['/home/izkula/Data/Results/COSMOS_SNR/20171017_', num2str(roiNum)];
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

%%% {'date', 'dir_name', 'label', templateInd}, where templateInd specifies
%%% which datasets can be grouped together (i.e. for crop coordinates).
% data = {...
%     {'20170713', 'cux2_post_8dayspst_tamox_10hz_50mW_f4_1', 'f4', 1};...
%     {'20170713', 'cux2_post_8dayspst_tamox_10hz_50mW_f2.8_1', 'f2.8', 1}; 
%     {'20170713', 'cux2_post_8dayspst_tamox_10hz_50mW_f5.6_1', 'f5.6', 1}; ...
%     {'20170713', 'cux2_post_8dayspst_tamox_10hz_50mW_f8_1', 'f8', 1}; ...
% };


%%%% This is the f-stop dataset you presented to Gordon the first time. 
data = {
    {'20171016', 'cux2ai148m1_2x2bin_50mW_f1.2_1', 'f1.2-50', 1};...
    {'20171016', 'cux2ai148m1_2x2bin_50mW_f2_1', 'f2-50', 1};...
% % % % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_f2.8_1', 'f2.8-50', 1};...
% % % % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_f4_1', 'f4-50', 1};...
% % % % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_f5.6_1', 'f5.6-50', 1};...
% % % % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_f8_1', 'f8-50', 1};...
% % % % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_f12_1', 'f12-50', 1};...
% % % % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_f16_1', 'f16-50', 1};...
}


% data = {
%     {'20171030', '20171030_top_v4.0_COSMOSTestSNR_50mW_f1.2_1', 'top-f1.2', 1};...
%     {'20171030', '20171030_side_v4.0_COSMOSTestSNR_50mW_f1.2_1', 'side-f1.2', 2};...
% }

% data = {...
% %     {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f1.2_1', 'f1.2', 1};...
% %     {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f2_1', 'f2', 1};...
% %     {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f2.8_1', 'f2.8', 1};...
% %      {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f4_1', 'f4', 1};...
% %      {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f5.6_1', 'f5.6', 1};...
% % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f8_1', 'f8', 1};...
% % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f12_1', 'f12', 1};...
% % %     {'20171016', 'cux2ai148m1_2x2bin_50mW_tk2_f16_1', 'f16', 1};...
% % %     
% % %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f1.2_1', 'f1.2-200', 1};...
% % %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f2_1', 'f2-200', 1};...
% % %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f2.8_1', 'f2.8-200', 1};...
% %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f4_1', 'f4-200', 1};...
% %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f5.6_1', 'f5.6-200', 1};...
% %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f8_1', 'f8-200', 1};...
% %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f12_1', 'f12-200', 1};...
% %     {'20171016', 'cux2ai148m1_2x2bin_200mW_f16_1', 'f16-200', 1};...
% }


%% Select (or load) coordinates for cropping small patches of each video

cropSaveDir =  fullfile(saveDir, 'cropCoords')
if ~exist(cropSaveDir, 'dir'); mkdir(cropSaveDir); end

if doSelectCrop
    templates = {}
    for kk = 1:numel(data)
        d = data{kk};
        templateInd = d{4};
        if numel(templates) < templateInd || isempty(templates{templateInd})
            templates{templateInd} = kk;
        end
    end
    
    allCropCoords = {}
    allMeanImages = {};
    for kk = 1:numel(templates)
        if ~isempty(templates{kk})
            %%% Load video and select crop coords
            d = data{templates{kk}}
            dirname = fullfile(baseDir, d{1}, d{2});

            currImages = LoadImageStack(dirname, 1:10);
            meanImage = mean(currImages, 3);
            figure('Position', [100, 100, 2000, 2000]), 
            imagesc(meanImage);

            nCropROIs = 4;
            disp(['Select ', num2str(nCropROIs), ' square ROIs (hold shift while dragging).']);
            title(['Select ', num2str(nCropROIs), ' square ROIs (hold shift while dragging).'])

            cropCoords = zeros(nCropROIs, 4);
            if doSelectAllROIs %%%% Select all of the ROIs individually
                for nn = 1:nCropROIs
                    cc = getrect;
%                     cc(3) = 35;
%     %                 cc(4) = 250;
%                     cc(4) = 200;
                    cropCoords(nn, :) = cc;
%                     cropCoords(nn, 2) = cropCoords(1,2);
                    cc = cropCoords(nn, :);
                    hold on; rectangle('Position', cc);
                end
            else %%%% Just select the limits of all of the ROIs and interpolate
                firstPt = ginput(1) %%% select bottom left edge left most rectangle
                secondPt = ginput(1); %%% select top right edge of right most rectangle
                width = 50;
                height = 200;
                cropCoords(:, 1) = linspace(firstPt(1), secondPt(1)-width, nCropROIs);
                cropCoords(:, 2) = firstPt(2) - height;
                cropCoords(:, 3) = width;
                cropCoords(:, 4) = height;
                for nn = 1:nCropROIs
                    hold on; rectangle('Position', cropCoords(nn,:));
                end
            end
                
                
            title(['Click on midline.'])
            midline = ginput(1);

            title(['template #', num2str(kk)])
            print('-dpng', fullfile(cropSaveDir, ['template_', num2str(kk)]))
            gcf(); close
        end
        allMidline{kk} = midline;
        allCropCoords{kk} = cropCoords;
        allMeanImages{kk} = meanImage;
       
    end
    save(fullfile(cropSaveDir, 'allCropCoords.mat'), 'data', 'allCropCoords', 'templates', 'allMeanImages', 'allMidline');
else
    load(fullfile(cropSaveDir, 'allCropCoords.mat'), 'allCropCoords', 'allMeanImages', 'allMidline', 'templates');
end



%% Compute SNR for the cropped patches for each dataset

allPatchesCn = cell(numel(data),1);
allPatchesPNR = cell(numel(data),1);
for kk = 1:numel(data)
    d = data{kk}
    cropTemplate = d{4};
    cropCoords = round(allCropCoords{cropTemplate});
    if doLoad
        useMat = false;

        % Number of frames to process for estimating PNR/correlation image
        numFrames = 600
        startFrame = 1

            dirname = fullfile(baseDir, d{1}, d{2});
%            dirname = fullfile(baseDir, d{1}, d{2}, 'Pos0');

        if useMat
            snr_choose_data;
        else
            
%             [vid, d1, d2, numFrame] = snr_choose_data_stack(dirname, [startFrame:numFrames+startFrame]);
            
            frames = [startFrame:numFrames + startFrame];
            vid = LoadImageStack(dirname, frames);
%             vid = LoadImageStack_orig(dirname, frames);

            vid = vid(:,:,1:end-1);
            [d1, d2, numFrame] = size(vid);
        end
        fprintf('\nThe data has been loaded into RAM. It has %d X %d pixels X %d frames. \nLoading all data requires %.2f GB RAM\n\n', d1, d2, numFrames, d1*d2*numFrame*8/(2^30));
    end

    %%% load small portion of data for computing correlation image
    nPatches = size(cropCoords,1);
    patchesCn = cell(nPatches, 1);
    patchesPNR = cell(nPatches, 1);
    for patch = 1:nPatches
        doCrop = true
        if doCrop
            if useMat
                Y = double(vid.Y(:, :, 1+(1:numFrames)-1));
            else
                Y = vid;
            end
            
            cc = cropCoords(patch,:); 
%             dataCrop = Y(100:1300, 100:1300, :);
            dataCrop = Y(cc(2):cc(2)+cc(4), cc(1):cc(1)+cc(3),:);
            figure, imagesc(dataCrop(:,:,1)) %%% Is the above line swapped?

              %%% Originally was doing this --- indices seems swapped.  
%             dataCrop = Y(cc(1):cc(1)+cc(3), cc(2):cc(2)+cc(4),:); 
%               imagesc(vid(cc(1):cc(1)+cc(3), cc(2):cc(2)+cc(4),1)) %%% 
            
            Y = dataCrop;

            d1 = size(Y, 1);
            d2 = size(Y, 2);

            neuron = Sources('d1',d1,'d2',d2, ... % dimensions of datasets
                             'ssub', 1, 'tsub', 1, ...  % downsampling
                             'gSig',  4, 'gSiz',  10);  % average neuron size
            neuron.Fs = 30; % frame rate

            Y = neuron.reshape(Y,1);
        end
        
        if doSaveCroppedVideos
            dirname = fullfile(processedDataDir, d{1}, d{2}, ['ROIs_', num2str(roiNum)]);
            if ~exist(dirname, 'dir'); mkdir(dirname); end
            patchdirname = fullfile(dirname, num2str(patch));
            fnames = SaveMultipageTiff( dataCrop, patchdirname, 'vid');
            save([dirname, '_cropCoords.mat'], 'cropCoords');
        end

        if doComputeSNR
            tic
            doPlot = false;
            [Cn, PNR] = snr_compute(Y, neuron, numFrames, doPlot);
            toc
            patchesCn{patch} = Cn;
            patchesPNR{patch} = PNR;
        end
    end
    allPatchCoords{kk} = cropCoords;
    allPatchMidlines{kk} = allMidline{cropTemplate};
    allPatchesCn{kk} = patchesCn;
    allPatchesPNR{kk} = patchesPNR;
    allLabels{kk} = d{3};
end


%% Compare results

nPatches = numel(allPatchesCn{1});

meanCn = zeros(numel(data), nPatches);
meanPNR = zeros(numel(data), nPatches);
stdCn = zeros(numel(data), nPatches);
stdPNR = zeros(numel(data), nPatches);
allX = zeros(numel(data), nPatches);
allY = zeros(numel(data), nPatches);
for kk = 1:numel(data)
    for patch = 1:nPatches
        patchX = allPatchCoords{kk}(patch, 1);
        patchY = allPatchCoords{kk}(patch, 2);
        midlineX = allPatchMidlines{kk}(1);
        midlineY = allPatchMidlines{kk}(2);
        allX(kk, patch) = patchX - midlineX;
        allY(kk, patch) = patchY - midlineY;
        currCn = allPatchesCn{kk}{patch};
        currPNR = allPatchesPNR{kk}{patch};
        meanCn(kk, patch) = mean(mean(currCn(:)));
        meanPNR(kk, patch) = mean(mean(currPNR(:)));
        stderrCn(kk, patch) = std(currCn(:))/sqrt(numel(currCn));
        stderrPNR(kk, patch) =  std(currPNR(:))/sqrt(numel(currPNR));
%         stderrCn(kk, patch) = std(currCn(:));
%         stderrPNR(kk, patch) =  std(currPNR(:));
    end
end

pixeldx = 2*6.5e-3; %%% in mm.

if isRotated
    allDist = allY;
else
    allDist = allX;
end

figure('Position', [100, 100, 3500, 1000])
cmap = hsv(numel(data));
for kk = 1:numel(data)
   subplot(131)
   errorbar(pixeldx*allDist(kk,:), meanCn(kk,:),stderrCn(kk, :), 'Color', cmap(kk,:), 'LineWidth',2); hold on;
   set(gca(), 'FontSize', 20)
   xlabel('distance from midline [mm]', 'FontSize', 20)
   title('Cn', 'FontSize', 20)
   subplot(132)
   errorbar(pixeldx*allDist(kk,:), meanPNR(kk,:), stderrPNR(kk, :), 'Color', cmap(kk,:), 'LineWidth',2); hold on;
   title('PNR', 'FontSize', 20) 
   set(gca(), 'FontSize', 20)
   xlabel('distance from midline [mm]', 'FontSize', 20)
   subplot(133)
   errorbar(pixeldx*allDist(kk,:), meanPNR(kk,:).*meanCn(kk,:), sqrt(stderrPNR(kk,:).^2 + stderrCn(kk,:).^2), 'Color', cmap(kk,:), 'LineWidth',2); hold on;
   title('Cn*PNR', 'FontSize', 20) 
   set(gca(), 'FontSize', 20)
   xlabel('distance from midline [mm]', 'FontSize', 20)
end
subplot(132)
legend(allLabels, 'Location', 'NorthWest', 'FontSize', 14)
export_fig(fullfile(saveDir, 'pnr_cn_vs_dist_to_midline.pdf'))


%% Plot actual Cn and PNR images

for kk = 1:numel(data)
    figure
    if ~isRotated
        subplot(141)
    else
        subplot(411),
    end
    imagesc(allPatchesCn{kk}{1})
     	axis image
    title(['Cn ', data{kk}{2}])


    if ~isRotated
        subplot(142)
    else
        subplot(412)
    end
    imagesc(allPatchesCn{kk}{2})
     	axis image

    if ~isRotated
        subplot(143)
    else
        subplot(413)
    end
    imagesc(allPatchesCn{kk}{3})
     	axis image

    if ~isRotated
        subplot(144)
    else
        subplot(414)
    end
    imagesc(allPatchesCn{kk}{4})
     	axis image

    
    figure
    if ~isRotated
        subplot(141)
    else
        subplot(411),
    end
    imagesc(allPatchesPNR{kk}{1})
     	axis image
    title(['PNR ', data{kk}{2}])

    if ~isRotated
        subplot(142)
    else
        subplot(412)
    end
    imagesc(allPatchesPNR{kk}{2})
     	axis image
    if ~isRotated
        subplot(143)
    else
        subplot(413)
    end
    imagesc(allPatchesPNR{kk}{3})
         	axis image

    if ~isRotated
        subplot(144)
    else
        subplot(414)
    end
    imagesc(allPatchesPNR{kk}{4})
     	axis image
end

%% Plot overlay ROIs nicely

for kk = 1:numel(templates)
        if ~isempty(templates{kk})
            %%% Load video and select crop coords
            d = data{templates{kk}}
            dirname = fullfile(baseDir, d{1}, d{2});

            currImages = LoadImageStack(dirname, 1:10);
            meanImage = mean(currImages, 3);
            figure('Position', [100, 100, 2000, 2000]), 
            imagesc(meanImage);      
            colormap('gray')
            
            cropCoords = allCropCoords{kk};

            for nn = 1:size(cropCoords, 1)
                hold on; rectangle('Position', cropCoords(nn,:), 'EdgeColor', 'r');
            end
            export_fig(fullfile(saveDir, ['rois_template_', num2str(kk), '.pdf']))
            mkdir(fullfile(processedDataDir, d{1}, d{2}))
            export_fig(fullfile(processedDataDir, d{1}, d{2}, ['ROIs_', num2str(roiNum)]));
        end
end
roisDir = fullfile(saveDir, 'allROIs');
if ~exist(roisDir, 'dir'); mkdir(roisDir); end

doPlotAllROIs = false
if doPlotAllROIs
    for kk = 1:numel(data)
            %%% Load video and select crop coords
            d = data{kk}
            dirname = fullfile(baseDir, d{1}, d{2}, 'Pos0');

            currImages = LoadImageStack(dirname, 1:10);
            roimeanImage = mean(currImages, 3);
            figure('Position', [100, 100, 2000, 2000]), 
            imagesc(histeq(roimeanImage/max(roimeanImage(:))));         
            colormap('gray')

            for nn = 1:size(cropCoords, 1)
                hold on; rectangle('Position', cropCoords(nn,:), 'EdgeColor', 'r');
            end
            export_fig(fullfile(roisDir, ['rois_', d{3}, '.pdf']))
    end
end
