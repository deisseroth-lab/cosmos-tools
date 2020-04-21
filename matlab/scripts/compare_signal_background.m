%%%% compare_signal_background.m
%%%% This function extracts the signal and background for a number of user
%%%% selected neurons, (using MANUAL, square rois)
%%%% to compute the total signal vs. the average background per pixel in
%%%% different regions of the window, for different f-numbers.



addpath(genpath('/home/izkula/src/COSMOS'))


baseDir = '/home/izkula/Data/data'
processedDataDir = '/home/izkula/Data/processedData'
saveDir = '/home/izkula/Dropbox/cosmos/SNR_analysis/background_vs_signal/20171207'
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

useProcessedPatches = false;

% data = {...
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session2_1', [1, 2, 3], 'f4'};
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session3_1', [1, 2, 3], 'f1.4'};
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session5_1', [1, 2, 3], 'f8'};
% }

data = {...
    {'20171207', 'm369_f1.2_1', [1], 'f1.2'};
    {'20171207', 'm369_f2_1', [1], 'f2'};
    {'20171207', 'm369_f2.8_1', [1], 'f2.8'};
    {'20171207', 'm369_f4_1', [1], 'f4'};
    {'20171207', 'm369_f8_1', [1], 'f8'};
}

data = {...
    {'20171207', 'm369_f1.2_1', [1], 'f1.4'};
    {'20171207', 'm369_f1.4_defocus500um_1', [1], 'f1.4-500um'};
    {'20171207', 'm369_f2_1', [1], 'f2'};
    {'20171207', 'm369_f2_defocus500um_1', [1], 'f2-500um'};
    {'20171207', 'm369_f2.8_1', [1], 'f2.8'};
    {'20171207', 'm369_f2.8_defocus500um_1', [1], 'f2.8-500um'};
    {'20171207', 'm369_f4_1', [1], 'f4'};
    {'20171207', 'm369_f4_defocus500um_1', [1], 'f4-500um'};
    {'20171207', 'm369_f5.6_1', [1], 'f5.6'};
    {'20171207', 'm369_f5.6_defocus500um_1', [1], 'f5.6-500um'};
    {'20171207', 'm369_f8_1', [1], 'f8'};
    {'20171207', 'm369_f8_defocus500um_1', [1], 'f8-500um'};
}

%%

%%% For each patch, load patch, select neuron locations (using max of
%%% min-sub?), extract traces of a 5x5 rectangle. Compute max signal based
%%% on the peak in the sum across the ROI. Compute the noise floor
%%% intensity per pixel based on the baseline.

%%% Compare baseline noise values across ROIs, and locations within an ROI.

whichPatch = []
whichDataset = [];
allLocs = [];
allSumTraces = [];
allMeanTraces = [];
allNpixels = [];

averageBackground = [];
totalPeakSignal = [];
iter = 1

nFrames = 2000;
%%% Manually select neurons
for kk = 1:numel(data)
    d = data{kk};
    
    for patch = d{3}
        if useProcessedPatches
            dirname = fullfile(processedDataDir, d{1}, d{2}, ['patch-', num2str(patch)], num2str(patch));
        else
            dirname = fullfile(baseDir, d{1}, d{2})
        end
        currImages = LoadImageStack(dirname, [1:nFrames]);
        minImage = min(currImages, [], 3);
        minSubImage = currImages - repmat(minImage, 1, 1, size(currImages, 3));
        
        figure, imagesc(max(minSubImage, [], 3));
        title('Click on neurons. Press enter when done.')
        [x, y] = ginput;
        
        for loc = 1:numel(x)
            whichPatch(iter) = patch;
            whichDataset(iter) = kk;
            allLocs(iter, :) = [x(loc), y(loc)];
            iter = iter+1;
        end
    end
end

save(fullfile(saveDir, 'rois.mat'), 'data', 'whichPatch', 'whichDataset', 'allLocs', 'processedDataDir')    
%% Now extract ROIs

roiSumTraces = {};
roiMeanTraces = {};
roiNpixels = [];

iter = 1
for nPixelsROI = [2, 4, 6, 8, 10]
    nFrames = 2000;
    for ii = 1:size(allLocs, 1)
        d = data{whichDataset(ii)};
        if useProcessedPatches
            dirname = fullfile(processedDataDir, d{1}, d{2}, ['patch-', num2str(patch)], num2str(patch));
        else
            dirname = fullfile(baseDir, d{1}, d{2})
        end
        currImages = LoadImageStack(dirname, [1:nFrames]);

        x = allLocs(ii,1);
        y = allLocs(ii, 2);
        [meanTrace, sumTrace, nPixels] = ExtractTraceROI(currImages, x, y, nPixelsROI);

         allSumTraces(ii, :) = sumTrace;
         allMeanTraces(ii, :) = meanTrace;
         allNpixels(ii) = nPixels;
    end
    roiSumTraces{iter} = allSumTraces;
    roiMeanTraces{iter} = allMeanTraces;
    roiNpixels(iter) = nPixels;
    iter = iter + 1;
end


save(fullfile(saveDir, 'traces.mat'), 'data', 'whichPatch', 'whichDataset', 'allLocs', 'allSumTraces', 'allMeanTraces', 'allNpixels', 'processedDataDir', 'roiSumTraces', 'roiMeanTraces', 'roiNpixels')

%% Extract signal and background
for ii = 1:size(allSumTraces, 1)
  averageBackground(ii) = quantile(allMeanTraces(ii, :), 0.5);
  totalPeakSignal(ii) = quantile(allSumTraces(ii, :), 0.99) - quantile(allSumTraces(ii, :), 0.50);
%   figure, plot(allMeanTraces(ii, :)); hold on; plot([1:size(allMeanTraces, 2)], averageBackground(ii)*ones(size(allMeanTraces, 2),1), 'r-'), title(data{whichDataset(ii)}{2});
%   figure, plot(allSumTraces(ii, :)/allNpixels(ii)); hold on; title(data{whichDataset(ii)}{2})
%           plot(allSumTraces10(ii,:)/allNpixels10(ii), 'r')
end


    
%% Make plots
fsize = 10
%%%% for an ROI, plot signal vs. f/# (plot all datapoints for a given ROI)

plotInds = 2:2:numel(data)
plotInds = 1:numel(data)

cc = hsv(numel(data));

for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk*ones(numel(inds)), averageBackground(inds), 'o', 'Color', cc(kk, :));
        hold on
        plot(kk, mean(averageBackground(inds)), '*', 'Color', cc(kk, :), 'MarkerSize', 5);

        title(['average background, patch = ', num2str(patch)])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['averagedBackground.pdf']))
end

for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk*ones(numel(inds)), totalPeakSignal(inds), 'o', 'Color', cc(kk, :));
        hold on
        plot(kk, mean(totalPeakSignal(inds)), '*', 'Color', cc(kk, :), 'MarkerSize', 5);

        title(['total peak signal, patch = ', num2str(patch)])
    end
%     ylim([0, 5e5])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['peakSignal.pdf']))
end


for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk*ones(numel(inds)), totalPeakSignal(inds)./averageBackground(inds), 'o', 'Color', cc(kk, :));
        hold on
        plot(kk, mean(totalPeakSignal(inds)./averageBackground(inds)), '*', 'Color', cc(kk, :), 'MarkerSize', 5);

        title(['peak signal / background, patch = ', num2str(patch)])
    end
%     ylim([0, 5e5])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['peakSignal_over_averageBackground_.pdf']))
end


for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk*ones(numel(inds)), totalPeakSignal(inds)./sqrt(averageBackground(inds)), 'o', 'Color', cc(kk, :));
        hold on
        plot(kk, mean(totalPeakSignal(inds)./sqrt(averageBackground(inds))), '*', 'Color', cc(kk, :), 'MarkerSize', 5);

        title(['peak signal / sqrt(background), patch = ', num2str(patch)])
    end
%     ylim([0, 5e5])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['peakSignal_over_sqrtBackground_.pdf']))
end


%% Compare allSumTraces divided by number of pixels
averageBackgroundROI = zeros(size(allSumTraces, 1), numel(roiSumTraces));
totalPeakSignalROI = zeros(size(allSumTraces, 1), numel(roiSumTraces));
totalPeakVarianceROI = zeros(size(allSumTraces, 1), numel(roiSumTraces));

figure
for roi = 1:numel(roiSumTraces)
    allSumTraces = roiSumTraces{roi};
    allMeanTraces = roiMeanTraces{roi};
    allNpixels = roiNpixels(roi);
    for ii = 1:size(allSumTraces, 1)
      averageBackgroundROI(ii, roi) = quantile(allMeanTraces(ii, :), 0.5);
      totalPeakSignalROI(ii, roi) = (quantile(allSumTraces(ii, :), 0.99) - quantile(allSumTraces(ii, :), 0.50));
      totalPeakVarianceROI(ii, roi) = quantile(allSumTraces(ii, :), 0.99);
      if ii == 20
          plot(allSumTraces(ii, :)); hold on
      end
%       figure, plot(allSumTraces(ii, :)/allNpixels(ii)); hold on; title(data{whichDataset(ii)}{2})
%               plot(allSumTraces10(ii,:)/allNpixels10(ii), 'r')
    end
    legend({'9', '25', '49', '81', '121'})
end


%% Plot total signal for different sized ROIs around neuron

cc = parula(numel(data)*2);
plotInds = 1:numel(data)
fsize = 10
for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk+linspace(0, 0.5, size(totalPeakSignalROI, 2)), mean(totalPeakSignalROI(inds, :), 1), '-o', 'Color', cc(kk, :), 'Markersize', 10);

        hold on

        title([''])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    ylabel('Total signal for various ROI sizes')
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['total_signal_roiSizeComparison.pdf']))
end



%% Plot average SNR for different sized ROIs around neuron

cc = parula(numel(data)*2);
plotInds = 1:numel(data)
fsize = 10
for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
%         plot(kk+linspace(0, 0.5, size(totalPeakSignalROI, 2)), mean(totalPeakSignalROI(inds, :), 1)./sqrt(roiNpixels.*mean(averageBackgroundROI(inds, :),1)), '-o', 'Color', cc(kk, :), 'Markersize', 10);
        plot(kk+linspace(0, 0.5, size(totalPeakSignalROI, 2)), mean(totalPeakSignalROI(inds, :), 1)./sqrt(mean(totalPeakVarianceROI(inds, :))), '-o', 'Color', cc(kk, :), 'Markersize', 10);

        hold on

        title([''])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
%     ylabel('Total signal / sqrt(average background * npixels) for various ROI sizes')
    ylabel('Peak of sum across pixels minus baseline / sqrt(peak of sum across pixels) for various ROI sizes')

    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['snr_roiSizeComparison.pdf']))
end

%% Plot average signal / total variance for different sized ROIs around neuron

cc = parula(numel(data)*2);
plotInds = 1:numel(data)
fsize = 10
for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk+linspace(0, 0.5, size(totalPeakSignalROI, 2)), mean(totalPeakSignalROI(inds, :), 1)./(roiNpixels.*mean(averageBackgroundROI(inds, :),1)), '-o', 'Color', cc(kk, :), 'Markersize', 10);

        hold on

        title([''])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    ylabel('Total signal / (average background * npixels) for various ROI sizes')
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['signal_over_variance_roiSizeComparison.pdf']))
end


%% Plot average background for different sized ROIs around neuron
cc = parula(numel(data)*2);
plotInds = 1:numel(data)
fsize = 10
for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk+linspace(0, 0.5, size(totalPeakSignalROI, 2)), mean(averageBackgroundROI(inds, :), 1), '-o', 'Color', cc(kk, :), 'Markersize', 10);
        hold on

        title([''])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    ylabel('Average background per pixel for various ROI sizes')
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['average_background_roiSizeComparison.pdf']))
end


%%

cc = hsv(numel(data)*2);

for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk*ones(numel(inds)), totalPeakSignalROI(inds, 1), 'o', 'Color', cc(kk, :));
        hold on
        plot(kk*ones(numel(inds)), totalPeakSignalROI(inds, 2), 'o', 'Color', cc(kk+numel(plotInds), :));
        
        plot(kk, mean(totalPeakSignalROI(inds, 1)), '*', 'Color', cc(kk, :));
        hold on
        plot(kk, mean(totalPeakSignalROI(inds, 2)), '*', 'Color', cc(kk+numel(plotInds), :));


        title(['average background, patch = ', num2str(patch)])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['roiSizeComparison.pdf']))
end

%%
for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk, mean(totalPeakSignalROI(inds, 1)) -  mean(totalPeakSignalROI(inds, 2)), '*', 'Color', cc(kk, :), 'Markersize', 10);
        hold on

        title(['Average signal value with small ROI - with large ROI'])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['roiSizeComparison.pdf']))
end

%%
for patch = [1]
    figure('Position', [100, 100, 1000, 1000])
    for kk = plotInds
        labels{kk} = data{kk}{4}
        inds = find(whichPatch == patch & whichDataset == kk);
        
        plot(kk, mean(totalPeakSignalROI(inds, 1))/sqrt(mean(averageBackgroundROI(inds, 1))) ...
                 -  mean(totalPeakSignalROI(inds, 2))/sqrt(mean(averageBackgroundROI(inds, 2))), '*', 'Color', cc(kk, :), 'Markersize', 10);
        hold on

        title(['Average signal value with small ROI - with large ROI (normalized by background level)'])
    end
%     ylim([0, 5e4])
    xticks(1:numel(labels))
    xlim([0, numel(labels)+1])
    set(gca,'xticklabel',labels)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['roiSizeComparison.pdf']))
end

