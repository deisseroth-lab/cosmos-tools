%%% This script starts to analyze (at least, plot) the traces extracted
%%% using cnmf-e. 
%%% 20171121: This is currently setup for you to move the trace file (named
%%% 'intermediate_results.mat') to a corresponding folder in
%%% Dropbox/cosmos_data.
%%%
%%% 20180312 --- From now on, use the python notebooks for doing this sort
%%% of analysis. 
%%%
%%% ikauvar@gmail.com

usingMacbookAir = true

if usingMacbookAir
    addpath(genpath('/home/izkula/src/COSMOS'))
else
    addpath(genpath('~/src/COSMOS'))
end

set(0,'defaultfigurecolor',[1 1 1])
set(0,'defaultAxesFontSize',20)
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 10);


if usingMacbookAir
    traceDir = '/home/izkula/Dropbox/cosmos_data/'
    saveDir = '/home/izkula/Dropbox/cosmos/trace_analysis/20171115'
else
    traceDir = '~/Dropbox/cosmos_data/'
    saveDir = '~/Dropbox/cosmos/trace_analysis/20171115'
end

plotDir = fullfile(saveDir, 'plots')
if ~exist(saveDir, 'dir'); mkdir(saveDir); end
if ~exist(plotDir, 'dir'); mkdir(plotDir); end

doLoadBpod = false
doTracePlots = true %%% Summary plots of all traces (ignores behavior)
doTraceWithBehaviorPlots = false

%%%%% {'date', 'neural_vid_name', 'bpod_filename', [rois], 'label'}
% data = {...
%    {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session2_1', 'cuxai148m1_COSMOSTrainGNG_20171115_144532', [1, 2, 3], 'f4'};
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session3_1', 'cuxai148m1_COSMOSTrainGNG_20171115_150409', [1,2, 3], 'f1.2'};
% %      {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session5_1', 'cuxai148m1_COSMOSTrainGNG_20171115_154259', [2], 'f8'}; %%% roi1 and roi3 yielded no neurons with current settings
% }

data = {...
%    {'20171207', 'm369_f1.2_1', '', [1], 'f1.4'};
%    {'20171207', 'm369_f1.4_defocus500um_1', '', [1], 'f1.4-500um'};
%    {'20171207', 'm369_f2_1', '', [1], 'f2'}; 
%    {'20171207', 'm369_f2_defocus500um_1', '', [1], 'f2-500um'}; 
%    {'20171207', 'm369_f2.8_1', '', [1], 'f2.8'}; 
%    {'20171207', 'm369_f2.8_defocus500um_1', '', [1], 'f2.8-500um'};
   {'20171207', 'm369_f4_1', '', [1], 'f4'}; 
%    {'20171207', 'm369_f4_defocus500um_1', '', [1], 'f4-500um'};
%    {'20171207', 'm369_f5.6_1', '', [1], 'f5.6'}; 
%    {'20171207', 'm369_f5.6_defocus500um_1', '', [1], 'f5.6-500um'};
%    {'20171207', 'm369_f8_1', '', [1], 'f8'}; 
%    {'20171207', 'm369_f8_defocus500um_1', '', [1], 'f8-500um'}; 
}


allTraces = {};
allRawTraces = {};
allSpikes = {};
allLabels = {};
allROIs = [];
allWhichDataset = [];
datasetNames = {};


traceIter = 1;
for kk = 1:numel(data)
    
    d = data{kk}
    datasetNames{kk} = d{5};
    for patch = d{4}
        tracePath = fullfile(traceDir, d{1}, d{2}, ['patch-', num2str(patch)]);
        DD = load(fullfile(tracePath, 'intermediate_results.mat'));
        fields = fieldnames(DD);
        tracefield = fields{end}; %%% A little sketchy--the naming of the fields is weird, but it seems like the final fieldname (when ordered alphabetically) is the most recent saved trace matrix. 
        traceStruct = getfield(DD, tracefield); 
        traces = traceStruct.C;
        spikes = traceStruct.S;
        allTraces{traceIter} = traces;
        allRawTraces{traceIter} = traceStruct.C_raw;
        allSpikes{traceIter} = spikes;
        allLabels{traceIter} = [d{5}];
        allROIs(traceIter) = patch;
        allWhichDataset(traceIter) = kk;
        allName{traceIter} = [d{5}, '-roi-', num2str(patch)];
        traceIter = traceIter + 1;
    end
end


allMidlineDist = [];
allBpod = {};
allLEDpeakframes = {};
traceIter = 1;
for kk = 1:numel(data)
    d = data{kk};
    if doLoadBpod
        allBpod{kk} = load(fullfile(traceDir, d{1}, d{2}, 'bpod.mat'));
    end

    cropInfo = load(fullfile(traceDir, d{1}, d{2}, 'cropCoords', 'allCropCoords.mat'));

    
    pf = load(fullfile(traceDir, d{1}, d{2}, 'LEDpeakframes.mat'));
    allLEDpeakframes{kk} = pf.LEDpeakframes;
     

    for patch = 1:numel(cropInfo.allCropCoords)
        cropCoords = cropInfo.allCropCoords{patch};
        midline = cropInfo.allMidline{patch};
        allMidlineDist(traceIter) = abs(midline(1) - (cropCoords(1) + cropCoords(3)/2));
        traceIter = traceIter + 1;
    end
end


%% Make trace summary plots (ignoring behavior..._
if doTracePlots
    %%%% Plot top 10 traces
    dt = 1/30.98; %%% Hz
    for kk = 1:numel(allTraces)
        figure('Position', [1000, 1000, 2000, 700]), 
    %     h = waterfall(allTraces{kk}); view(0, 87)
    %     set(gca, 'ztick', []); set(gca, 'zticklabel', [])
        hold on
        nToPlot = min(size(allTraces{kk}, 1), 10);
        plotStackedTraces(allRawTraces{kk}(1:nToPlot,:), [.4,.4,.4], dt)
        plotStackedTraces(allTraces{kk}(1:nToPlot,:), 'r', dt)
        PlotVerticalLines(dt*allLEDpeakframes{allWhichDataset(kk)});
        ylabel('Neuron #')
        xlabel('Time [s]')
        title(allName{kk})
        export_fig(fullfile(saveDir, ['best_', allName{kk}, '.pdf']))
        close
    end

    %%%% Plot traces
    dt = 1/30.98; %%% Hz
    for kk = 1:numel(allTraces)
        figure('Position', [1000, 1000, 1000, 1000]), 
    %     h = waterfall(allTraces{kk}); view(0, 87)
    %     set(gca, 'ztick', []); set(gca, 'zticklabel', [])
        hold on
        plotStackedTraces(allRawTraces{kk}, [.4,.4,.4], dt)
        plotStackedTraces(allTraces{kk}, 'r', dt)
        PlotVerticalLines(dt*allLEDpeakframes{allWhichDataset(kk)});
        ylabel('Neuron #')
        xlabel('Time [s]')
        title(allName{kk})
        export_fig(fullfile(saveDir, [allName{kk}, '.pdf']))
        close
    end

    %%%% Plot correlation within a region
    for kk = 1:numel(allTraces)
        figure('Position', [1000, 1000, 1000, 1000]), 
        imagesc(corr(allTraces{kk}'))
    end


    %%%%% Plot number of recovered neurons vs. distance to midline
    nneurons = [];
    midlineDist = [];
    for kk = 1:numel(allTraces)
        roi = allROIs(kk);
        dd = allWhichDataset(kk);
        nneurons(dd, roi) = size(allTraces{kk}, 1);
        midlineDist(dd, roi) = allMidlineDist(kk);
    end
    [sorted_midlineDist, I] = sort(midlineDist, 2);
    sorted_nneurons = zeros(size(nneurons));
    for j = 1:size(nneurons, 1), sorted_nneurons(j,:) = nneurons(j, I(j,:)); end

    figure()
    pixeldx = 0.0065*2;
    for kk = 1:numel(data)
        hold on
        plot(pixeldx*sorted_midlineDist(kk, :), sorted_nneurons(kk, :), 'o-')
    end
    xlabel('ROI distance from midline [mm]')
    ylabel('# of recovered neurons')
    legend(datasetNames)
    export_fig(fullfile(saveDir, ['numNeuronCompare.pdf']))

    

%% Plot the location of the neurons
%%% TO DO TO DO

    
end


if doTraceWithBehaviorPlots
%% Load behavior data and put traces into trial structure form
[ allTraceData, allSpikeData, allTrialTypes, datasetID, whichTraceID, ...
           allLicksCell, allLickRates, whichTrial, whichNeuronID, ...
           allT, allOdorTimes, allRewardTimes, allPunishTimes, allBaselineTimes ] = COSMOSOrganizeTrialTraceData(allBpod, allTraces, allSpikes, allLEDpeakframes, allWhichDataset);


%% Get statistically modulated cells
T = allT{1}(1:size(allTraceData, 2));

dt = median(diff(T));

rewardTimes = nanmedian(allRewardTimes(:,1));
baselineTimes = median(allBaselineTimes, 1);
odorTimes = median(allOdorTimes, 1);

baselineStart = round(round(1./dt)*(baselineTimes(1)));
baselineEnd = round(round(1./dt)*(baselineTimes(2)));

baseline = [baselineStart, baselineEnd];
taskInds = round([baselineEnd, (rewardTimes*round(1./dt)+30)]) %%% Added 30 frames (i.e. one second) beyond the reward delivery

modulated_labels = zeros(numel(unique(whichNeuronID)), numel(unique(allTrialTypes)));

for trialType = unique(allTrialTypes)
    for neuron = 1:numel(unique(whichNeuronID))
        disp(num2str(neuron));
        whichRows = whichNeuronID == neuron & allTrialTypes == trialType;
        modulated_labels(neuron, trialType) = logical(taskSelectiveCOSMOS(allSpikeData(whichRows, :), dt, 0.05, baseline, 'bootstrap', taskInds));
    end
end

%% Make plots    

%%%% Plot average across cells in each ROI for each trial type
doOrderTraces = true
doOnlyShowModulated = true


traceAvgs = {};
lickAvgs = {};
rewardAvg = [];
rewardSem = [];
T = allT{1}(1:size(allTraceData, 2));
for trialType = [3]
    for roi = [1, 2, 3]
        traceAvg = zeros(numel(unique(whichNeuronID)), size(allTraceData, 2));
        normAvg = zeros(numel(unique(whichNeuronID)), size(allTraceData, 2));
        traceSEM = zeros(numel(unique(whichNeuronID)), size(allTraceData, 2));
        lickAvg =  zeros(numel(unique(whichNeuronID)), size(allTraceData, 2));
        lickSEM = zeros(numel(unique(whichNeuronID)), size(allTraceData, 2));

        for neuron = 1:numel(unique(whichNeuronID))

%              whichRows = whichNeuronID == neuron & allTrialTypes == trialType;
            whichRows = whichNeuronID == neuron & allTrialTypes == trialType & datasetID == 2 & allROIs(whichTraceID) == roi;
            ntrials = numel(find(whichRows));
            traceAvg(neuron, :) = mean(allTraceData(whichRows,:), 1);
            traceSEM(neuron, :) = std(allTraceData(whichRows,:), [], 1)/sqrt(ntrials);
            lickAvg(neuron, :) = mean(allLickRates(whichRows,:), 1);
            lickSEM(neuron, :) = std(allLickRates(whichRows,:), [], 1)/sqrt(ntrials);

            normAvg(neuron, :) = traceAvg(neuron, :)/max(traceAvg(neuron, :));

            rewardAvg(neuron) = nanmean(allRewardTimes(whichRows, 1));
            rewardSem(neuron) = nanstd(allRewardTimes(whichRows, 1))/sqrt(ntrials);
        end

        if doOnlyShowModulated
            traceAvg = traceAvg(find(modulated_labels(:, trialType)), :);
            lickAvg = lickAvg(find(modulated_labels(:, trialType)), :);
        end
        
        keepNeurons = find(any(~isnan(traceAvg), 2)); 
        traceKeep = traceAvg(keepNeurons,:);
        
        XL = [2, 9];
        figure
        subplot(211)
        if doOrderTraces
            orderedInds = plotOrderedTraces(traceAvg(keepNeurons,:), T);
        else
            imagesc(T, 1:numel(keepNeurons), traceAvg(keepNeurons, :))
        end
        PlotVerticalLines([odorTimes(1), odorTimes(2), rewardTimes(1)])
        xlim(XL)
        ylabel('Neuron #')
        subplot(212)
        plot(T, mean(lickAvg(keepNeurons, :), 1));
        PlotVerticalLines([odorTimes(1), odorTimes(2), rewardTimes(1)])
        xlim(XL)
        xlabel('time [s]')
        ylabel('lick rate')
        export_fig(fullfile(plotDir, ['avg_traces_', 'trialtype_', num2str(trialType), '_roi_', num2str(roi), '_onlyModulated_', num2str(doOnlyShowModulated), '.png']))
        
        if doOrderTraces
            figure, plotStackedTraces(traceKeep(flipud(orderedInds), :), 'k', median(diff(T)), 2, [])
        else
            figure, plotStackedTraces(traceKeep, 'k', median(diff(T)), 2, [])
        end
            
        PlotVerticalLines([odorTimes(1), odorTimes(2), rewardTimes(1)])
        xlim(XL)
        xlabel('time [s]')
        export_fig(fullfile(plotDir, ['avg_traces_stacked_', 'trialtype_', num2str(trialType), '_roi_', num2str(roi), '_onlyModulated_', num2str(doOnlyShowModulated), '.pdf']))

    end
end


%% Rearrange neurons so that can step through with videoslider
minTrials = 12;
reshapedNeurons = zeros(minTrials, numel(T), numel(unique(whichNeuronID)));

trialType = 3;
for neuron = 1:numel(unique(whichNeuronID));
      whichRows = whichNeuronID == neuron & allTrialTypes == trialType;
      whichRows = find(whichRows);
      whichRows = whichRows(1:minTrials);
      reshapedNeurons(:,:, neuron) = allTraceData(whichRows, :);
end
VideoSlider(reshapedNeurons)

           
%% Plot invidual neurons across trials
% neurons = [165, 184, 180, 101, 112, 46, 205]
% neurons = [138]
neurons = [127, 133, 180, 184, 186, 205, 206, 278, 300]

for neuron = neurons
    for trialType = [3]

        whichRows = whichNeuronID == neuron & allTrialTypes == trialType;
        roi = mean(allROIs(whichTraceID(find(whichRows))));
        dataset = mean(datasetID(find(whichRows)));
        figure, 
        subplot(121)
        imagesc(T, 1:numel(find(whichRows)),allTraceData(whichRows, :));
        title(['neuron # ', num2str(neuron), ' roi ', num2str(roi), ' dataset ', num2str(dataset)])
        xlabel('time [s]')
        PlotVerticalLines([odorTimes(1), odorTimes(2), rewardTimes(1)], [], [], 'w--')
        xlim([3, 9])
        subplot(122)
        imagesc(T, 1:numel(find(whichRows)), allLickRates(whichRows,:));
        PlotVerticalLines([odorTimes(1), odorTimes(2), rewardTimes(1)], [], [], 'w--')
        xlim([3, 9])
        export_fig(fullfile(plotDir, ['all_trials_', 'neuron_', num2str(neuron), 'trialtype_', num2str(trialType), '_roi_', num2str(roi), '.png']))

    end
end


end


