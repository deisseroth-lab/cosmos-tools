%%% This script starts to analyze (at least, plot) the traces extracted
%%% using cnmf-e. This just plots the traces, and does not include any
%%% other info such as trial starts or ROI or behavior. For that, use
%%% trace_analysis.m
%%% 20171207: This is currently setup for you to move the trace file
%%% (either the completed trace file, or if that did not work, then
%%% 'intermediate_results.mat') to a corresponding folder in
%%% Dropbox/cosmos_data.
%%%
%%% ikauvar@gmail.com


addpath(genpath('/home/izkula/src/COSMOS'))
addpath(genpath('~/src/COSMOS/matlab'))
addpath('~/Software/CNMF_E-master')
cnmfe_setup
close all; clear;

set(0,'defaultfigurecolor',[1 1 1])
set(0,'defaultAxesFontSize',20)
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 10);

traceDir = '/home/izkula/Dropbox/cosmos_data/'
% saveDir = '/home/izkula/Dropbox/cosmos/trace_analysis/20171230_photons'
saveDir = '~/Dropbox/cosmos/fig_plots/figS2/20171230_photons'
plotDir = fullfile(saveDir, 'plots')
if ~exist(saveDir, 'dir'); mkdir(saveDir); end
if ~exist(plotDir, 'dir'); mkdir(plotDir); end

doLoadBpod = false
doTracePlots = false %%% Summary plots of all traces (ignores behavior)
doTraceWithBehaviorPlots = false
doPlotLocations = true; %%% Plot spatial footprints of all neurons


doLoadCulled = true;
doCullNeuronsAutomatic = false;
doCullNeuronsManual = false; %%% You can do this after you have done the automatic culling. 

dt = 1/30.98; %%% Hz


%%%%% {'date', 'neural_vid_name', 'bpod_filename', [rois], 'label', scaleFactor, groupID}
% data = {...
%    {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session2_1', 'cuxai148m1_COSMOSTrainGNG_20171115_144532', [1, 2, 3], 'f4'};
%     {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session3_1', 'cuxai148m1_COSMOSTrainGNG_20171115_150409', [1,2, 3], 'f1.2'};
% %      {'20171115', 'cuxai148m1_COSMOSTrainGNG_Nov15_2017_Session5_1', 'cuxai148m1_COSMOSTrainGNG_20171115_154259', [2], 'f8'}; %%% roi1 and roi3 yielded no neurons with current settings
% }

%%%% Mid-december test of effect of f-stop on neuron number in a patch over rsp. 
%%%% Made a very good plot. 
data = {...
%     {'20171207', 'm369_f1.2_1', '', [1], 'f1.4'};
%    {'20171207', 'm369_f1.4_defocus500um_1', '', [1], 'f1.4-500um'};
%    {'20171207', 'm369_f2_1', '', [1], 'f2'}; 
%    {'20171207', 'm369_f2_defocus500um_1', '', [1], 'f2-500um'}; 
%     {'20171207', 'm369_f2.8_1', '', [1], 'f2.8'}; 
%    {'20171207', 'm369_f2.8_defocus500um_1', '', [1], 'f2.8-500um'};
   {'20171207', 'm369_f4_1', '', [1], 'f4'}; 
%    {'20171207', 'm369_f4_defocus500um_1', '', [1], 'f4-500um'};
%    {'20171207', 'm369_f5.6_1', '', [1], 'f5.6'}; 
%    {'20171207', 'm369_f5.6_defocus500um_1', '', [1], 'f5.6-500um'};
%    {'20171207', 'm369_f8_1', '', [1], 'f8'}; 
%    {'20171207', 'm369_f8_defocus500um_1', '', [1], 'f8-500um'}; 
}


% data = {...
%     {'20171228', 'f1.4_cux2m1_1', '', [1], 'f1.4-lat'};
%     {'20171228', 'f1.4_cux2m1_1', '', [2], 'f1.4-med'};
%     {'20171228', 'f2_cux2m1_1', '', [1], 'f2-lat'};
%     {'20171228', 'f2_cux2m1_1', '', [2], 'f2-med'};
%     {'20171228', 'f2.8_cux2m1_1', '', [1], 'f2.8-lat'};
%     {'20171228', 'f2.8_cux2m1_1', '', [2], 'f2.8-med'};
%     {'20171228', 'f4_cux2m1_1', '', [1], 'f4-lat'};
%     {'20171228', 'f4_cux2m1_1', '', [2], 'f4-med'};
%     {'20171228', 'f5.6_cux2m1_1', '', [1], 'f5.6-lat'};
%     {'20171228', 'f5.6_cux2m1_1', '', [2], 'f5.6-med'};
%     {'20171228', 'f8_cux2m1_1', '', [1], 'f8-lat'};
%     {'20171228', 'f8_cux2m1_1', '', [2], 'f8-med'};
%     {'20171228', 'v10_cux2m1_1', '', [1], 'v10-lat'};
%     {'20171228', 'v10_cux2m1_1', '', [2], 'v10-med'};
%     {'20171228', 'v11_cux2m1_1', '', [1], 'v11-lat'};
%     {'20171228', 'v11_cux2m1_1', '', [2], 'v11-med'};
% }

data = {...   


    {'20171230', 'f1.4_cux2m1_1_1', '', [1], 'f1.4-1-lat', 1, 1};
        {'20171230', 'f1.4_cux2m1_1_1', '', [2], 'f1.4-1-med', 1, 1};
    {'20171230', 'f2_cux2m1_1_1', '', [1],'f2-1-lat', 1, 2};
        {'20171230', 'f2_cux2m1_1_1', '', [2],'f2-1-med', 1, 2};
    {'20171230', 'f2.8_cux2m1_1_1', '', [1],'f2.8-1-lat', 1, 3};
        {'20171230', 'f2.8_cux2m1_1_1', '', [2],'f2.8-1-med', 1, 3};
    {'20171230', 'f4_cux2m1_1_1','', [1], 'f4-1-lat', 1, 4};
        {'20171230', 'f4_cux2m1_1_1','', [2], 'f4-1-med', 1, 4};
    {'20171230', 'f5.6_cux2m1_1_1', '', [1], 'f5.6-1-lat', 1, 5};
        {'20171230', 'f5.6_cux2m1_1_1', '', [2], 'f5.6-1-med', 1, 5};
    {'20171230', 'f8_cux2m1_1_1', '', [1],'f8-1-lat', 1, 6};
        {'20171230', 'f8_cux2m1_1_1', '', [2],'f8-1-med', 1, 6};
    {'20171230', 'v10_cux2m1_1_1','', [1], 'v10-1-lat', 40/50, 7};
        {'20171230', 'v10_cux2m1_1_1','', [2], 'v10-1-med', 40/50, 7};
    
    
   {'20171230', 'f1.4_cux2m1_2_1','', [1], 'f1.4-2-lat', 1, 1};
       {'20171230', 'f1.4_cux2m1_2_1','', [2], 'f1.4-2-med', 1, 1};
   {'20171230', 'f2_cux2m1_2_1', '', [1],'f2-2-lat',  1, 2};
       {'20171230', 'f2_cux2m1_2_1', '', [2],'f2-2-med',  1, 2};
   {'20171230', 'f2.8_cux2m1_2_1', '', [1],'f2.8-2-lat',  1, 3};
       {'20171230', 'f2.8_cux2m1_2_1', '', [2],'f2.8-2-med',  1, 3};
   {'20171230', 'f4_cux2m1_2_1', '', [1], 'f4-2-lat',  1, 4};
      {'20171230', 'f4_cux2m1_2_1', '', [2], 'f4-2-med',  1, 4};
    {'20171230', 'f5.6_cux2m1_2_1', '', [1],'f5.6-2-lat', 1, 5};
        {'20171230', 'f5.6_cux2m1_2_1', '', [2],'f5.6-2-med', 1, 5};
   {'20171230', 'f8_cux2m1_2_1', '', [1],'f8-2-lat', 1, 6};
       {'20171230', 'f8_cux2m1_2_1', '', [2],'f8-2-med', 1, 6};
   {'20171230', 'v10_cux2m1_2_1', '', [1],'v10-2-lat', 40/50, 7};
        {'20171230', 'v10_cux2m1_2_1', '', [2],'v10-2-med', 40/50, 7};

    {'20171230', 'f1.4_cux2m1_3_1', '', [1],'f1.4-3-lat',  1, 1};
        {'20171230', 'f1.4_cux2m1_3_1', '', [2],'f1.4-3-med',  1, 1};
    {'20171230', 'f2_cux2m1_3_1', '', [1],'f2-3-lat',  1, 2};
        {'20171230', 'f2_cux2m1_3_1', '', [2],'f2-3-med',  1, 2};
    {'20171230', 'f2.8_cux2m1_3_1', '', [1],'f2.8-3-lat',  1, 3};
        {'20171230', 'f2.8_cux2m1_3_1', '', [2],'f2.8-3-med',  1, 3};
    {'20171230', 'f4_cux2m1_3_1', '', [1],'f4-3-lat', 1, 4};
        {'20171230', 'f4_cux2m1_3_1', '', [2],'f4-3-med', 1, 4};
    {'20171230', 'f5.6_cux2m1_3_1','', [1], 'f5.6-3-lat', 1, 5};
        {'20171230', 'f5.6_cux2m1_3_1','', [2], 'f5.6-3-med', 1, 5};
    {'20171230', 'f8_cux2m1_3_1', '', [1],'f8-3-lat', 1, 6};
        {'20171230', 'f8_cux2m1_3_1', '', [2],'f8-3-med', 1, 6};
    {'20171230', 'v10_cux2m1_3_1', '', [1], 'v10-3-lat',  40/50, 7};
        {'20171230', 'v10_cux2m1_3_1', '', [2], 'v10-3-med',  40/50, 7};

}

% data = {...
%    {'20180102', 'm52_1wk_post_tamox_2', '', [1], 'm52-lat', 1, 1};
%    {'20180102', 'm52_1wk_post_tamox_2', '', [2], 'm52', 1, 1};
% }
% 
% data = {...
%    {'20180108', 'm52_2wk_post_tamox_real_1', '', [1], 'm52-med', 40/50, 1};
%    {'20180108', 'm52_2wk_post_tamox_real_1', '', [2], 'm52-lat', 40/50, 1};
% }

allTraces = {};
allRawTraces = {};
allSpikes = {};
allLabels = {};
allROIs = [];
allWhichDataset = [];
datasetNames = {};
allNeuronTraceStructs = {};

hasBeenCulled = zeros(numel(data));
allFnames = {};

traceIter = 1;
for kk = 1:numel(data)
    
    d = data{kk}
    datasetNames{kk} = d{5};
    for patch = d{4}
        
        %%% TO DO: Replace all of this with the actual saved out workspace
        
        tracePath = fullfile(traceDir, d{1}, d{2}, ['patch-', num2str(patch)]);
        
        if doLoadCulled
            fnames = dir(fullfile(tracePath, '*culled.mat'));
            if isempty(fnames)
                fnames = dir(fullfile(tracePath, '*.mat'));
                hasBeenCulled(kk) = false;
            else
                hasBeenCulled(kk) = true;
            end
        else
            fnames = dir(fullfile(tracePath, '*.mat'));
            hasBeenCulled(kk) = false;
        end
        
        fname = fnames(1).name;
        if contains(fname, 'intermediate_results')
%             neuron = LoadFromIntermediateResults(fullfile(tracePath, fname));
            ff = load(fullfile(tracePath, fname));
            if ~doLoadCulled
                neuron = ff.initialization.neuron;
            else
                neuron = ff.neuron;
            end
        else
            ff = load(fullfile(tracePath, fname));
            neuron = ff.neuron;
        end
        allFnames{kk} = fullfile(tracePath, fname);
        
        allTraces{traceIter} = neuron.C;
        allRawTraces{traceIter} = neuron.C_raw;
        allSpikes{traceIter} = neuron.S;
        allLabels{traceIter} = [d{5}];
        allROIs(traceIter) = patch;
        allWhichDataset(traceIter) = kk;
        allName{traceIter} = [d{5}, '-roi-', num2str(patch)];
        allNeuronStructs{traceIter} = neuron;
        traceIter = traceIter + 1;
    end
end

%% Make trace summary plots (ignoring behavior...)
if doTracePlots
    %%%% Plot top 10 traces
    for kk = 1:numel(allTraces)
        figure('Position', [1000, 1000, 2000, 700]), 
    %     h = waterfall(allTraces{kk}); view(0, 87)
    %     set(gca, 'ztick', []); set(gca, 'zticklabel', [])
        hold on
        nToPlot = min(size(allTraces{kk}, 1), 10);
        plotStackedTraces(allRawTraces{kk}(1:nToPlot,:), [.4,.4,.4], dt)
        plotStackedTraces(allTraces{kk}(1:nToPlot,:), 'r', dt)
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

end

if doCullNeuronsAutomatic
    for kk = 1:numel(data)
        if ~hasBeenCulled(kk)
            neuron = allNeuronStructs{kk};
            corr_thresh = 0.85;
            aspect_ratio_thresh = 1.7; %%%2
            [goodNeurons, badNeurons, corrs, aspect_ratios] = classifyNeurons(neuron, corr_thresh, aspect_ratio_thresh); 
%             neuron.viewNeurons(goodNeurons, neuron.C_raw);
%             neuron.viewNeurons(badNeurons, neuron.C_raw);
            
            neuron.delete(badNeurons);
            fname = allFnames{kk};            
            save([fname(1:end-4), '_culled.mat'], 'neuron');
        end
    end
end

%% Only select traces that look like neural traces??
if doCullNeuronsManual
    for kk = 1:numel(data)
        if ~hasBeenCulled(kk)
            neuron = allNeuronStructs{kk};
            try
                neuron.viewNeurons([], neuron.C_raw);
            catch
                disp('Could not access viewNeurons field')
            end
            fname = allFnames{kk};

            %%%% Save out culled neural traces
            save([fname(1:end-4), '_culled.mat'], 'neuron');
        end
    end
end

%% Plot the location of the neurons
if doPlotLocations
    for kk = 1:numel(allNeuronStructs)
        d = data{kk};
        neuron = allNeuronStructs{kk};
        try
            Coor = neuron.show_contours(0.6, [], log10(neuron.PNR), false);
        catch
            disp('Could not show contours')
        end
        title([strrep(d{2}, '_', '-'), ' #=', num2str(size(neuron.A, 2))], 'FontSize', 15)
        export_fig(fullfile(saveDir, ['recovered_neurons_', d{5}, '.pdf']))
    end
end
    

%% Extract the background vs. signal for each of the f-numbers and compare them....

doCompareBackground = true;
allMedBg = [];
allMedSignal = [];
allMedPixelsPerNeuron = [];
allMedSignalPerPixel = [];
allMedCentralSignal = [];
allMedTotalSignal = [];

allStdBg = [];
allStdSignal = [];
allStdPixelsPerNeuron = [];
allStdSignalPerPixel = [];
allStdTotalSignal = [];
allNumSamples = [];
allNumNeurons = [];
allStdCentralSignal = [];

fsize = 15;
labels = {};

group = zeros(numel(allNeuronStructs), 1);

doBg = true
if doCompareBackground
    for kk = 1:numel(allNeuronStructs)
        d = data{kk};
        neuron = allNeuronStructs{kk};

        scaleFactor = double(d{6});
        
        if doBg
    %         Ybg = neuron.reconstruct_background(neuron.frame_range);
            nframes = 500
            try
                Ybg = neuron.reconstruct_background([1, nframes]);
            catch
                print('Could not reconstruct background.')
                Ybg = zeros(size(neuron.Cn,1), size(neuron.Cn, 2), nframes);
            end
                
        end
        
        A = neuron.A; %%% #pixels x #neurons
        C_raw = neuron.C_raw; %%%% #neurons x #timepoints
        numNeurons = size(A, 2);
        
        max_signal = quantile(C_raw, 0.999, 2); 
        med_signal = quantile(C_raw, 0.5, 2);
        median(med_signal)
        
        total_neuron_signals = sum(A, 1).*max_signal';
        
        pixels_per_neuron = zeros(numNeurons, 1); %%% # of pixels in the footprint that are responsible for 90% of the signal. 
        total_sig_per_neuron = zeros(numNeurons, 1);
        total_central_sig_per_neuron = zeros(numNeurons, 1);
        sorted_sig_per_neuron = zeros(size(A'));
        signal_fraction = 0.9;
        figure
        for ii = 1:numNeurons
            footprint = A(:, ii); 
            footprint = sort(footprint, 'descend');
            sorted_sig_per_neuron(ii,:) = footprint;
            total_sig_per_neuron(ii) = sum(footprint); 
            total_central_sig_per_neuron(ii) = sum(footprint(1:10));
            s_footprint = sort(footprint, 'descend');
            pixels_per_neuron(ii) = max(find(cumsum(s_footprint) < signal_fraction*total_sig_per_neuron(ii)));
            pixels_per_neuron(ii) = 1.0/scaleFactor*pixels_per_neuron(ii);
        end
        plot(median(sorted_sig_per_neuron(:, 1:100), 1))
        ylim([0, 50])
        title(d{5})

        neuron_signals_per_pixel = total_neuron_signals./pixels_per_neuron';
        
        if doBg
        medYbg = median(Ybg, 3);
        medYbg = sort(medYbg(:), 'descend');
        medYbg = medYbg(0.1*numel(medYbg):0.9*numel(medYbg));
        medYbg = scaleFactor*medYbg;
        allMedBg(kk) = median(medYbg(:));
        allStdBg(kk) = std(medYbg(:));
        allNumSamples(kk) = numel(medYbg);
        end
        
        allMedSignal(kk) = median(total_sig_per_neuron);
        allStdSignal(kk) = std(total_sig_per_neuron);
        
        allMedCentralSignal(kk) = median(total_central_sig_per_neuron);
        allStdCentralSignal(kk) = std(total_central_sig_per_neuron);
        
        allMedPixelsPerNeuron(kk) = median(pixels_per_neuron);
        allStdPixelsPerNeuron(kk) = std(pixels_per_neuron);
        
        allMedSignalPerPixel(kk) = median(neuron_signals_per_pixel);
        allStdSignalPerPixel(kk) = std(neuron_signals_per_pixel);
        
        allMedTotalSignal(kk) = median(total_neuron_signals);
        allStdTotalSignal(kk) = std(total_neuron_signals);
        
        allNumNeurons(kk) = numNeurons;
        
        
        labels{kk} = d{5}
        group(kk) = d{7};
    end
end

%% Plot results

depth = zeros(numel(allMedBg),1);
for kk = 1:numel(depth)
    d = data{kk};
    if contains(d{5}, '500um') 
        depth(kk) = 1;
    end
    if contains(d{5}, 'med') 
        depth(kk) = 1;
    end
end


doPlotGroupedBars = true
if doPlotGroupedBars
    figure('Position', [100, 100, 800, 400])
    groups = unique(group)
    groupLabels = {}
    groupVals = [];
    for i = 1:numel(groups)
        g = groups(i)
        cc = {'c', 'g'}
        for d = 0:1
            vals = allNumNeurons(group == g & depth == d)
            bar(g+(d-.5)*0.4, mean(vals), 0.3, cc{d+1});
            hold on
            plot(g+(d-0.5)*0.4, vals, 'k.', 'Markersize', 5);
        end
        
        %%% Just extract the shortened group name for plot labels
        ind = find(group == g)
        labelFull = labels{ind(1)};
        labelShort = strsplit(labelFull, '-');
        groupLabels{i} = labelShort{1};
    end
    xticks([1:numel(groups)])
    set(gca,'xticklabel',groupLabels, 'FontSize', 12)
    title('Number of recovered neurons', 'FontSize', 12)
    export_fig(fullfile(saveDir, ['grouped_num_neurons.pdf']))

    
    figure('Position', [100, 100, 400, 300])
    groups = unique(group)
    groupLabels = {}
    groupVals = [];
    for i = 1:numel(groups)
        g = groups(i)
        vals = [allNumNeurons(group == g & depth == 1);
                allNumNeurons(group == g & depth == 0)]
        groupVals(:, i) = mean(vals, 2);
            
        %%% Just extract the shortened group name for plot labels
        ind = find(group == g)
        labelFull = labels{ind(1)};
        labelShort = strsplit(labelFull, '-');
        groupLabels{i} = labelShort{1};
    end
    bar(groupVals', 'stacked')
    
    for i = 1:numel(groups)
        g = groups(i)
        vals = [allNumNeurons(group == g & depth == 1);
                allNumNeurons(group == g & depth == 0)]     
        hold on
        plot(g, sum(vals, 1), 'k.', 'Markersize', 5);
    end
    xticks([1:numel(groups)])
    set(gca,'xticklabel',groupLabels, 'FontSize', 12)
    title('Total number of recovered neurons', 'FontSize', 12)
    legend({'lateral', 'medial'}, 'FontSize', 12, 'Location', 'NorthWest')
%     export_fig(fullfile(saveDir, ['grouped_num_neurons_stacked.pdf']))
    print('-depsc',fullfile(saveDir, ['grouped_num_neurons_stacked.eps']))

    
    
    %%%% 
    QE = 0.93 %% Quantum efficiency of photometrics Prime95B. Percentage of electrons produced from the number of incident photons
    electron_photon_conversion = 1/QE
    pixel_value_to_photons = electron_photon_conversion  %% Assumption (seems to be the case) is that it is one electron per pixel value. This is different than Hamamatsu orca, which is 2 electrons per value. 
    
    
    figure('Position', [100, 100, 800, 400])
    groups = unique(group)
    groupLabels = {}
    groupVals = [];
    for i = 1:numel(groups)
        g = groups(i)
        cc = {'c', 'g'}
        for d = 1
            vals = allMedSignalPerPixel(group == g & depth == d)
            vals = pixel_value_to_photons*vals;
            bar(g, mean(vals), 0.7, cc{d+1});
            hold on
            plot(g, vals, 'k.', 'Markersize', 5);
        end
        
        %%% Just extract the shortened group name for plot labels
        ind = find(group == g)
        labelFull = labels{ind(1)};
        labelShort = strsplit(labelFull, '-');
        groupLabels{i} = labelShort{1};
    end
    xticks([1:numel(groups)])
    set(gca,'xticklabel',groupLabels, 'FontSize', 12)
    ylabel('# photons')
    title('Median signal photons per pixel', 'FontSize', 12)
    print('-depsc',fullfile(saveDir, ['grouped_med_signal_per_pixel.eps']))

    
    
    figure('Position', [100, 100, 800, 400])
    groups = unique(group)
    groupLabels = {}
    groupVals = [];
    for i = 1:numel(groups)
        g = groups(i)
        cc = {'c', 'g'}
        for d = 1
            vals = allMedBg(group == g & depth == d)
            vals = pixel_value_to_photons*vals;
            bar(g, mean(vals), 0.7, cc{d+1});
            hold on
            plot(g, vals, 'k.', 'Markersize', 5);
        end
        
        %%% Just extract the shortened group name for plot labels
        ind = find(group == g)
        labelFull = labels{ind(1)};
        labelShort = strsplit(labelFull, '-');
        groupLabels{i} = labelShort{1};
    end
    xticks([1:numel(groups)])
    set(gca,'xticklabel',groupLabels, 'FontSize', 12)
    ylabel('# photons')
    title('Median background photons per pixel', 'FontSize', 12)
    yticks([0:2000:12000])
    ylim([0, 12000])
    set(gcf, 'Units', 'Inches', 'Position', [0, 0,  3, 1.5], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    print('-depsc',fullfile(saveDir, ['grouped_med_bg_per_pixel.eps']))
    
    
    figure('Position', [100, 100, 800, 400])
    groups = unique(group)
    groupLabels = {}
    groupVals = [];
    for i = 1:numel(groups)
        g = groups(i)
        cc = {'c', 'g'}
        for d = 1
            vals = allMedTotalSignal(group == g & depth == d)
            vals = pixel_value_to_photons*vals;
            bar(g, mean(vals), 0.7, cc{d+1});
            hold on
            plot(g, vals, 'k.', 'Markersize', 5);
        end
        
        %%% Just extract the shortened group name for plot labels
        ind = find(group == g)
        labelFull = labels{ind(1)};
        labelShort = strsplit(labelFull, '-');
        groupLabels{i} = labelShort{1};
    end
    xticks([1:numel(groups)])
    set(gca,'xticklabel',groupLabels, 'FontSize', 12)
    ylabel('# photons')
    yticks([0:2e4:16e4])
    ylim([0,16e4])
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 3, 1.5], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    title('Median total signal photons per neuron', 'FontSize', 12)
    print('-depsc',fullfile(saveDir, ['grouped_total_signal.eps']))

end


doPlotIndivBars = false
if doPlotIndivBars
    figure('Position', [100, 100, 2000, 1000])
    % errorbar([1:numel(allMedSignal)], allMedSignal, allStdSignal)
    b = barwitherr(allStdBg./sqrt(allNumSamples), allMedBg)
    b.FaceColor = 'flat'
    xticks([1:numel(allNumSamples)])
    set(gca,'xticklabel',labels)
    b.CData(find(depth), :) = repmat([0, 1, 0], numel(find(depth)), 1);
    title('Median background per pixel', 'FontSize', fsize)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
     export_fig(fullfile(saveDir, ['med_bg_per_pixel.pdf']))

    figure('Position', [100, 100, 2000, 1000])
    % errorbar([1:numel(allMedSignal)], allMedSignal, allStdSignal)
    b = barwitherr(allStdPixelsPerNeuron./sqrt(allNumNeurons), allMedPixelsPerNeuron)
    b.FaceColor = 'flat'
    xticks([1:numel(allNumSamples)])
    b.CData(find(depth), :) = repmat([0, 1, 0], numel(find(depth)), 1);
    set(gca,'xticklabel',labels)
    title('Median number of pixels per neuron', 'FontSize', fsize)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['med_num_pixel_per_neuron.pdf']))

    figure('Position', [100, 100, 2000, 1000])
    % errorbar([1:numel(allMedSignal)], allMedSignal, allStdSignal)
    b = bar(allNumNeurons)
    b.FaceColor = 'flat'
    xticks([1:numel(allNumSamples)])
    b.CData(find(depth), :) = repmat([0, 1, 0], numel(find(depth)), 1);
    set(gca,'xticklabel',labels)
    title('Number of recovered neurons', 'FontSize', fsize)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['num_recovered_neuron.pdf']))


    figure('Position', [100, 100, 2000, 1000])
    % errorbar([1:numel(allMedSignal)], allMedSignal, allStdSignal)
    % barwitherr(allStdSignalPerPixel./sqrt(allNumNeurons), allMedSignalPerPixel)
    b=bar(allMedSignalPerPixel)
    xticks([1:numel(allNumSamples)])
    b.FaceColor = 'flat'
    b.CData(find(depth), :) = repmat([0, 1, 0], numel(find(depth)), 1);
    set(gca,'xticklabel',labels)
    title('Median signal per neuron pixel', 'FontSize', fsize)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['med_signal_per_pixel.pdf']))
end
%% Actually compute the SNR on the best neurons and compare

doComputeSNR = true




if doComputeSNR
    topN = 10;
    fsize = 15;
    topPNR = nan*ones(numel(allNeuronStructs), topN);
    labels = {};
    for kk = 1:numel(allNeuronStructs)
        d = data{kk};
        neuron = allNeuronStructs{kk};


        A = neuron.A; %%% #pixels x #neurons
        C_raw = neuron.C_raw; %%%% #neurons x #timepoints
        C = neuron.C;
        numNeurons = size(A, 2);

        labels{kk} = d{5}

        sn = [];
        peak = [];
        for nn = 1:numNeurons
            trace = C_raw(nn, :);
    %         if nn == 2 %%% Plot an example trace
    %             figure, plot(trace), title(labels{kk})
    %         end
            sn(nn) = get_noise_fft(trace);
            peak(nn) = quantile(trace, .999);
        end

        PNR = peak./sn;
        [sPNR, sinds] = sort(PNR, 'descend');


        numN = min(topN, size(sPNR, 2));
        topPNR(kk, 1:numN) = sPNR(1:numN);

        figure
        plotStackedTraces(C_raw(sinds(1:numN),:), [.4,.4,.4], dt)
        hold on
        plotStackedTraces(C(sinds(1:numN),:),  'r', dt)   
        title(labels{kk})    
        export_fig(fullfile(saveDir, 'plots', ['traces_', num2str(kk), '.pdf']))

    end

    figure, boxplot(topPNR')
    set(gca,'xticklabel',labels)
    title(['PNR of top ', num2str(topN), ' neurons'], 'FontSize', fsize)
    set(gca,'FontSize',fsize);
    set(gca, 'XTickLabelRotation', 45)
    export_fig(fullfile(saveDir, ['PNR_of_top_neurons.pdf']))
    
    
    
    margin = 0.1;
    nPerGroup = 3;
    groupedPNR = nan*ones(numel(allNeuronStructs)/nPerGroup, topN*nPerGroup);
    figure('Position', [100, 100, 400, 400])
    for i = 1:numel(groups)
            g = groups(i)
            
            rows = find(group == g & depth == 1); 
            pnr = topPNR(rows,:);
            groupedPNR(i*2, :) = pnr(:);
            plot(i*ones(size(pnr))+margin, pnr, 'g.', 'MarkerSize', 15)
            hold on
            plot(i + margin, nanmedian(pnr(:)), 'k+')
            
            rows = find(group == g & depth == 0); 
            pnr = topPNR(rows,:);
            groupedPNR(i*2-1, :) = pnr(:);
            plot(i*ones(size(pnr))- margin, pnr, 'c.', 'MarkerSize', 15)
            plot(i-margin, nanmedian(pnr(:)), 'k+')
            
            %%% Just extract the shortened group name for plot labels
            ind = find(group == g)
            labelFull = labels{ind(1)};
            labelShort = strsplit(labelFull, '-');
            groupLabels{i} = labelShort{1};
    end
    xticks([1:numel(groups)])
    set(gca,'xticklabel',groupLabels, 'FontSize', 12)
    title('PSNR of top neurons', 'FontSize', 12)
    export_fig(fullfile(saveDir, ['grouped_PNR_of_top_neurons.pdf']))

%      figure, boxplot(groupedPNR')
end

