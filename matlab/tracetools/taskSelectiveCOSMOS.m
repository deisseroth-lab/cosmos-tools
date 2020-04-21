function [ isSelective ] = taskSelectiveCOSMOS( spikes, dt, threshold,baseline, statType, taskInds)

%%% This function determines whether a neuron is significantly more active
%%% during the task as opposed to the baseline. (This may not be what you
%%% always want, so be careful). 
%%% Input is a matrix with the trials for a single neuron with a single
%%% trial type. That is trials x time. 

%%% Spikes - a neurons*trials x time matrix.
%%% dt - the time of one frame in seconds. 
%%% threshold - the p-value threshold for significance of task modulation
%%% statType - 'ttest' or 'bootstrap'
%%% whichTrials - if a subset of trials are to be queried
%%% baseline - the frame indices corresponding to the baseline of a trial
%%% (from the end of the baseline to the end of the trial is counted as
%%% part of the non-baseline of the trial). 
%%% taskInds - similar to baseline, but the indices in the task to compare
%%% with the baseline.

fps = round(1./dt);

if ~exist('statType', 'var') || isempty(statType)
    statType = 'bootstrap';
end

if ~exist('taskInds', 'var') || isempty(taskInds)
    %%%% Similar to 'baseline', but the indices in the task to compare with
    %%%% the baseline. 
    taskInds = [];
end



nT = size(spikes,2);
nPerms = 1000.;
pSpikes = zeros(1,size(spikes,2));


%find whole-task selective (vs pre-task)

if isempty(taskInds)
    taskInds = [baseline(2), size(spikes, 2)];
end

if strcmp(statType, 'ttest')
%     p = [];
%     for i=1:size(spikes,1)
%         if isempty(taskInds)
%             [~,p(end+1)] = ttest(mean(spikes(i,taskInds(1):taskInds(2),:),2), mean(spikes(i,baseline(1):baseline(2),:),2));
%         end
%     end
%     p = p < threshold;
%     p = p';
elseif strcmp(statType,'bootstrap')
    tmod = zeros(1,nPerms);
    parfor i=1:nPerms
        pSpikes = mean(spikes(:, randperm(nT)), 1);
        %tmod = tmod + (mean(pSpikes(:,baseline(1):baseline(2)),2) < mean(pSpikes(:,baseline(2):end),2));
        tmod(:,i) = mean(pSpikes(taskInds(1):taskInds(2))) - mean(pSpikes(baseline(1):baseline(2)));
    end
    %tmod = (tmod/nPerms) < threshold;
    p = zeros(size(tmod,1),1);
    for i=1:size(tmod,1)
        p(i) = sum(tmod(i,:) >= mean(spikes(taskInds(1):taskInds(2))) - mean(spikes(baseline(1):baseline(2))))./nPerms;
    end
    isSelective = p < threshold;
end

end
