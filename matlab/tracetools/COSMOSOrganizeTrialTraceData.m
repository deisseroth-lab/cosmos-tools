function [ allTraceData, allSpikeData, allTrialTypes, datasetID, whichTraceID, ...
           allLicksCell, allLickRates, whichTrial, whichNeuronID, ...
           allT, allOdorTimes, allRewardTimes, allPunishTimes, allBaselineTimes ] = COSMOSOrganizeTrialTraceData(allBpod, allTraces, allSpikes, allLEDpeakframes, allWhichDataset);


%%%% This function reads in the bpod behavioral data, splits up and aligns the neural
%%%% traces into a trial structure, and returns big matrices with all
%%%% trials for all mice stacked, and associated matrices labeling each trial.  

allTraceDataCell = {};
allSpikeDataCell = {};
allLicksCell = {};

allTraceData = []; %%%% Trace for each neuron and trial
allSpikeData = [];
allLickRates = []; %%%% Lick rate for each neuron and trial (this is the same for a given trial for each neuron in a dataset)


allTrialTypes = []; %%%% For each neuron and trial, which trial type that trial was.
datasetID = []; %%%% ID into the 'data' array that is initially used to load datasets
whichTraceID = []; %%%% Corresponds to traceIter (i.e. can be used to find which ROI a cell is from)
whichNeuronID = []; %%%% Assigns a unique ID to each neuron (in each ROI and each dataset)
whichTrial = []; %%%% Which number trial a given trace is in the session
allOdorTimes = []; %%%% For each neuron and trial, the time of odor onset
allRewardTimes = []; %%%% For each neuron and trial, the time of reward onset
allPunishTimes = []; %%%% For each neuron and trial, the time of punish
allBaselineTimes = [];
allT = {}; %%%% Time corresponding to each frame in trace (as computed based on the LED peak frames). 


rowIter = 1;
neuronIter = 1;
for k=1:numel(allTraces)
    whichDataset = allWhichDataset(k);
    traces = allTraces{k};
    spikes = allSpikes{k};
    
    %%%% Convert traces for each cell to trial structured matrix and fill
    %%%% in whichTraceID (with the iteration number).
    LEDpeakframes = allLEDpeakframes{whichDataset};
    
    %%%% Get lick times, odor times, reward times, punish times. Reference
    %%%% to the LED onset time. (double check that the time between LEDs
    %%%% matches the traces).
    bpod = allBpod{whichDataset};
    behavior = bpod.SessionData;
    
    ledTimes = GetStateTimes(behavior, 'TrialStartAlert', false);
    odorTimes = GetStateTimes(behavior, 'Stimulus', false);
    rewardTimes = GetStateTimes(behavior, 'Reward', false);
    punishTimes = GetStateTimes(behavior, 'Punish', false);
    lickTimes = GetEventTimes(behavior, 'Port1In', false);
    baselineTimes = GetStateTimes(behavior, 'Baseline', false);
    trialStartTimes = behavior.TrialStartTimestamp;

    
    
    for neuron = 1:size(traces, 1)
        for trial = 1:numel(LEDpeakframes)-1
            allTraceDataCell{rowIter} = traces(neuron, LEDpeakframes(trial):LEDpeakframes(trial+1));
            allSpikeDataCell{rowIter} = spikes(neuron, LEDpeakframes(trial):LEDpeakframes(trial+1));
            allT{rowIter} = (trialStartTimes(trial+1) - trialStartTimes(trial))*linspace(0, 1, LEDpeakframes(trial+1)-LEDpeakframes(trial)+1);
            allTrialTypes(rowIter) = behavior.TrialTypes(trial);
            datasetID(rowIter) = whichDataset;
            whichTraceID(rowIter) = k;
            whichNeuronID(rowIter) = neuronIter;
            whichTrial(rowIter) = trial;
            
            allLicksCell{rowIter} = lickTimes{trial};
            allOdorTimes(rowIter,:) = odorTimes(trial,:) - ledTimes(trial,1);
            allRewardTimes(rowIter,:) = rewardTimes(trial,:) - ledTimes(trial, 1);
            allPunishTimes(rowIter,:) = punishTimes(trial,:) - ledTimes(trial, 1);
            allBaselineTimes(rowIter, :) = baselineTimes(trial, :) - ledTimes(trial, 1);
            
            rowIter = rowIter + 1;
        end
        neuronIter = neuronIter + 1;
    end
        
end

%%%% Now collapse some of the cell arrays into fixed size matrices
minFrame = 1e10;
for row = 1:numel(allTraceDataCell)
    nFrame = size(allTraceDataCell{row}, 2);
    if nFrame < minFrame
        minFrame = nFrame;
    end
end


allTraceData = zeros(numel(allTraceDataCell), minFrame);
allSpikeData = zeros(numel(allTraceDataCell), minFrame);
for row = 1:numel(allTraceDataCell)   
    allTraceData(row, :) = allTraceDataCell{row}(1:minFrame); 
    allSpikeData(row, :) = allSpikeDataCell{row}(1:minFrame); 
end


alldt = zeros(numel(allTraceDataCell), 1);
for row = 1:numel(allTraceDataCell)
   alldt(row) = mean(diff(allT{row}));
end
dt = median(alldt);
allLickRates = EventRate(allLicksCell, dt, minFrame*dt);
allLickRates = allLickRates';

figure, plot(allT{1}(1:size(allLickRates, 2)), allLickRates(1,:)); title('Test trace plot')
figure, plot(allT{1}(1:size(allTraceData, 2)), allTraceData(1,:));  title('Test lick plot')
figure, plot(allT{1}(1:size(allSpikeData, 2)), allSpikeData(1,:));  title('Test spike plot')

