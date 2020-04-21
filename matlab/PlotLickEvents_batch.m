%% Get ready
close all; clear all;

%% Load a dataset
%str = 'E:\Data\Behavior\cux2m73\COSMOSTrainMultiBlockGNG\Session Data\cux2m73_COSMOSTrainMultiBlockGNG_20180303_141118.mat';
%str = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180228_130348.mat';
%str = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180302_131322.mat';
%str = 'E:\Dropbox\Dropbox\behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180314_142833.mat';
%str = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180315_162530.mat';
% str = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180316_160237.mat';


dsstr{1} = 'E:\Data\Behavior\cux2m72\COSMOSTrainMultiGNG\Session Data\cux2m72_COSMOSTrainMultiGNG_20180219_160349.mat';
dsstr{2} = 'E:\Data\Behavior\cux2m72\COSMOSTrainMultiBlockGNG\Session Data\cux2m72_COSMOSTrainMultiBlockGNG_20180227_132125.mat'
dsstr{3} = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180302_131322.mat';
dsstr{4} = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180315_162530.mat';
dsstr{5} = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiBlockGNG\Session Data\vtapfcm36_COSMOSTrainMultiBlockGNG_20180316_160237.mat';
dsstr{6} = 'E:\Data\Behavior\vtapfcm36\COSMOSTrainMultiGNG\Session Data\vtapfcm36_COSMOSTrainMultiGNG_20180214_135919.mat';

dsname{1} = 'cux2m72_COSMOSTrainMultiGNG_20180219_160349';
dsname{2} = 'cux2m72_COSMOSTrainMultiBlockGNG_20180227_132125';
dsname{3} = 'vtapfcm36_COSMOSTrainMultiBlockGNG_20180302_131322';
dsname{4} = 'vtapfcm36_COSMOSTrainMultiBlockGNG_20180315_162530';
dsname{5} = 'vtapfcm36_COSMOSTrainMultiBlockGNG_20180316_160237';
dsname{6} = 'vtapfcm36_COSMOSTrainMultiGNG_20180214_135919';

spout_count{1} = false;
spout_count{2} = false;
spout_count{3} = false;
spout_count{4} = true;
spout_count{5} = true;
spout_count{6} = false;

for zz=4
    load(dsstr{zz})
    mouse=dsname{zz};

    %% Define constants and parameters
    for lag = [0 5 10]

            %lag = 10; % if selected trials, only keep CORRECT trials >= lag trials since last change?
            three_spouts = spout_count{zz}; % THREE or FOUR spouts
            if lag == 0
                select_trials = false;
            else
                select_trials = true;
            end
            nogo_flag = 4; % what is the bpod trial type for NOGO trials
            go_flag = 3;
            explore_flag = 2;
            stim_on = 0; % in seconds, when odor comes on
            stim_off = 1; % in seconds, when reward turns off
            reward_on = 1.5; % in seconds, earliest time reward could come on
            reward_off = 4; % a few seconds after reward would have been dispensed (this is somewhat arbitrary)

            % don't change these parameters
            cull_nogo = false;
            % min_move = -1;
            min_move = -1;
            Data = SessionData;

            %% Compute outcomes vector
            Outcomes = zeros(1,Data.nTrials);
            for x = 1:Data.nTrials
                if ~isnan(Data.RawEvents.Trial{x}.States.Reward(1))
                    Outcomes(x) = 1;
                elseif Data.TrialTypes(x) == 4 && isnan(Data.RawEvents.Trial{x}.States.Punish(1))
                    Outcomes(x) = 1;
                elseif Data.TrialTypes(x) == 1 && isnan(Data.RawEvents.Trial{x}.States.Punish(1))
                    Outcomes(x) = 1;
                else
                    Outcomes(x) = 0;

                end
            end

            %% Find how many trials it has been since the spout last moved
            changes = find(diff(Data.SpoutPositions) ~= 0);
            closest = zeros(length(changes),1);
            for trial = 1:Data.nTrials
                if isempty(changes(changes<trial))
                    continue
                end
                closest(trial) = min(abs(changes(changes<trial)-trial));
            end
            % closest is zero only for the first spout position (since there was no
            % previous spout position)

            %% Extract quantities for subsequent analyses and plot raw data
            figure('position', [0, 0, 550, 650]); clf;
            set(gcf,'Color','w');
            set(gcf,'Renderer','painters')
            k = 1;
            color = 'wrgy';
            rewards = zeros(Data.nTrials, 1);
            triggered_events = cell(4, 1);
            triggered_event_types = cell(4, 1);
            all_event_types = [];
            event_types_trials = cell(Data.nTrials, 1);
            event_times_trials = cell(Data.nTrials, 1);

            ct = 0;
            if select_trials
                %tr = 80:200;
                %tr = 10:80;
                %tr = tr(find(closest(tr) >= 5));
                %tr = find(closest >= lag)
                tr = intersect(find(Outcomes), find(closest >= lag));
            else
                tr = 1:Data.nTrials;
            end
            for j=1:length(tr)
                i = tr(j);
                if cull_nogo && Data.TrialTypes(i) == nogo_flag
                    continue
                end
                if closest(i) <= min_move
                    continue
                end
                ct=ct+1;
                startTime = Data.RawEvents.Trial{i}.States.Stimulus(1);
                rewardTime = Data.RawEvents.Trial{i}.States.Reward(1)-startTime;
                pos = Data.SpoutPositions(i);
                if isfield(Data.RawEvents.Trial{i}.Events, 'Port1In')
                    events = Data.RawEvents.Trial{i}.Events.Port1In - startTime;
                    plot(events, k, 'w.'); hold on;
                    triggered_events{pos}  = horzcat(triggered_events{pos}, events);
                    triggered_event_types{pos}  = horzcat(triggered_event_types{pos}, ones(1,length(events))*0);
                    all_event_types = [all_event_types ones(1,length(events))*0];
                    event_types_trials{i} = [event_types_trials{i} ones(1,length(events))*0];
                    event_times_trials{i} = [event_times_trials{i} events];
                end
                if isfield(Data.RawEvents.Trial{i}.Events, 'Port5In')
                    events = Data.RawEvents.Trial{i}.Events.Port5In - startTime;
                    plot(events, k, 'w.'); hold on;
                    triggered_events{pos}  = horzcat(triggered_events{pos}, events);
                    triggered_event_types{pos}  = horzcat(triggered_event_types{pos}, ones(1,length(events))*1);
                    all_event_types = [all_event_types ones(1,length(events))*1];
                    event_types_trials{i} = [event_types_trials{i} ones(1,length(events))*1];
                     event_times_trials{i} = [event_times_trials{i} events];
                end
                if isfield(Data.RawEvents.Trial{i}.Events, 'Port6In')
                    events = Data.RawEvents.Trial{i}.Events.Port6In - startTime;
                    plot(events, k, 'r.'); hold on;
                    triggered_events{pos}  = horzcat(triggered_events{pos}, events);
                    triggered_event_types{pos}  = horzcat(triggered_event_types{pos}, ones(1,length(events))*2);
                    all_event_types = [all_event_types ones(1,length(events))*2];
                    event_types_trials{i} = [event_types_trials{i} ones(1,length(events))*2];
                    event_times_trials{i} = [event_times_trials{i} events];
                end
                if isfield(Data.RawEvents.Trial{i}.Events, 'Port7In')
                    events = Data.RawEvents.Trial{i}.Events.Port7In - startTime;
                    plot(events, k, 'g.'); hold on;
                    triggered_events{pos}  = horzcat(triggered_events{pos}, events);
                    triggered_event_types{pos}  = horzcat(triggered_event_types{pos}, ones(1,length(events))*3);
                    all_event_types = [all_event_types ones(1,length(events))*3];
                    event_types_trials{i} = [event_types_trials{i} ones(1,length(events))*3];
                    event_times_trials{i} = [event_times_trials{i} events];
                end
                if isfield(Data.RawEvents.Trial{i}.Events, 'Port8In')
                    events = Data.RawEvents.Trial{i}.Events.Port8In - startTime;
                    plot(events, k, 'y.'); hold on;
                    triggered_events{pos}  = horzcat(triggered_events{pos}, events);
                    triggered_event_types{pos}  = horzcat(triggered_event_types{pos}, ones(1,length(events))*4);
                    all_event_types = [all_event_types ones(1,length(events))*4];
                    event_types_trials{i} = [event_types_trials{i} ones(1,length(events))*4];
                    event_times_trials{i} = [event_times_trials{i} events];
                end
                plot(0, k, [color(pos), '.']); hold on;
                if ~isnan(Data.RawEvents.Trial{i}.States.Reward(1))
                    plot(rewardTime, k,'g^'); hold on;
                    rewards(i) = rewardTime-startTime;
                elseif ~isnan(Data.RawEvents.Trial{i}.States.Punish(1))
                    plot(Data.RawEvents.Trial{i}.States.Punish(1)-startTime, k, 'r^'); hold on;
                    rewards(i) = inf;
                else
                    rewards(i) = inf;
                end    
                k = k + 1;
                ylim([-1  Data.nTrials+1]);
            end
            set(gca,'Color','k')
            ylim([0, ct]);
            xlim([-2, 6]);
            xlabel('Time (s)');
            ylabel('Trial');

            set(gcf, 'InvertHardcopy', 'off');
            saveas(gcf, [mouse '-Raster.png']);


            %% Get trial type counts
            type_counts = zeros(4,1);
            for x = 1:4
                ok = find(closest > min_move);
                if cull_nogo
                    go_trial = find(Data.TrialTypes ~= nogo_flag);
                    ok = intersect(ok, go_trial);
                end
                culled = Data.SpoutPositions(ok);
                type_counts(x) = length(find(culled == x));
            end
            disp(type_counts)
            disp(sum(type_counts))

            %% Plot lick histograms
            figure('position', [0, 0, 600, 400]); hold on;
            set(gcf,'Color','w');
            set(gcf,'Renderer','painters');
            norm_factor = 200;
            edges = -2.5:.25:6;

            subplot(3,2,[2 4 6]); hold on;
            plot([stim_on, stim_on], [-1, 5], 'w') % stimulus onset
            plot([stim_off, stim_off], [-1, 5], 'w') % stimulus offset
            plot([reward_on, reward_on], [-1, 5], 'w') % approximate reward time

            % Plot spout licked histograms as a function of active spout
            densities = cell(4,1);
            for spout_pos = 1:4
                densities{spout_pos} = zeros(length(edges)-1,4);
                for lick_pos = 1:4
                    kind = find(triggered_event_types{spout_pos} == lick_pos);
                    density = histcounts(triggered_events{spout_pos}(kind), edges);
                    densities{spout_pos}(:,lick_pos) = density;
                    ds = density/norm_factor+spout_pos;
                    if spout_pos == lick_pos
                        plot(edges(2:end), ds, color(lick_pos), 'linewidth', 3);
                    else
                        plot(edges(2:end), ds, color(lick_pos));
                    end
                end
                target_spout = num2str(type_counts(spout_pos));
                text(2.85,spout_pos+.8,['Trials = ',target_spout],'color','w')
            end

            % Make plot pretty
            set(gca,'Color','k');
            xlabel('Time (s)');
            if cull_nogo
                ss = 'go trials only';
            else
                ss = 'all trials';
            end
            ylabel(['Active Spout (1 = ', num2str(norm_factor), ' licks; ', ss, ')']);
            ylim([0.5, 5]);
            yticklabels({'1', '2', '3', '4'})
            yticks([1.35, 2.35, 3.35, 4.35])

            % Plot fraction of on target licks over time
            subplot(3,2,[1 3 5]); hold on;
            before = find(edges >= stim_off, 1, 'first'):find(edges < reward_on, 1, 'last');
            after = find(edges >= reward_on, 1, 'first'):find(edges < reward_off, 1, 'last');
            for pos = 1:4
                ds = densities{pos};
                bar(pos+.25, sum(ds(before,pos))/sum(sum(ds(before,:),2)), .25, 'r');
                bar(pos+.5, sum(ds(after,pos))/sum(sum(ds(after,:),2)), .25, 'b');
            end
            xticklabels({'1', '2', '3', '4'})
            xticks([1.35, 2.35, 3.35, 4.35])
            xlabel('Active spout');
            ylabel('Fraction of licks to active spout');
            legend({'Before reward','After reward'}, 'location','northwest');

            % Plot fraction of on target licks over time
            %subplot(3,2,6); hold on;
            %plot([0, 0], [0, 1], 'w') % stimulus onset
            %plot([1, 1], [0, 1], 'w') % stimulus offset
            %plot([1.5, 1.5], [0, 1], 'w') % approximate reward time

            %set(gca,'Color','k');
            %for pos = 1:4
            %    ds = densities{pos};
            %    plot(edges(2:end),ds(:,pos)./sum(ds,2), [color(pos), '.'], 'markersize',15);
            %end
            %xlabel('Time (s)');
            %ylabel('Fraction');

            set(gcf, 'InvertHardcopy', 'off');
            if select_trials    
               saveas(gcf, [mouse '-select_trials-lag' num2str(lag) '-MultiPlot.png']);
            else
               saveas(gcf, [mouse '-MultiPlot.png']);
            end

            %% Plot lick preference as a function of time since spout moved
            before_ = [true false]; % only use licks before, or licks after reward onset?
            for tr = 1:2
               before = before_(tr);
                if length(Data.SpoutPositions) == length(event_types_trials)
                    figure('position', [0, 0, 500, 400]); hold on;
                    set(gcf,'Color','w');
                    bins = cell(max(closest),1);

                    % at each lag relative to when the active spout changed look at the
                    % fraction of all licks that were towards the active spout position
                    for i=1:Data.nTrials
                        if  Data.TrialTypes(i) == nogo_flag
                            continue
                        end

                        % restrict analysis to events between stimulus onset and reward
                        if before
                            good = intersect(find(event_times_trials{i} < reward_on),...
                                             find(event_times_trials{i} >= stim_on));
                            tt = 'Licks before reward onset';
                            nn = 'before';
                        else
                            good = intersect(find(event_times_trials{i} >= reward_on),...
                                             find(event_times_trials{i} < reward_off));
                            tt = 'Licks after reward onset';
                            nn = 'after';
                        end

                        % in the closeset vector 1 means the spout just changed, 0 is
                        % reserved for the first few trials before the spout ever moved
                        if ~isempty(good) && closest(i) > 0
                            types = event_types_trials{i}(good);

                            % get fraction of licks to target spout on this trial
                            x = length(find(types == Data.SpoutPositions(i)))/length(types);

                            % save the result
                            bins{closest(i)} = [bins{closest(i)} x];
                            plot(ones(length(x),1)*(closest(i)-1), x, 'k.');
                            plot(ones(length(x),1)*(closest(i)-1)+rand(length(x),1)*.5-.25, x, 'k.');
                        end
                    end

                    % plot all timepoints
                    if 0
                        % get mean and std error of on target lick fraction for each lag
                        means = zeros(length(bins),1);
                        errs = zeros(length(bins),1);
                        for i=1:length(bins)
                            means(i) = mean(bins{i});
                            errs(i) = std(bins{i})/sqrt(length(bins{i}));
                        end 

                        % how many lags to plot
                        ee = 12;

                        % make plot
                        x = 0:ee-1;
                        lo = means - errs;
                        hi = means + errs;
                        lo = lo(1:ee);
                        hi = hi(1:ee);
                        hp = patch([x x(end:-1:1) x(1)], [lo; hi(end:-1:1); lo(1)], 'r');
                        hl = line(x,means(1:ee));
                        set(hp, 'facecolor', [1 0.8 0.8], 'edgecolor', 'none');
                        set(hl, 'color', 'r', 'marker', '.');
                        set(gca,'TickDir','out');
                        alpha(0.5);
                        xlim([-.05,ee-1]);
                        ylabel('Fraction of licks to active spout');
                        xlabel('Trials since active spout last changed');
                        ylim([0, 1]);
                        title(tt);
                    end
                    % plot binned by 2
                    if 1
                        rng = 1:2:18;
                        %rng = 1:3:13;
                        means = zeros(length(bins),1);
                        errs = zeros(length(bins),1);
                        bb = cell(length(rng),1);
                        for i=rng
                            bb{i} = [bins{i} bins{i+1}];% bins{i+2}];
                            means(i) = mean(bb{i});
                            errs(i) = std(bb{i})/sqrt(length(bb{i}));
                        end 

                        x = rng-1;
                        lo = means - errs;
                        hi = means + errs;
                        lo = lo(rng);
                        hi = hi(rng);

                        hp = patch([x x(end:-1:1) x(1)], [lo; hi(end:-1:1); lo(1)], 'r');
                        hl = line(x,means(rng));
                        set(hp, 'facecolor', [1 0.8 0.8], 'edgecolor', 'none');
                        set(hl, 'color', 'r', 'marker', '.');
                        set(gca,'TickDir','out');
                        alpha(0.5);
                        xlim([-.05,max(x)]);
                        ylabel('Fraction of licks to active spout');
                        xlabel('Trials since active spout last changed');
                        ylim([0, 1]);
                        try
                            title([tt ', p=' sprintf('%2.3f',ranksum(bb{1}, bb{9}))]);
                        catch
                        end

                    end
                end
                set(gcf, 'InvertHardcopy', 'off');
                saveas(gcf, [mouse '-' nn '-LagPlot.png']);
            end

            %% Make polar plot
            figure('position', [0, 0, 900, 400]);
            set(gcf,'Color','w')
            colors = 'krgy';
            offsets = deg2rad([285, 335, 25, 75]);
            spouts = 1:4;

            if three_spouts
                offsets = deg2rad([285, 0, 75]);
                spouts=[1 3 4];
            end

            before = find(edges >= stim_off, 1, 'first'):find(edges < reward_on, 1, 'last');
            after = find(edges >= reward_on, 1, 'first'):find(edges < reward_off, 1, 'last');

            for x = 1:3
                subplot(1,3,x);
                for spout_pos = spouts
                    ds = densities{spout_pos};
                    if three_spouts
                        ds = ds(:,spouts);
                    end
                    switch x
                        case 1
                            tt = 'all licks';
                        case 2
                            tt = 'before reward';
                            ds = ds(before,:);
                        case 3
                            tt = 'after reward';
                            ds = ds(after,:);
                    end
                    mu = mean(ds);
                    disp(mu)
                    mu = mu / max(mu);
                    polarplot([offsets 0 offsets(1)], [mu 0 mu(1)], colors(spout_pos), 'LineWidth', 3); hold on;
                end
                ax = gca;
                ax.RTick = [];
                ax.LineWidth = 2;
                ax.Color = [.9, .9, .9];
                ax.ThetaTick = sort(rad2deg(offsets));
                ax.ThetaTickLabel = [3 4 1 2];
                ax.ThetaDir = 'counterclockwise';
                ax.ThetaZeroLocation = 'top';
                title(tt);
            end

            set(gcf, 'InvertHardcopy', 'off');
            set(gcf, 'InvertHardcopy', 'off');
            if select_trials    
               saveas(gcf, [mouse '-select_trials-lag' num2str(lag) '-PolarPlot.png']);
            else
               saveas(gcf, [mouse '-PolarPlot.png']);
            end
    end
end