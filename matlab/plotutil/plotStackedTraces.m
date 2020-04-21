function plotStackedTraces(traces, color, dt, scale, semTraces)
%%% If sem is included, then will errorshade each plot with the provided
%%% standard error of the mean.

nneurons = size(traces, 1);

if ~exist('color', 'var') || isempty(color)
   color = 'k'; 
end

if ~exist('dt', 'var') || isempty(dt)
    dt = 1;
end

if ~exist('scale', 'var') || isempty(scale)
    scale = 1;
end

if ~exist('semTraces', 'var') || isempty(semTraces)
    semTraces = [];
end

for kk = 1:nneurons
    hold on
    if ~isempty(semTraces)
        errorshade(dt*[0:size(traces, 2)-1], kk + traces(kk, :)/max(traces(kk, :))*scale, semTraces,[0 1 0],'errorAlpha',0.2);
    else
        plot(dt*[0:size(traces, 2)-1], kk + traces(kk, :)/max(traces(kk, :))*scale, 'Color', color, 'LineWidth', 1)
    end
end

ylim([0, nneurons+2])
    