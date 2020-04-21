function orderedInds = plotOrderedTraces(traces, T)
%%%%% Given a neuron x time matrix (i.e. of the averages across trials)
%%%%% Plots them such that the peak times are ordered

[m, mind] = max(traces, [], 2);

[s, orderedInds] = sort(mind, 'ascend');


if ~exist('T', 'var') || isempty(T)
    imagesc(traces(orderedInds,:));
else
    imagesc(T, 1:numel(orderedInds), traces(orderedInds,:));
end