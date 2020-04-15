function out = fwhm(data, doPlot)
% Compute full width half max on a vector
%%% (assumes one peak, with two half max locations, minimal error checking)

% Find the half max value.
halfMax = (min(data) + max(data)) / 2;
% Find where the data first drops below half the max.
index1 = find(data >= halfMax, 1, 'first');
% Find where the data last rises above half the max.
index2 = find(data >= halfMax, 1, 'last');

if doPlot
    plot(data); hold on; plot([index1, index2], [data(index1), data(index2)], 'o-');
end

out = abs(index2-index1) + 1; % FWHM in indexes.

end

