function PlotVerticalLines(times, minY, maxY, cmap)

if ~exist('minY', 'var') || isempty(minY)
    YL = ylim;
    minY = YL(1);
    maxY = YL(2);
end

if ~exist('cmap', 'var') || isempty(cmap)
    cmap = 'k--';
end
hold on;
for i= 1:numel(times)
    plot([times(i), times(i)], [minY, maxY], cmap, 'LineWidth', 2)
end
hold off;