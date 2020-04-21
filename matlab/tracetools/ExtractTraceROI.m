function [meanTrace, sumTrace, nPixels] = ExtractTraceROI(vid, x, y, nPixelsROI)

if ~exist('nPixelsROI', 'var') || isempty(nPixelsROI)
    nPixelsROI = 3;
end


d = floor(nPixelsROI/2);
x = round(x);
y = round(y);
cropX = [x-d:x+d];
cropY = [y-d:y+d];

cropVid = vid(cropY, cropX,:);

meanTrace = squeeze(mean(mean(cropVid, 2), 1));
sumTrace = squeeze(sum(sum(cropVid, 2), 1));
nPixels = numel(cropX)*numel(cropY);
