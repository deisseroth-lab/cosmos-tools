function currImages = LoadImageStack_orig( imageDir, subSeq)
% Load single color image stack from folder

currFiles = dir(fullfile(imageDir, '*.tif'));
numImages = size(currFiles, 1);
fname = fullfile(imageDir, currFiles(1).name);
im = imread(fname);
imSize = size(im);
if ~exist('subSeq', 'var') || isempty(subSeq)
    subSeq = 1:numImages;
end

currImages = zeros(imSize(1), imSize(2), length(subSeq));
for j=subSeq
    
    imName = fullfile(imageDir,  currFiles(j).name);
     try
        currImages(:, :, j-subSeq(1)) = imread(imName);            
    catch e
        disp(e)
    end
end

end