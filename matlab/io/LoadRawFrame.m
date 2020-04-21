function im = LoadRawFrame( imagePath )
%LOADRAWFRAME Load single raw frame

trialName = 'Trial00001';
imageDir = fullfile(imagePath, trialName);
currFiles = dir(fullfile(imageDir, '*.tif'));
numImages = size(currFiles, 1);
fname = fullfile(imageDir, currFiles(2).name);
im = im2double(imread(fname));

end

