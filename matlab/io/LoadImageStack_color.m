function currImages = LoadImageStack_color( fname, subSeq)
% Load single color image stack from folder
% Input is a multipage tif filename


info = imfinfo(fname);
num_images = info.FileSize/(info.Width*info.Height*info.BitDepth)
imSize = [info.Height, info.Width]
if ~exist('subSeq', 'var') || isempty(subSeq)
    subSeq = [1:num_images];
end
currImages = zeros(imSize(1), imSize(2), 3, numel(subSeq));
for j = subSeq
    try
        currImages(:,:, :, j-subSeq(1)+1)  = imread(fname, j);
    catch e
        disp(e)
    end
end

% 
% currFiles = dir(fullfile(imageDir, '*.tif'));
% 
% if size(currFiles, 1) == 0
%     imageDir = fullfile(imageDir, 'Pos0')
%     currFiles = dir(fullfile(imageDir, '*.tif'));
%     
%     if size(currFiles, 1) == 0
%         warning(['No files found in path: ', imageDir])
%         return
%     end
% end
%     
%     
% numImages = size(currFiles, 1);
% 
% if numImages == 1
%     doLoadMultipage = true;
% else
%     doLoadMultipage = false;
% end
% 
% if doLoadMultipage
%     fname = fullfile(imageDir, currFiles(1).name);
%     info = imfinfo(fname);
%     num_images = numel(info);
%     currImages = zeros(imSize(1), imSize(2), 3, numel(subSeq));
%     for j = subSeq
%         try
%             currImages(:,:, j-subSeq(1)+1)  = imread(fname, j, 'Info', info);
%         catch e
%             disp(e)
%         end
%     end
% else
%     fname = fullfile(imageDir, currFiles(1).name);
%     im = imread(fname);
%     imSize = size(im);
%     if ~exist('subSeq', 'var') || isempty(subSeq)
%         subSeq = 1:numImages;
%     end
% 
%     currImages = zeros(imSize(1), imSize(2), length(subSeq));
%     for j=subSeq
% 
%         imName = fullfile(imageDir,  currFiles(j).name);
%         try
%             currImages(:, :, j-subSeq(1)+1) = imread(imName);            
%         catch e
%             disp(e)
%         end
%     end
% end
% 
% end