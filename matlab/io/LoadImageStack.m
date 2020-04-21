function currImages = LoadImageStack( imageDir, subSeq)
% Load single color image stack from folder

if imageDir(end-3:end) == '.tif'
    currFiles = dir(imageDir);
    fnameProvided = true
else
    currFiles = dir(fullfile(imageDir, '*.tif'));
    fnameProvided = false
end

if size(currFiles, 1) == 0
    imageDir = fullfile(imageDir, 'Pos0')
    currFiles = dir(fullfile(imageDir, '*.tif'));
    
    if size(currFiles, 1) == 0
        warning(['No files found in path: ', imageDir])
        return
    end
end
    
    
numImages = size(currFiles, 1);

if numImages == 1
    doLoadMultipage = true;
else
    doLoadMultipage = false;
end

if doLoadMultipage
    if fnameProvided
       fname = imageDir;
    else
        fname = fullfile(imageDir, currFiles(1).name);
    end
    info = imfinfo(fname);
    num_images = numel(info);
    for j = subSeq
        try
            currImages(:,:, j-subSeq(1)+1)  = imread(fname, j, 'Info', info);
        catch e
            disp(e)
        end
    end
else
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
            currImages(:, :, j-subSeq(1)+1) = imread(imName);            
        catch e
            disp(e)
        end
    end
end

end