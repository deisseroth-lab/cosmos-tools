function  fnames = SaveMultipageTiff( stack, saveDir, prefix, keepScaling)
%SaveTiffStack
% stack - a 3 dimensional matrix
% savePath - string i.e. 'directory/prefix_'

if nargin < 3
    prefix = '';
end

if ~exist('keepScaling', 'var') || isempty(keepScaling)
   keepScaling = false 
end

if keepScaling
    saveStack = uint16(double(max(stack(:)))*mat2gray(stack));
else
    saveStack = uint16(2^16*mat2gray(stack));
end

if ~exist(saveDir, 'dir'); mkdir(saveDir); end
fnames = {};
for k = 1:size(saveStack,3)-1
    fnames{k} = [saveDir, '/', prefix, '.tif'];
    imwrite(saveStack(:,:,k), fnames{k}, 'compression', 'none', 'writemode', 'append');
end

end
