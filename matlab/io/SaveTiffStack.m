function  fnames = SaveTiffStack( stack, saveDir, prefix)
%SaveTiffStack
% stack - a 3 dimensional matrix
% savePath - string i.e. 'directory/prefix_'

if nargin < 3
    prefix = '';
end

stack = uint16(2^16*mat2gray(stack));

mkdir(saveDir);
fnames = {};
for k = 1:size(stack,3)
    fnames{k} = [saveDir, '/', prefix, num2str(k, '%05d'), '.tif'];
    imwrite(stack(:,:,k), fnames{k}, 'compression', 'none');
end

end
