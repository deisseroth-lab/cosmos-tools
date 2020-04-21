function MakeAVI(v1, fps, fname, cmap, maxval)
% MakeAVI(v1, fps, fname, cmap, maxval)
% cmap is optional
% maxval sets the contrast
if ~exist('cmap', 'var') || isempty(cmap)
    cmap = 'jet';
end
if ~exist('maxval', 'var') || isempty(maxval)
    maxval = max(v1(:));
end
writer = VideoWriter(fname);
%writer.LosslessCompression = true;
writer.FrameRate = fps;
writer.Quality = 100;

%writer.VideoCompressionMethod = 'H.264';
%writer.CompressionRatio = 2;
open(writer);
figure;
for i=1:size(v1,3)
    if mod(i,10) == 0
        fprintf('%d, ', i);
    end
    if mod(i, 100) == 0
        fprintf('\n');
    end
    imagesc(v1(:,:,i)', [0 maxval]); colormap(hot); axis off; axis square; axis tight;
    set(gcf, 'Color', 'w');
    frame = getframe;
    writeVideo(writer, frame);    
    %pause(1/fps);
end
close(writer);

