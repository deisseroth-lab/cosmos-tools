function [mov, framerate, frames] = LoadMP4(vidfname);

try
    obj = VideoReader([vidfname]);

    vidWidth = obj.Width;
    vidHeight = obj.Height;
    mov = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
        'colormap',[]);
    k = 1;
    disp(['loading ', vidfname])
    while hasFrame(obj)
        if mod(k,100) == 0
            fprintf('%d, ', k);
        end
        mov(k).cdata = readFrame(obj);
        k = k+1;
    end
    fprintf('\n\n');

    framerate = obj.FrameRate;

    f1 = mov(1).cdata(:,:,1);
    frames = zeros(size(f1, 1), size(f1, 2), length(mov));
    for i = 1:length(mov)
        frames(:,:,i) = mov(i).cdata(:,:,1);
    end
catch
    disp('Matlab''s VideoReader did not work. Trying mmread (from matlab file exchange)')
    try         
        disp(['loading ', vidfname])
        obj = mmread([vidfname]);
        
%         vidWidth = obj.width;
%         vidHeight = obj.height;
        nframes = obj.nrFramesTotal;
        mov = obj.frames;
        
        framerate = obj.rate;

        f1 = mov(1).cdata(:,:,1);
        frames = zeros(size(f1, 1), size(f1, 2), length(mov));
        for i = 1:length(mov)
            frames(:,:,i) = mov(i).cdata(:,:,1);
        end
        
    catch
        error('Could not load mp4 file - this is linux"s fault')
    end
end