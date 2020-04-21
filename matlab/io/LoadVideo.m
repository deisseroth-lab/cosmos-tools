function data = LoadVideo(filename)

if ~exist('filename', 'var') || isempty(filename)
    [filename, pathname] = uigetfile({'*.mp4'; '*.avi'}, 'Pick a video', 'E:\Data');
    filename = [pathname, filename]
end

vid = VideoReader(filename);
data = zeros(vid.Width, vid.Height, vid.FrameRate*ceil(vid.Duration));
numFrames = size(data, 3)

if numFrames < 3000
    for i=1:size(data,3)    
        if mod(i, 20) == 0
            disp(['Load video, frame: ', num2str(i)])
        end
        try
            temp = readFrame(vid);
        catch
        end
        data(:, :, i) = temp(:, :, 1)';
    end
else
    parfor i=1:size(data,3)
        i
        try
            temp = readFrame(vid);
        catch
        end
        data(:, :, i) = temp(:, :, 1)';
    end
end