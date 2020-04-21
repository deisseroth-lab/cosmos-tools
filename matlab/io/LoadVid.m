function     [vid, atlas, fps, vid405, vid470] = LoadVid(dff_fname, load405, load470)
%%% Load video given full path to the h5 file
%%% For registered and masked videos, it should something like:
%%% fullfile(processedDataDir, dataList{whichTrials(i)}, 'masked_dff.h5')
if ~exist('load405', 'var') || isempty(load405)
    load405 = false;
end

if ~exist('load470', 'var') || isempty(load470)
    load470 = false;
end

vid = h5read(dff_fname,'/vid');
atlas = h5read(dff_fname, '/atlas');
fps = h5read(dff_fname, '/fps');

if load405
    try
        vid405 = h5read(dff_fname, '/dfof405'); 
    catch
        vid405 = [];
    end
else
    vid405 = [];
end

if load470
    try
        vid470 = h5read(dff_fname, '/dfof470'); 
    catch
        vid470 = [];
    end
else
    vid470 = [];
end
    