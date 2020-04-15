function VideoSlider(data, scaleBar, titles)
%%% Usage: VideoSlider(data) where data is an [m x n x t] image stack. 
%%%        Allows continuous sliding through a video. 

if ~exist('scaleBar', 'var') || isempty(scaleBar)
    scaleBar = [0, max(data(:))];
end

if ~exist('titles', 'var') || isempty(titles)
    titles = [];
end

figsize = 1000;
f = figure;
set(f, 'Position', [100, 100, figsize, figsize]);
% cm = bluewhitered(64, scaleBar);
% cm = gray(64);
cm = parula(128);
hplot = imagesc(data(:,:,1), scaleBar); colormap(cm);
nsteps = size(data,3);
h = uicontrol('style','slider','units','pixel', 'Min', 1, 'Max', nsteps, ...
              'sliderStep', [1/(nsteps-1) , 1/(nsteps-1) ], 'Value', 1, 'position',[10, 10, figsize-20 20]);
addlistener(h,'ContinuousValueChange',@(hObject, event) makeplot(hObject, event,data,scaleBar, hplot, titles));

function makeplot(hObject,event,data, scaleBar, hplot, titles)
n = round(get(hObject,'Value'));
imagesc(data(:,:,n), scaleBar); 
titlestr = [num2str(n), ': ', num2str(round(sum(sum(data(:,:,n)> 0))), '%4.2e')];
if ~isempty(titles)
    titlestr = [titlestr, '; ', num2str(titles(n), '%4.2f')];
end
title(titlestr);
drawnow;            
