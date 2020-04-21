%%% register with normcorre
addpath(genpath('/home/izkula/src/COSMOS'))


dirname = '/home/izkula/Data/processedData/20180102/m52_1wk_post_tamox_2/patch-1/1/'

frames = [1:500];
vid = LoadImageStack(dirname, frames);

%%
options = NoRMCorreSetParms('d1',size(vid,1),'d2',size(vid,2), 'grid_size',  [50, 50, 1], 'max_shift', [30,30,1]) );
[M_final,shifts,template,options,col_shift] = normcorre(vid,options);