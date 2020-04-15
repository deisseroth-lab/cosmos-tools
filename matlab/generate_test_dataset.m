%%%% Generate a test dataset for cosmos/test/test_cosmos_dataset.py
%%% Copy the output of this file into cosmos/test/test_cosmos_dataset dir

fdir =  '/media/Data/processedData/20180227/cux2m72_COSMOSTrainMultiBlockGNG_1/';
outdir = '/media/Data/processedData/20180227/testdata_cux2m72_COSMOSTrainMultiBlockGNG_1/';
top_fname = fullfile(fdir, 'top/top_source_extraction/top_out.mat');
bot_fname = fullfile(fdir, 'bot/bot_source_extraction/bot_out.mat');

top = load(top_fname);
bot = load(bot_fname);

top_small = top;
bot_small = bot;


ncells = 10
start = 105
nt = 500
top_small.A = top_small.A(:, start:start+ncells-1); 
top_small.C = top_small.C(start:start+ncells-1, 1:nt);
top_small.C_raw = top_small.C_raw( start:start+ncells-1, 1:nt);
top_small.S = top_small.S( start:start+ncells-1,1:nt);
top_small.W = {}
top_small.Coor = top_small.Coor( start:start+ncells-1);

bot_small.A = bot_small.A(:,  start:start+ncells-1); 
bot_small.C = bot_small.C( start:start+ncells-1, 1:nt);
bot_small.C_raw = bot_small.C_raw( start:start+ncells-1, 1:nt);
bot_small.S = bot_small.S( start:start+ncells-1,1:nt);
bot_small.W = {}
bot_small.Coor = bot_small.Coor( start:start+ncells-1);

mkdir(outdir);
   
tic
neuron = top_small;
save([outdir, '/', 'top/','top', '_out.mat'],'-v6','-struct','neuron');
toc

tic
neuron = bot_small;
save([outdir, '/', 'bot/', 'bot', '_out.mat'],'-v6','-struct','neuron');
toc