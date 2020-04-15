function [Y, d1, d2, numFrame] = snr_choose_data_stack(dir_nm, frames)

% global data Ysiz d1 d2 numFrame; 

%% select dir
if ~exist('dir_nm', 'var') || isempty(dir_nm)
    dir_nm = uigetdir('/home/izkula/Data/data');
end

if ~exist('num2read', 'var') || isempty(num2read)
    num2read = [];
end

%%
% currImages = LoadImageStack(dir_nm, frames); %, 1:num2read);

dir_nm2 = fullfile(dir_nm, 'Pos0');
currImages = LoadImageStack_orig(dir_nm2, frames); %

Y = currImages;

d1 = size(currImages, 1)
d2 = size(currImages, 2)
numFrame = size(currImages, 3)

fprintf('\nThe data has been mapped to RAM. It has %d X %d pixels X %d frames. \nLoading all data requires %.2f GB RAM\n\n', d1, d2, numFrame, d1*d2*numFrame*8/(2^30));

%% select file 
% if ~exist('nam', 'var') || isempty(nam)
%     % choose files manually 
%     try
%         load .dir.mat; %load previous path
%     catch
%         dir_nm = [cd(), filesep]; %use the current path
%     end
%     [file_nm, dir_nm] = uigetfile(fullfile(dir_nm, '*.tif;*.mat;*.h5'));
%     nam = [dir_nm, file_nm];  % full name of the data file
%     [dir_nm, file_nm, file_type] = fileparts(nam);
% else
%     % use pre-specified file 
%     if exist(nam, 'file')
%         [dir_nm, file_nm, file_type] = fileparts(nam);
%     else
%         dir_nm = 0; 
%     end
% end
% if dir_nm~=0
%     save .dir.mat dir_nm;
% else
%     fprintf('no file was selected. STOP!\n');
%     return;
% end

% %% convert the data to mat file
% nam_mat = [dir_nm, filesep, file_nm, '.mat'];
% if strcmpi(file_type, '.mat')
%     fprintf('The selected file is *.mat file\n');
% elseif  exist(nam_mat, 'file')
%     % the selected file has been converted to *.mat file already
%     fprintf('The selected file has been replaced with its *.mat version.\n');
% elseif or(strcmpi(file_type, '.tif'), strcmpi(file_type, '.tiff'))
%     % convert
%     tic;
%     fprintf('converting the selected file to *.mat version...\n');
%     nam_mat = tif2mat(nam);
%     fprintf('Time cost in converting data to *.mat file:     %.2f seconds\n', toc);
% elseif or(strcmpi(file_type, '.h5'), strcmpi(file_type, '.hdf5'))
%     fprintf('the selected file is hdf5 file\n'); 
%     temp = h5info(nam);
%     dataset_nam = ['/', temp.Datasets.Name];
%     dataset_info = h5info(nam, dataset_nam);
%     dims = dataset_info.Dataspace.Size;
%     ndims = length(dims);
%     d1 = dims(2); 
%     d2 = dims(3); 
%     numFrame = dims(end);
%     Ysiz = [d1, d2, numFrame]; 
%     fprintf('\nThe data has %d X %d pixels X %d frames. \nLoading all data requires %.2f GB RAM\n\n', d1, d2, numFrame, prod(Ysiz)*8/(2^30));
%     return; 
% else
%     fprintf('The selected file type was not supported yet! email me to get support (zhoupc1988@gmail.com)\n');
%     return;
% end
% 
% %% information of the data 
% data = matfile(nam_mat);
% Ysiz = data.Ysiz;
% d1 = Ysiz(1);   %height
% d2 = Ysiz(2);   %width
% numFrame = Ysiz(3);    %total number of frames

