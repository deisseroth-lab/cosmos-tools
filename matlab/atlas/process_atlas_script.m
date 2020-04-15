%%% Generate top-down view of atlas.
%%% Written: March 12, 2018.

%%% USE THIS WEBSITE FOR UPDATED ATLAS
%%% http://help.brain-map.org/display/mouseconnectivity/API
%%%% NOT this one: http://help.brain-map.org/display/mousebrain/API

atlas_loc = '/home/izkula/Dropbox/cosmos_data/atlas/'
addpath(genpath(atlas_loc));

[AVGT, metaAVGT] = nrrdread(fullfile(atlas_loc, 'average_template_25.nrrd'));
[NISSL, metaNISSL] = nrrdread(fullfile(atlas_loc, 'ara_nissl_25.nrrd'));
[ANO, metaANO] = nrrdread(fullfile(atlas_loc, 'annotation_25.nrrd'));
 
%%% Load annotations
c = containers.Map; % will hold annotations

%%% Instructions: To generate the atlas_annotations_trimmed.json from the original
%%% atlas_annotation.json.
% See: https://stackoverflow.com/questions/7842333/delete-matching-search-pattern-in-vim
% In vim:
% :g!/\<\(id\|parent_structure_id\|acronym\|name\)\>/d
% :%s/"//gc
% a
% :%s/,//gc
% a
% :%le
% :%s/id: 0//gc
% a
% dd
% :w %:h/atlas_annotations_trimmed.json

fileName = fullfile(atlas_loc, 'atlas_annotations_trimmed.json');
inputfile = fopen(fileName);

acronyms = {}
parents = {}
names = {}
ids = {}

iter = 1
while 1    
    id_str = fgetl(inputfile);
    % Quit if end of file
    if ~ischar(id_str)
        break
    end
    [t, id] = strtok(id_str);
    id = id(2:end);
    
    acronym_str = fgetl(inputfile);
    [t, acronym] = strtok(acronym_str);
    acronym = acronym(2:end);

    name_str = fgetl(inputfile);
    [t, name] = strtok(name_str);
    name = name(2:end);

    parent_str = fgetl(inputfile);
    [t, parent] = strtok(parent_str);
    parent = parent(2:end);

    s = struct();
    s.acronym = acronym;
    s.name = name;
    s.parent = parent;
    
    c(id) = s;
    
    acronyms{iter} = acronym;
    names{iter} = name;
    parents{iter} = parent;
    ids{iter} = id;
    iter = iter + 1;
end

fclose(inputfile); 

labeled_vol = ANO;
annotations = c;


% Display one coronal section
figure;imagesc(squeeze(AVGT(:,264,:)));colormap(gray(256)); axis equal;
figure;imagesc(squeeze(NISSL(:,264,:)));colormap(gray(256)); axis equal;
figure;imagesc(squeeze(ANO(:,264,:)));
caxis([1,2000]); colormap(lines(256)); axis equal;
 
% Display one sagittal section
figure;imagesc(squeeze(AVGT(:,:,220)));colormap(gray(256)); axis equal;
figure;imagesc(squeeze(NISSL(:,:,220)));colormap(gray(256)); axis equal;
figure;imagesc(squeeze(ANO(:,:,220)));
caxis([1,2000]); colormap(lines(256)); axis equal;


%% Show horizontal sections
doShowHorizontalSections = false
if doShowHorizontalSections
    for i = 320:-1:1
        figure(2);imagesc(squeeze(ANO(i, :, :)), [0, 1121]);
    %     figure(2);imagesc(squeeze(ANO(1:320, :, i)), [0, 1121]);
        pause(0.05);
        max(max(ANO(1:320, i, :)));
        min(min(ANO(:, i, :)));
    end
end

%% Compute top projection
X = size(ANO,2); Y = size(ANO, 3);
top_projection = zeros(X, Y);
for i = 1:X
    i
    for j = 1:Y
        f = find(squeeze(ANO(:, i, j)));
        if ~isempty(f)
            top_projection(i, j) = ANO(f(1), i, j);
        end
    end
end

top_projection = rot90(top_projection);

figure, imagesc(top_projection, [0, 10000])

%%
top_proj_cortex = top_projection;
non_cortex_inds = [834, 1056, 936, 1007, 1041, 1064, 1091, ...
                   944, 828, 313, 820, 1025, 997, 482, 851, 512];
for ind=non_cortex_inds
    top_proj_cortex(top_proj_cortex == ind) = 0;
end
            
figure, imagesc(top_proj_cortex, [0, 1000])
%% Get outline of top projection, clean it up, and save it out
atlas_outline = GetAtlasOutline(top_projection);
cortex_outline = GetAtlasOutline(top_proj_cortex);

dilatedImage = imdilate(atlas_outline,strel('disk',3));
clean_atlas_outline = bwmorph(dilatedImage,'thin',inf);

dilatedImage = imdilate(cortex_outline,strel('disk',3));
clean_cortex_outline = bwmorph(dilatedImage,'thin',inf);

%%% To do: apply median filter?
clean_cortex_outline = medfilt2(clean_cortex_outline, [2,2]);

save(fullfile(atlas_loc, 'atlas_top_projection.mat'), 'top_projection', ...
        'atlas_outline', 'clean_atlas_outline', 'cortex_outline', ...
        'clean_cortex_outline', 'acronyms', ...
        'names', 'parents', 'ids')


    
%% Get a name for an id
id_list = unique(top_projection)
for kk = 1:numel(id_list)
    id = id_list(kk)
    figure, imagesc(top_projection == id)
    
    for i = 1:numel(ids)
        if id == str2num(ids{i})
            title(names{i})
        end
    end
end




