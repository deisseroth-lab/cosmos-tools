function atlas_outline = GetAtlasOutline(aligned_atlas, just_edge)
% Get outline version of atlas

if ~exist('just_edge', 'var') || isempty(just_edge)
    just_edge = false
end

atlas_outline = zeros(size(aligned_atlas));

if just_edge
    atlas_outline = bwperim(aligned_atlas>0 & aligned_atlas ~= 834);
else

    id_list = unique(aligned_atlas);
    id_list = id_list(id_list > 0);
    for k = 1:length(id_list)
        id = id_list(k);
        mask = (aligned_atlas == id);
        outline = bwperim(mask);
        atlas_outline = atlas_outline | outline;
    end
end