%%% This function loads PSFs, plots their cross sections, etc. 


addpath(genpath('~/src/COSMOS/matlab'))
baseDir = '/home/izkula/Data/data/';
processedDataDir = '/home/izkula/Data/processedData/';
close all
datasetName = 'v6';

saveDir = ['/home/izkula/Data/Results/COSMOS_PSF/'];
saveDir = fullfile(saveDir, datasetName);
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

doLoadCentroids = true;
doSavePlots = true;
plotSuffix = '.png'

dz = 100; %%% um per zstep

fsize = 20; %%% font size for plots
e_to_p = 0.46; %%% electron to photon conversion for orca camera


%%%% {'date', 'fname', 'label', pixel size in um}
% data = {...
%     {'20171109', 'f1.2_100umstep_100ms', 'f1.2', 6.5}; ...
%     {'20171109', 'f4_100umstep_100ms', 'f4', 6.5}; ...
% %     {'20171109', 'f4_100umstep_100ms_second_time_after_f8', 'f4-2', 6.5}; ...
%     {'20171109', 'f8_100umstep_100ms', 'f8', 6.5}; ...
%     {'20171109', 'ac127_050-no_relay_to_sensor', '12.7lenslet-no-relay', 6.5}; ....
%     {'20171109', 'ac127-050-BL_10umbeads_100umstep_100ms', '12.7lenslet-BL', 11.9}; ...
%     {'20171109', 'ac127-050-BR_10umbeads_100umstep_100ms', '12.7lenslet-BR', 11.9}; ...
%     {'20171109', 'ac127-050-TL_10umbeads_100umstep_100ms', '12.7lenslet-TL', 11.9}; ...
%     {'20171109', 'ac127-050-TR_10umbeads_100umstep_100ms', '12.7lenslet-TR', 11.9}; ...
%     {'20171109', 'ac127-050-center_10umbeads_100umstep_100ms', '12.7lenslet-center', 11.9};
% %     {'20171109', 'ac127_050_before_dichroic_then_50f1.4_50f1.4', '12.7lenslet-bf-dichroic', 11.9};
% };
% 
% data = {...
%     {'20171110', 'f1.2_1', 'f1.2', 6.5}; ...
%     {'20171110', 'f4_1', 'f4', 6.5}; ...
%     {'20171110', 'lenslet_4x_ac127-050_direct_camera_BL', 'BL-direct', 6.5}; ...
%     {'20171110', 'lenslet_4x_ac127-050_direct_camera_TL', 'TL-direct', 6.5}; ...
%     {'20171110', 'lenslet_4x_ac127-050_direct_camera_TR', 'TR-direct', 6.5}; ...
%     {'20171110', 'lenslet_4x_ac127-050_direct_camera_BR', 'BR-direct', 6.5}; ...
% }

% data = {...
%     {'20171110', 'f1.2_3_at_end', 'f1.2', 6.5}; ...
%     {'20171110', 'f4_3_at_end', 'f4', 6.5}; ...
%     {'20171110', 'f8_3_at_end', 'f8', 6.5}; ...
%     {'20171110', 'single_lenslet_r10_3', 'r10', 6.5}; ...
%     {'20171110', 'single_lenslet_l10_3', 'l10', 6.5}; ...
%     {'20171110', 'single_lenslet_b10_3', 'b10', 6.5}; ...
%     {'20171110', 'single_lenslet_t10_3', 't10', 6.5}; ...
%     {'20171110', 'single_lenslet_t5_3', 't5', 6.5}; ...
%     {'20171110', 'single_lenslet_centered_3', 'centered', 6.5}; ...
% }
    
% data = {...
%     {'20171114', 'f1.2_25um_2', 'f1.2', 6.5}; ...
%     {'20171114', 'f4_25um_2', 'f4', 6.5}; ...
%     {'20171114', 'f8_25um_2', 'f8', 6.5}; ...
%     {'20171114', 'BL_25um_2', 'BL', 6.5}; ...    
%     {'20171114', 'BR_25um_2', 'BR', 6.5}; ...    
%     {'20171114', 'TL_25um_2', 'TL', 6.5}; ...    
%     {'20171114', 'TR_25um_2', 'TR', 6.5}; ...    
%     {'20171114', 'lenslet_55mmf1.2_35mmf2_2', '55-35', 6.5*55/35}; ...
%     {'20171114', 'lenslet_63mmMedFormf2.8_35mmf2_2', '63-35', 6.5*63/35}; ...
% }


data = {...
    {'20171207', 'f1.2_psf', 'f1.2', 11}; ...
    {'20171207', 'f4_psf', 'f4', 11}; ...
    {'20171207', 'f8_psf', 'f8', 11}; ...
    {'20171207', 'v6_psf', 'BL', 11/.8}; ...    
    {'20171207', 'v6_psf', 'BR', 11/.8}; ...    
    {'20171207', 'v6_psf', 'TL', 11/.8}; ...    
    {'20171207', 'v6_psf', 'TR', 11/.8}; ...    
}



%% Load PSFs and select rough center of PSF

allcx_rough = [];
allcy_rough = [];
dx = []; %%% Calibrated pixel size in microns for each dataset (includes magnification)
allPSF = {};
for kk = 1:numel(data)
    d = data{kk};
    date = d{1};
    name = d{2};
    label = d{3};
    dx(kk) = d{4};
    imageDir = fullfile(baseDir, date, name);
    disp(['loading ', imageDir])
    PSF = LoadImageStack( imageDir);
    allPSF{kk} = PSF; 
    close all
end

for kk = 1:numel(data)
    if ~doLoadCentroids
        PSF = allPSF{kk};
        figure('Position', [500,500,1500,1500]), imagesc(max(PSF, [], 3));
        title('click on center of bead')
        [x, y] = ginput(1);
        allcx_rough(kk) = round(x);
        allcy_rough(kk) = round(y);    
    end
end
close all

if ~doLoadCentroids
    save(fullfile(saveDir, [datasetName, '_centroids_rough.mat']), 'allcx_rough', 'allcy_rough', 'data')
else
    load(fullfile(saveDir, [datasetName, '_centroids_rough.mat']), 'allcx_rough', 'allcy_rough', 'data')
end


%% Crop and then reselect centroid in zoomed in version
cropx = 70;
cropy = 70;

allPSFcrop = {};
allcx_precise = [];
allcy_precise = [];

for kk = 1:numel(data)
    cropx_coords = allcx_rough(kk) - cropx: allcx_rough(kk) + cropx;
    cropy_coords = allcy_rough(kk) - cropy: allcy_rough(kk) + cropy;
    PSF = allPSF{kk};
    cropPSF = PSF(cropy_coords, cropx_coords, :);
    allPSFcrop{kk} = cropPSF;
    
    if ~doLoadCentroids
        figure('Position', [500, 500, 1500, 1500]), imagesc(max(cropPSF, [], 3));
        title('click on center of bead')
        [x, y] = ginput(1);
        allcx_precise(kk) = round(x);
        allcy_precise(kk) = round(y);    
    end
end

if ~doLoadCentroids
    save(fullfile(saveDir, [datasetName, '_centroids_precise.mat']), 'allcx_precise', 'allcy_precise', 'data')
else
    load(fullfile(saveDir, [datasetName, '_centroids_precise.mat']), 'allcx_precise', 'allcy_precise', 'data')
end

close all


%% Plot line through PSF across depth

all_xcross = {};
all_ycross = {};
all_xycross = {};
all_yxcross = {};

for kk = 1:numel(data)
    cropPSF = allPSFcrop{kk};
    cx = allcx_precise(kk);
    cy = allcy_precise(kk);

    useSlice = false
    if useSlice
        all_xcross{kk} = squeeze(cropPSF(cy, 1:end,:));
        all_ycross{kk} = squeeze(cropPSF(1:end, cx,:));
    else
        all_xcross{kk} = squeeze(max(cropPSF(:, 1:end,:), [], 1));
        all_ycross{kk} = squeeze(max(cropPSF(1:end, :,:), [], 2));
    end
    
    %%% all_xycross
    %%% all_yxcross
end

clear allPSF

%% Crop all cross sections to be the same length, centered on the max of the PSF
nz = 10;
nx = 20;

allPSFtrim =  zeros(nx*2 + 1, nx*2 + 1, nz*2+1, numel(data));
all_xcross_trim = zeros(nx*2 + 1, nz*2+1, numel(data));
all_ycross_trim = zeros(nx*2 + 1, nz*2+1, numel(data));
for kk = 1:numel(data)
    xcross = all_xcross{kk};
    ycross = all_ycross{kk};
    [max_per_x, maxxs] = max(xcross, [], 1);    
    [max_per_y, maxys] = max(ycross, [], 1);
    [max_per_z, maxz] = max(max_per_x);
    maxx = maxxs(maxz);
    maxy = maxys(maxz);
        
    xcross_trim = xcross(maxx - nx:maxx + nx, maxz - nz: maxz + nz);
    ycross_trim = ycross(maxy - nx:maxy + nx, maxz - nz: maxz + nz);
    all_xcross_trim(:, :, kk) = xcross_trim - min(xcross_trim(:));
    all_ycross_trim(:, :, kk) = ycross_trim - min(ycross_trim(:));
    
%     figure, imagesc(xcross), hold on, plot(maxz, maxx, 'ro')
    zvals = squeeze(dz*[1:size(all_xcross_trim, 2)]');
    zvals = zvals - zvals(round(numel(zvals)/2));
    xvals = dx(kk)*[1:size(all_xcross_trim, 1)];
    xvals = xvals - xvals(round(numel(xvals/2)));
    figure('Position', [500, 500, 1500, 1500]), imagesc(zvals, xvals, xcross_trim)
    colorbar
    title(data{kk}{3})
    xlabel('z [um]')
    ylabel('x [um]')
    if doSavePlots
       export_fig(fullfile(saveDir, ['x_cross_section_im_', data{kk}{3}, plotSuffix]))
    end
    
    psfCrop = allPSFcrop{kk};
    allPSFtrim(:,:,:, kk) = psfCrop(maxx - nx:maxx + nx, maxy - nx:maxy + nx, maxz - nz: maxz + nz);
end

%% Compute full width half max at each depth

nz = size(all_xcross_trim, 2);
all_xfwhm = zeros(nz, numel(data));
all_yfwhm = zeros(nz, numel(data));

for kk = 1:numel(data)
    xcross_trim = all_xcross_trim(:,:,kk);
    ycross_trim = all_ycross_trim(:,:,kk);
    
    figure
    for zz = 1:nz
        all_xfwhm(zz, kk) = fwhm(xcross_trim(:, zz), true);
        all_yfwhm(zz, kk) = fwhm(ycross_trim(:, zz), false);
%         pause(0.1)
    end
    title(data{kk}{2})

end
%% Compare cross sections

for kk = 1:size(all_xcross_trim, 3)
    d = data{kk};
    figure, plot(all_xcross_trim(:,:,kk))
    title([d{3}, ' xcross'])
    xlim([0, nx*2 + 1])
    if doSavePlots
       export_fig(fullfile(saveDir, ['x_cross_section_', data{kk}{3}, plotSuffix]))
    end
end

%% Compare intensity fall off

zvals = dz*[1:size(all_xcross_trim, 2)];
zvals = zvals - zvals(round(numel(zvals)/2));
figure('Position', [500, 500, 1000, 1000])
cc = hsv(numel(data));
for kk = 1:size(all_xcross_trim, 3);
    plot(zvals, squeeze(all_xcross_trim(round(size(all_xcross_trim,1)/2), :, kk))/(dx(kk)^2), 'Linewidth', 2, 'color', cc(kk, :))
    ylabel('defocus depth [um]')
    title('power density along optical axis vs. defocus')
    hold on
    labels{kk} = data{kk}{3};
end
legend(labels)
if doSavePlots
   export_fig(fullfile(saveDir, ['intensity_falloff_', data{kk}{3}, plotSuffix]))
end
%% Compare normalized intensity fall off 
zvals = dz*[1:size(all_xcross_trim, 2)];
zvals = zvals - zvals(round(numel(zvals)/2));
figure('Position', [500, 500, 1000, 1000])

cc = hsv(numel(data));
for kk = 1:size(all_xcross_trim, 3)
    norm_xcross = squeeze(all_xcross_trim(round(size(all_xcross_trim, 1)/2), :, kk));
    norm_xcross = norm_xcross/max(norm_xcross(:));
    plot(zvals, norm_xcross, 'Linewidth', 2, 'color', cc(kk, :));
    ylabel('defocus depth [um]')
    title('normalized central intensity vs. defocus')
    hold on
    labels{kk} = data{kk}{3};
end
legend(labels)
if doSavePlots
   export_fig(fullfile(saveDir, ['intensity_falloff_normalized_', data{kk}{3}, plotSuffix]))
end
%% Compare depth of field based on full width half max

zvals = dz*[1:size(all_xcross_trim, 2)];
zvals = zvals - zvals(round(numel(zvals)/2));
figure('Position', [500, 500, 1000, 1000])
cc = hsv(numel(data));
for kk = 1:size(all_xcross_trim, 3)    
    plot(zvals, squeeze(dx(kk)*all_xfwhm(:, kk)), 'Linewidth', 2, 'color', cc(kk, :))
    ylabel('full width half max [um]')
    title('fwhm vs. defocus')
    hold on
    labels{kk} = data{kk}{3};
end
legend(labels)
if doSavePlots
   export_fig(fullfile(saveDir, ['fwhm_vs_defocus_', data{kk}{3}, plotSuffix]))
end

%% Compare maximum intensity for each dataset

max_intensities = squeeze(max(max(all_xcross_trim, [], 2), [], 1));

figure('Position', [500, 500, 1000, 1000])
bar(e_to_p*max_intensities./(dx'.^2));
title('max intensity')
ylabel('max photons density')
set(gca,'xticklabel',labels)
set(gca,'FontSize',fsize);
set(gca, 'XTickLabelRotation', 45)
if doSavePlots
   export_fig(fullfile(saveDir, ['max_intensity_comparison_', data{kk}{3}, plotSuffix]))
end


%% Compare photon density per neuron

%%% Approximate mean photons per pixel, computed as max val at focus
%%% divided by FWHM, at each z-depth

zvals = dz*[1:size(all_xcross_trim, 2)];
zvals = zvals - zvals(round(numel(zvals)/2));
figure('Position', [500, 500, 1000, 1000])

cc = hsv(numel(data));
for kk = 1:size(all_xcross_trim, 3)    
    fwhm_area = (dx(kk)*all_xfwhm(:,kk)).^2;
    max_photons = e_to_p*max_intensities(kk);
    photons_per_pixel = max_photons./fwhm_area;
    
    plot(zvals, log10(photons_per_pixel), 'Linewidth', 2, 'color', cc(kk, :))
    ylabel('max(photons)/(fwhm.^2) [um]')
    title('photons per pixel vs. defocus')
    hold on
    labels{kk} = data{kk}{3};
end
legend(labels)
if doSavePlots
   export_fig(fullfile(saveDir, ['photons_per_pixel_vs_defocus_', data{kk}{3}, plotSuffix]))
end


%% Compute total intensity at every depth
sumPerZ = zeros(numel(allPSFcrop), size(allPSFcrop{1}, 3));
for i = 1:numel(allPSFcrop)
    psfCrop = allPSFcrop{i};
%     VideoSlider(psfCrop)
    sumPerZ(i, :) = squeeze(sum(sum(psfCrop, 1), 2))' - size(psfCrop,1)*size(psfCrop,2)*quantile(psfCrop(:), 0.1);
    sumPerZ(i, :) = squeeze(max(max(psfCrop, [], 1), [], 2))';

end

figure
for i = 1:numel(allPSFcrop)
    plot(sumPerZ(i, :)); hold on;
    labels{i} = data{i}{3};
end
legend(labels)