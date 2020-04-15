
addpath(genpath('~/src/COSMOS'))
set(0,'defaultfigurecolor', 'w')
% saveDir = '~/Dropbox/cosmos/SNR_analysis/simulation/'
% saveDir = '~/Dropbox/cosmos/SNR_analysis/simulation_withQE/'
saveDir = '~/Dropbox/cosmos/fig_plots/figS2/with4lenslets/'

if ~exist(saveDir); mkdir(saveDir); end

% suffix = '.pdf'
suffix = '.pdf'
% suffix = ''

fsize = 10; % font size for plots

% target_dof = 2e-3; %%% Required overall depth of field. 2 mm
% dof = @(N, c, m) 2*N*c*(m+1)/m^2 %%% From wikipedia
% acceptable_coc = 6.5e-6; %%% Acceptable circle of confusion. Pixel size.

% xmax = (11/2)*1e-3;
xmax = (10/2)*1e-3;
rcurve = 11e-3;
%whichCases = [1:5,7,8,10,11]
% % % % % % % % whichCases = [1,3,4,7,8,11] %%% **Use this for presentation**
% whichCases = [1,3,4,10, 13, 14, 15, 16, 17, 18] 
whichCases = [1,2,3,4,10, 17, 18] %%%% This has the settings for final plot
whichCases = [1,2,3,4, 8, 10, 17, 18] %%%% This has the settings for final plot with 4 lenslets

% % %  whichCases = [1,2,3,4,18]

%%% User options %%%
doWindowZ = true %%% True: make plots vs. z; False: plot according to curvature of the window.
% zdefault = zmax/1.8 %%%% Use this if you assume that you focus somewhere in the middle


lambda = 530e-9;


if doWindowZ
    %%%% Z value for each lateral position
    xvals = linspace(-xmax, xmax, 1000);
    zvals = sqrt(rcurve.^2 - xvals.^2);
    zvals = zvals - min(zvals(:))
    zmax = max(zvals(:))
     zdefault = zmax;
%     zdefault = zmax/1.8;
else
    zvals = linspace(0, 1.5e-3, 1000);
    xvals = zvals;
    zmax = max(zvals(:));
    zdefault = 0;
end


blurs = zeros(numel(whichCases), numel(zvals));
diffraction_blurs = zeros(numel(whichCases),1);
light_collections = zeros(numel(whichCases), 1);
labels = {}

for ind = 1:numel(whichCases)
    whichCase = whichCases(ind)
    switch whichCase
        case 1
            N = 1.4
            f1 = 50e-3 
            f2 = 50e-3
            z0 = zdefault
            label = 'f/1.4'
            D = f1/N
            aperA = pi*(D/2)^2;
%             N = 1.2
%             f1 = 50e-3 
%             f2 = 50e-3
%             z0 = zdefault
%             label = 'f/1.2'
%             D = f1/N
%             aperA = pi*(D/2)^2;
        case 2
            N = 2
            f1 = 50e-3 
            f2 = 50e-3
            z0 = zdefault
            label = 'f/2'
            D = f1/N
            aperA = pi*(D/2)^2;
        case 3
            N = 4
            f1 = 50e-3 
            f2 = 50e-3
            z0 = zdefault
            label = 'f/4'
            D = f1/N
            aperA = pi*(D/2)^2;
        case 4
            N = 8
            f1 = 50e-3 
            f2 = 50e-3
            z0 = zdefault
            label = 'f/8'
            D = f1/N
            aperA = pi*(D/2)^2;
        case 5
            N = 16
            f1 = 50e-3 
            f2 = 50e-3
            z0 = zdefault
            label = 'f/16'
            D = f1/N
            aperA = pi*(D/2)^2;
        case 6
            N = 32
            f1 = 50e-3 
            f2 = 50e-3
            z0 = zdefault
            label = 'f/32'
            D = f1/N
            aperA = pi*(D/2)^2;
        case 7
            N = 1.4
            f1 = 50e-3 
            f2 = 50e-3
            z0 = [zmax/5, 3*zmax/4]
            label = 'mirrorF1.4'
            D = f1/N
            aperA = pi*(D/2)^2;
        case 8
            N = 50/11 %%% each lenslet: 50mm focal, 11mm diameter
            f1 = 50e-3 
            f2 = 50e-3
%             z0 = [maxz/6, 3*maxz/6, 5*maxz/6]
            z0 = [0, 1*zmax/3, 2*zmax/3, 3*zmax/3]
            label = 'lenslet4'
            aperA = 1*(11e-3)^2 %%% This assumes you only use information from one image for each neuron. 
                                %%% This also assumes that the light
                                %%% throughput of each lenslet is
                                %%% equivalent to that of a centrally
                                %%% positioned lenslet. This is
                                %%% unfortunately not the case, there is
                                %%% vignetting across the aperture. 
        case 9
            N = 2
            f1 = 50e-3 
            f2 = 50e-3
            z0 = [zmax/5, 3*zmax/4]
            label = 'beamsplitterF2'
            D = f1/N
            aperA = (pi*(D/2)^2)/2; %%% Each image has half the total light
        case 10
            D = 16e-3
            f1 = 50e-3 
            N = f1/D 
            z0 = zdefault;
            label = 'optotune16'
            aperA = pi*(D/2)^2;
        case 11
            D = 10e-3
            f1 = 50e-3 
            N = f1/D 
            z0 = zdefault;
            label = 'optotune10'
            aperA = pi*(D/2)^2;
        case 12
            D = 30e-3
            f1 = 50e-3 
            N = f1/D 
            z0 = zdefault;
            label = 'optotune30'
            aperA = pi*(D/2)^2;
        case 13
            N = 4
            f1 = 50e-3 
            f2 = 50e-3
            z0 = [zmax/5, 3*zmax/4]
            label = 'beamsplitterF4'
            D = f1/N
            aperA = (pi*(D/2)^2)/2;
        case 14
            N = 4
            f1 = 50e-3 
            f2 = 50e-3
            z0 = [zmax/5, 3*zmax/4]
            label = 'lenslet12.5mm'
            D = f1/N
            aperA = (pi*(D/2)^2);
        case 15
            N = 2
            f1 = 50e-3 
            f2 = 50e-3
            z0 = [zmax/5, 3*zmax/4]
            label = 'mirrorF2'
            D = f1/N
            aperA = pi*(D/2)^2;
        case 16
            N = 3.333
            f1 = 50e-3 
            f2 = 50e-3
            z0 = [zmax/5, 3*zmax/4]
            label = 'lenslet15mm'
            D = f1/N
            aperA = (pi*(D/2)^2);
        case 17
            N = 2
            f1 = 50e-3 
            f2 = 50e-3
%             z0 = [zmax/5, 3*zmax/4]
%             z0 = [zmax-300e-6, zmax]
            z0 = [zmax-500e-6, zmax]
            label = 'beamsplitterF2'
            D = f1/N
            aperA = (pi*(D/2)^2)/2;%%% Each image has half the total light
        case 18
%             %% Original simulation
%             N = 2.3
%             f1 = 50e-3 
%             f2 = 50e-3
% %             z0 = [zmax/5, 3*zmax/4]
%             z0 = [zmax-500e-6, zmax]
%             label = 'lenslet25mm'
%             D = f1/N
%             aperA = (pi*(D/2)^2);
            f1 = 40e-3 
            f2 = 40e-3
%             z0 = [zmax/5, 3*zmax/4]
            z0 = [zmax-500e-6, zmax]
            label = 'lenslet25mmEquiv'
            D = 21.9e-3  % Diameter of circular lens with same area as 25mm lens with 7mm milled off. 
            N = f1/D
            aperA = (pi*(D/2)^2);
        case 19
            N = 1.4
            f1 = 50e-3 
            f2 = 50e-3
            z0 = [zmax/5, 3*zmax/4]
            label = 'beamsplitterF1.4'
            D = f1/N
            aperA = (pi*(D/2)^2)/2;
    end 
    
    
    theta_image = atan(1/(2*N)); %%% since N = f/D, or 2*f/r
    rayleigh_res = .61*lambda*2*N;
    
    if whichCase == 10 || whichCase == 11 || whichCase == 12
        %%% Blur for tunable lens.
        %%% Set blur radius at all depths to be the mean blur radius across the
        %%% cone. We compute this as 1/(b-a)*\int^b_a r; 
        %%% where r = z*tan(theta_image), and b = zmax-z, and a = zmin-z
        
        if doWindowZ
            zmin = min(zvals(:));
        else
            zmin = -max(zvals(:));
        end
        
%         blur = abs(1/(2*(zmax - zmin)) .* ((zmax-zvals).^2*tan(theta_image)  - (zmin-zvals).^2*tan(theta_image)))
%         blur = abs((zmax - zvals))*tan(theta_image) + abs((zmin- zvals))*tan(theta_image)
%         f = @(x) abs(x).*x/2
%         blur = 1/(2.*(zmax - zmin)).*(f(zmax - zvals) - f(zmin - zvals))

        cc = tan(theta_image);  
        blur = 1/(2*(zmax - zmin)).*cc.*((zmax - zvals).^2.*sign(cc.*(zmax-zvals)) - (zmin-zvals).^2.*sign(cc.*(zmin-zvals)))
    else
        nearestz = zvals;
        for kk = 1:numel(zvals)
            nearestz(kk) = min(abs(zvals(kk) - z0));
        end
        blur = nearestz*tan(theta_image); %%% This is the radius of the blur, since theta_image is the half angle
    end
    
    diffraction_blurs(ind) = rayleigh_res;
    blurs(ind, :) = blur; 
    
    %%% Right now ignore any vignetting.
    light_collections(ind) = aperA;
    
    labels{ind} = label
end

%% Plot blur and light transmission

%%%% Plot defocus blur as a function of lateral position (i.e. should see
%%%% multiple peaks for the lenslet and mirror options). 
figure('Position', [100, 100, 1500, 1000])
cc = hsv(size(blurs, 1))
for i = 1:size(blurs, 1)
    plot(xvals*1e3, (blurs(i, :) + diffraction_blurs(i))*1e3, 'LineWidth', 1, 'Color', cc(i, :)), hold on
end
ylabel('Defocus blur [mm]', 'FontSize', fsize)
if doWindowZ
    xlabel('Lateral position [mm]', 'FontSize', fsize)
    title('Defocus (with diffraction) vs. lateral position', 'FontSize', fsize)
else
    xlabel('Axial position [mm]', 'FontSize', fsize)
    title('Defocus (with diffraction) vs. axial position', 'FontSize', fsize)
end
set(gca,'FontSize',fsize);
% xlim([0, zmax*1e3])
% ylim([min(blurs(:)*1e3), max(blurs(:)*1e3)*1.1])
ylim([0, 0.5])
yticks([0:.1:.5])
xticks([-5:5])
legend(labels, 'Location', 'NorthWest', 'FontSize', fsize, 'Orientation', 'Horizontal')
if numel(suffix)>0
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 2.45, 1.75], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    export_fig(fullfile(saveDir, ['blurs_with_diffraction', suffix]))
end



%%%% Overall light collection.
figure('Position', [100, 100, 2000, 1000])
bar(light_collections/max(light_collections(:)))
set(gca,'xticklabel',labels)
title('Light collection efficiency', 'FontSize', fsize)
set(gca,'FontSize',fsize);
set(gca, 'XTickLabelRotation', 45)
yticks([0:.1:1])
if numel(suffix)>0
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 1.9, 2.36], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    print('-depsc', fullfile(saveDir, ['light_collection.eps']))
%     export_fig(fullfile(saveDir, ['light_collection', suffix]))
end

%%
%%%% Relative photons per pixel as a function of position for a given neuron emitting at a certain brightness. 
%%%% (for a defocus blur, the # of pixels is the area of the blur divided by pixel area. (6.5um)^2

% pixel_size = 2*6.5e-6; %%% [um]
% emission_signal = (3.1e4-2.7e4)*25; %%% I averaged the max deltaF across 25 pixels around a neuron and then multiplied by 25
% background_photons = 2.7e4/2; %%% Taken from the above example neuron with f1.2, 50mW illumination
% max_emission_photons = emission_signal/2; %%% Orca conversion factor is ~0.46 electrons/photon

pixel_size = 11e-6; %%% [um]
emission_signal = 12e4; %%% ----> This is the TOTAL SIGNAL based on CNMF-e for f/1.4 macroscope, in focus (from 20171230). 
background_signal = 9000;  %%% ----> This is the TOTAL RECOVERED BACKGROUND based on CNMF-e for f/1.4 macroscope, in focus (from 20171230). 
% electron_photon_conversion = 0.86; %%% From Prime95b?
QE = 0.93 %% Quantum efficiency of Prime95B in green wavelengths
electron_photon_conversion = 1/QE; %%% Based on quantum efficiency in green wavelengths


max_emission_photons = emission_signal*electron_photon_conversion; %%% Orca conversion factor is ~0.46 electrons/photon
background_photons = background_signal*electron_photon_conversion;

nx = size(blurs,2);
blur_areas = pi.*( blurs + repmat(diffraction_blurs, 1, nx) ).^2;
light_collections_norm = light_collections/max(light_collections(:));
pixel_area = pixel_size.^2;

%%% This was the original, but confusing way to compute photons per pixel.
% photons_per_m2 = max_emission_photons.*repmat(light_collections_norm, 1, nx) ./ max(blur_areas,pixel_area); %%% Clip at the size of a pixel (doesn't matter if point is smaller).
% % photons_per_m2 = max_emission_photons.*repmat(light_collections_norm, 1, nx) ./ blur_areas;
% photons_per_pixel = photons_per_m2 * pixel_area;

blur_pixel_count = ceil(blur_areas/pixel_area); %%% Pixels are quantized. 

total_photons = max_emission_photons.*repmat(light_collections_norm, 1, nx);
photons_per_pixel = total_photons./blur_pixel_count;

background_photons_per_aperture = background_photons*light_collections_norm;
background_photons_per_location = repmat(background_photons_per_aperture, 1, nx);
noise_var = background_photons_per_location + photons_per_pixel; %%% Assume background noise is constant across voxels (in contrast to emission which comes from one voxel)


doUseGaussianPSF = true

if ~doUseGaussianPSF
    %%% Uses a uniform PSF
    % snr = photons_per_pixel ./ sqrt(noise_var);   %%%% ---> This is what you were doing originally. 
                                                    %%%% This is not equivalent to the derived inversion. 
                                                    %%%% ---> It would actually be: total signal/sqrt(pixel_count * noise_std)   
    snr = total_photons ./ sqrt(blur_pixel_count.*background_photons_per_location + total_photons); %%% This assumes a uniform distribution of photons in the forward model
else
%     figure
    snr = zeros(size(total_photons));
    for design = 1:size(total_photons,1) %%% Loop through each design
        for position = 1:size(total_photons, 2)
            num_pixels = blur_pixel_count(design, position); 
            gaussian_std = num_pixels/2; %%% Since 95 percent of data is within 2 standard deviations of the mean of a gaussian.

            pixel_sidelength = ceil(sqrt(num_pixels));
            xx = -ceil(5000):ceil(5000); %%% Should be in units of pixels?
            xx = xx';
            a = exp(-(xx.^2)./(2*gaussian_std.^2));
            
%             a(abs(xx) > gaussian_std) = 0;
%             a(abs(xx) < gaussian_std) = 1;
%             a = ones(num_pixels,1)/num_pixels;
            
            a = a/sum(a(:));
            hold on
            if mod(position, 100) == 0
%                 plot(a)
%                 pause(0.1)
            end
            B = background_photons_per_location(design, position);
            b0 = B*ones(size(a));
            c0 = total_photons(design, position); %%% This already incorporates alpha?
        
            adag = inv(a'*a)*a';

            snr(design, position) = c0 / sqrt(sum(adag.^2*(a*c0 + b0)));            
        end
    end
end

%%

figure('Position', [100, 100, 1200, 400])
cc = hsv(size(blurs, 1)+10)
cc = cc(round(linspace(1,size(cc, 1), size(blurs,1))),:);
bar(1:size(photons_per_pixel, 1), median(photons_per_pixel, 2))
% errorbar(1:size(photons_per_pixel, 1),std(photons_per_pixel,[], 2),'.')
%  boxplot(photons_per_pixel', 'symbol', '')
%   boxplot(photons_per_pixel')

% ylim([0, 3e5])
set(gca,'xticklabel',labels, 'FontSize', 10)
ylabel('Median number of signal photons per pixel', 'FontSize', fsize)
title('Median (across window) of signal photons per pixel')
set(gca,'FontSize',fsize);
set(gca, 'XTickLabelRotation', 45)
yticks([0:250:1500])
if numel(suffix)>0
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 1.9, 2.36], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    print('-depsc', fullfile(saveDir, ['median_photons_per_pixel.eps']))
end



figure('Position', [100, 100, 1200, 400])
cc = hsv(size(blurs, 1)+10)
cc = cc(round(linspace(1,size(cc, 1), size(blurs,1))),:);
bar(1:size(snr, 1), median(snr, 2))
% boxplot(log10(snr'), 'symbol', '')

set(gca,'xticklabel',labels, 'FontSize', 10)
ylabel('Median SNR per pixel', 'FontSize', fsize)
title('Median (across window) of SNR per pixel')
set(gca,'FontSize',fsize);
set(gca, 'XTickLabelRotation', 45)
yticks([0:10:100])
if numel(suffix)>0
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 1.9, 2.36], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    print('-depsc', fullfile(saveDir, ['median_snr_per_pixel.eps']))
end

figure('Position', [100, 100, 1200, 400])
cc = hsv(size(blurs, 1)+10)
cc = cc(round(linspace(1,size(cc, 1), size(blurs,1))),:);
bar(1:size(snr, 1),mean(snr, 2))

set(gca,'xticklabel',labels, 'FontSize', 10)
ylabel(['SNR'], 'FontSize', fsize)
title('Mean SNR across window')

set(gca,'FontSize',fsize);
set(gca, 'XTickLabelRotation', 45)
if numel(suffix)>0
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 1.9, 2.36], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    print('-depsc', fullfile(saveDir, ['mean_snr_across_window.eps']))
end



%%% Plot # of signal photons per pixel lateral position (incorporates blur and number of photons)
figure('Position', [100, 100, 1500, 1000])
cc = hsv(size(blurs, 1))

for i = 1:size(blurs, 1)
%            plot(xvals*1e3, photons_per_pixel(i, :), 'LineWidth', 3, 'Color', cc(i, :)), hold on
      plot(xvals*1e3, log10(photons_per_pixel(i, :)), 'LineWidth', 1, 'Color', cc(i, :)), hold on
%     plot(xvals*1e3, blur_areas(i, :)*(1e3).^2, 'LineWidth', 3, 'Color', cc(i, :)), hold on
end
ylabel('log_{10}(photons per pixel)', 'FontSize', fsize)
if doWindowZ
    xlabel('Lateral position [mm]', 'FontSize', fsize)
else
    xlabel('Axial position [mm]', 'FontSize', fsize)
end
legend(labels, 'Location', 'NorthWest', 'FontSize', fsize, 'Orientation', 'Horizontal')

set(gca,'FontSize',fsize);
title('Photons per pixel')
if numel(suffix)>0
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 2.45, 1.75], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    export_fig(fullfile(saveDir, ['photons_per_pixel', suffix]))
end



%%% Plot SNR as a function of lateral position (incorporates blur and number of photons)
figure('Position', [100, 100, 1500, 1000])
% cc = hsv(size(blurs, 1)+10)
% cc = cc(round(linspace(1,size(cc, 1), size(blurs,1))),:);
cc = hsv(size(blurs, 1))

for i = 1:size(blurs, 1)
      plot(xvals*1e3, log10(snr(i, :)), 'LineWidth', 1, 'Color', cc(i, :)), hold on
end
ylabel('log_{10}(snr)', 'FontSize', fsize)
if doWindowZ
    xlabel('Lateral position [mm]', 'FontSize', fsize)
else
    xlabel('Axial position [mm]', 'FontSize', fsize)
end
yl = ylim;
ylim([yl(1), 1.1*yl(end)])
set(gca,'FontSize',fsize);
legend(labels, 'Location', 'NorthWest', 'FontSize', fsize, 'Orientation', 'Horizontal')
title('SNR relative to background shot noise')
xticks([-5:5])
yticks([1:0.2:2.8])
if numel(suffix)>0
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 2.45, 1.75], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
    export_fig(fullfile(saveDir, ['snr_relative_to_background', suffix]))
end


%% Resolution


%%%% 1.22*lambda/NA
%%%% where NA = 1/(2*N)
%%%% So: rayleigh distance = 1.22*lambda*2*N

doPlotRes = false

if doPlotRes
    N = linspace(1.2, 16, 30);

    diffraction_res = .61*lambda*2*N;
    figure, plot(N, diffraction_res*1e6, '-o', 'LineWidth', 4 )
    ylabel('Rayleigh criterion [um]', 'FontSize', fsize)
    xlabel('F-number', 'FontSize', fsize)
    set(gca, 'FontSize', fsize)
    xlim([N(1), N(end)])
    title('Resolution limit (at focal plane)')
end



%% Also think about the actual number of pixels for each image  --- how does this matter?
%%% (i.e. it will be slightly less for the lenslet)

%% Notes
%%%% Maybe the measure that we want is: in-focus light collection as a
%%%% function of lateral position. Have a defocus blur threshold?

            

%%%% Plot depth of field as a function of aperture size
%%%% Here, we use wikipedia's formula