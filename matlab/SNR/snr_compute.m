function [Cn, pnr] = snr_compute(Y, neuron, max_frames, doPlot)

if nargin < 2
    max_frames = 1000;
end
if nargin < 3
    doPlot = false
end

% these are reasonable minimum values for high snr data
min_corr = 0.6;     % minimum local correlation for a seeding pixel
min_pnr = 10;       % minimum peak-to-noise ratio for a seeding pixel

T = size(Y,2);
[Cn, pnr] = neuron.correlation_pnr(Y(:, round(linspace(1, T, min(T, max_frames)))));

if doPlot
% show correlation image 
figure('position', [2052, 116, 993, 625]);
subplot(221);
imagesc(Cn, [0, 1]); colorbar;
axis equal off tight;
title('correlation image');

% show peak-to-noise ratio 
subplot(222);
imagesc(pnr,[0,max(pnr(:))*0.98]); colorbar;
axis equal off tight;
title('peak-to-noise ratio');

% show pointwise product of correlation image and peak-to-noise ratio 
subplot(223);
imagesc(Cn.*pnr, [0,max(pnr(:))*0.98]); colorbar;
axis equal off tight;
title('Cn*PNR');

% show pointwise product of correlation image and peak-to-noise ratio 
subplot(224);
imagesc((Cn >= min_corr).*(pnr >= min_pnr)); colorbar;
axis equal off tight;
title('Cn*PNR thresholded');

% plot scatter plot of PNR/Corr
figure('position', [2052, 116, 993, 625]);
plot(pnr(:), Cn(:),'k.');
hold on;
plot([min_pnr min_pnr], ylim);
plot(xlim, [min_corr min_corr]);
xlabel('PNR');
ylabel('Corr');
title('contours of estimated neurons');
end