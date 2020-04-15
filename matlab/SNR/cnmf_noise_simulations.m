%%%% Simulations to explore how the shape of the psf, the total light
%%%% transmission, and the background intensity level affect SNR of the
%%%% reconstructed signal. 

usingMacbookAir = true

if ~usingMacbookAir
    addpath(genpath('/home/izkula/src/COSMOS'))
else
    addpath(genpath('~/src/COSMOS'))
end

set(0,'defaultfigurecolor', 'w')
saveDir = '~/Dropbox/cosmos/SNR_analysis/simulation/'
if ~exist(saveDir); mkdir(saveDir); end

whichCase = 1

suffix = '.png'
fsize = 10


pixel_size = 11e-6; %%% [um]
total_signal = 12e4; %%% ----> This is the TOTAL SIGNAL based on CNMF-e for f/1.4 macroscope, in focus (from 20171230). 
background_photons = 9000; %%% ----> This is the TOTAL RECOVERED BACKGROUND based on CNMF-e for f/1.4 macroscope, in focus (from 20171230). 

switch whichCase 
    case 1
        x = [1:1000];    %%% Independent variable
        SNR = zeros(size(x));
        xstr = '# nonzero pixels'
        for i = 1:numel(x)
            a = ones(x(i),1)/x(i);
            a = a/sum(a(:));
            alpha = 1; 
%             B = 10;
            B = background_photons;
            b0 = B*ones(size(a));
%             c0 = 1;
            c0 = total_signal;
        
            adag = inv(a'*a)*a';
%             SNR(i) = (sqrt(alpha)*c0) / (adag*sqrt(a*c0 + b0));
            SNR(i) = (sqrt(alpha)*c0) / sqrt(sum(adag.^2*(a*c0 + b0)));
        end
        figure, plot(x, log10(SNR))
        xlabel(xstr)
        ylabel('SNR')
        
    case 2
        x = linspace(0, 1, 100);    %%% Independent variable
        SNR = zeros(size(x));
        xstr = 'fraction of total light transmitted'
        for i = 1:numel(x)
            a = ones(1, 1);
            a = a/sum(a(:));
            alpha = x(i); 
            B = 10;
            b0 = B*ones(size(a));
            c0 = 1;
        
            adag = inv(a'*a)*a';
%             SNR(i) = (sqrt(alpha)*c0) / (adag*sqrt(a*c0 + b0));
            SNR(i) = (sqrt(alpha)*c0) / sqrt(sum(adag.^2*(a*c0 + b0)));
        end
        figure, plot(x, SNR)
        xlabel(xstr)
        ylabel('SNR')
        
    case 3
        x = linspace(.1, 1, 100);    %%% Independent variable
        SNR = zeros(size(x));
        xstr = 'width of gaussian psf'
        figure
        for i = 1:numel(x)
            xx = linspace(-5, 5, 1001)';
            a = exp(-(xx.^2)./(2*x(i).^2));
            a = a/sum(a(:));
            hold on
            if mod(i, 10) == 0
                plot(a)
            end
            alpha = x(i); 
            B = 1;
            b0 = B*ones(size(a));
            c0 = 100;
        
            adag = inv(a'*a)*a';
%             SNR(i) = (sqrt(alpha)*c0) / (adag*sqrt(a*c0 + b0));
            SNR(i) = (sqrt(alpha)*c0) / sqrt(sum(adag.^2*(a*c0 + b0)));
        end
        export_fig(fullfile(saveDir, ['gaussian_widths', suffix]))

        figure, plot(x, SNR)
        xlabel(xstr)
        ylabel('SNR')
        set(gca,'FontSize',fsize);
        title('SNR as a function of PSF width')
        export_fig(fullfile(saveDir, ['snr_vs_psf_width', suffix]))
        
        
    case 4
        x = [1, 2];    %%% Independent variable
        SNR = zeros(size(x));
        xstr = ''
        figure
        for i = 1:numel(x)
            %%%% Try putting real numbers in
            
%             xx = linspace(-5, 5, 1001)';
%             a = exp(-(xx.^2)./(2*x(i).^2));
%             a = a/sum(a(:));
%             hold on
%             if mod(i, 10) == 0
%                 plot(a)
%             end
%             alpha = x(i); 
%             B = 10;
%             b0 = B*ones(size(a));
%             c0 = 1;
        
            adag = inv(a'*a)*a';
%             SNR(i) = (sqrt(alpha)*c0) / (adag*sqrt(a*c0 + b0));
            SNR(i) = (sqrt(alpha)*c0) / sqrt(sum(adag.^2*(a*c0 + b0)));
        end
        figure, plot(x, SNR)
        xlabel(xstr)
        ylabel('SNR')
        
end


















%%%% What happens as you add more shot noise to the individual pixels, does
%%%% the reconstruction get worse? 
%%%% If you can show this rigorously, you are golden. It almost must be
%%%% true, potentially only for overlapping neurons, but if so, how. 



%%%% Problem #1:
%%%% You have a single emitter. You measure Y, which is shot-noise modified
%%%% instantiation of the neural signal and a (for now) constant
%%%% background. You are trying a rank-1 factorization. What changes as you
%%%% increase the value of B.

%%%% | Y - AC - B|^2. Where A is the PSF, C is the time series, and B is
%%%% the background. 

%%%% Problem #2:
%%%% Same as above, but now with two overlapping neurons. 


%%%% Problem #3:
%%%% Constant background, but changing the size of the PSF in the forward
%%%% model. 