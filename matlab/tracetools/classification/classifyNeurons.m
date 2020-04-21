function [goodNeurons, badNeurons, corrs, aspect_ratios] = classifyNeurons(neuron, corr_thresh, aspect_ratio_thresh)
%%% Automatically classify traces as being neuron-like or not (i.e. blood
%%% vessels).
%%% Input: 
%%%     neuron - a Sources2D struct from CNMF-E, containing the results of
%%%     a single imaging session.
%%%     corr_thresh - threshold for correlation between deconvolved trace and raw
%%%     trace (actually, the correlation between parts of the signal that
%%%     are multiple standard deviations above the baseline). 
%%% Output:
%%%     goodNeuron --- indices of good neurons
%%%     badNeuron --- indices of bad neurons

if ~exist('corr_thresh', 'var') || isempty(corr_thresh)
    corr_thresh = 0.8;
end

if ~exist('aspect_ratio_thresh', 'var') || isempty(aspect_ratio_thresh)
    aspect_ratio_thresh = 2;
end


C = neuron.C;
C_raw = neuron.C_raw;
    
numNeurons = size(C, 1); 
keepNeuron = zeros(numNeurons, 1);
corrs = zeros(numNeurons, 1);
aspect_ratios = zeros(numNeurons, 1);

doPlot = false
if doPlot
    figure;
end
for i = 1:numNeurons
    peak_thresh = 2*std(C(i,:)); %%% Look at correlation only during large magnitude events
    inds = find(C(i,:)>peak_thresh);
    if ~isempty(inds)
        cc = corr(C(i,inds)', C_raw(i,inds)');
        corrs(i) = cc;

        a = neuron.reshape(neuron.A(:, i),2);
        aspect_ratio = getAspectRatio(a); 
        aspect_ratios(i) = aspect_ratio;

        if doPlot
            subplot(121), plot(C(i,:)); hold on; plot(C_raw(i,:));
            title([num2str(i), ': ', num2str(cc)]); hold off
            subplot(122), imagesc(a)
            title([num2str(aspect_ratio)]); hold off
            pause(1);
        end

        keepNeuron(i) = cc> corr_thresh && aspect_ratio < aspect_ratio_thresh;
    end
end

goodNeurons = find(keepNeuron);
badNeurons = find(~keepNeuron)


%%% Also look at the spatial shape of the neuron. 
%     [a_fit, s1, s2, rot] = fit2DGaussian(a);
%     x0 =x0(1:5);
%     xin(6) = 0; 
%     x =x(1:5);
%     lb = [0,-MdataSize/2,0,-MdataSize/2,0];
%     ub = [realmax('double'),MdataSize/2,(MdataSize/2)^2,MdataSize/2,(MdataSize/2)^2];
%     [x,resnorm,residual,exitflag] = lsqcurvefit(@D2GaussFunction,x0,xdata,Z,lb,ub);
%     x(6) = 0;
%   

% 
% 
% %%%% From CaImAn
% 
% rval_space = classify_comp_corr(data,A,C,b,f,options);
% ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
%                                         % this test will keep processes
%                                         
% %% further classification with cnn_classifier
% try  % matlab 2017b or later is needed
%     [ind_cnn,value] = cnn_classifier(A,FOV,'cnn_model',options.cnn_thr);
% catch
%     ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
% end     
%                             
% %% event exceptionality
% 
% fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
% ind_exc = (fitness < options.min_fitness);
% 
% %% select components
% 
% keep = (ind_corr | ind_cnn) & ind_exc;
% 
% 
% %% view contour plots of selected and rejected components (optional)
% throw = ~keep;
% Coor_k = [];
% Coor_t = [];
% figure;
%     ax1 = subplot(121); plot_contours(A(:,keep),Cn,options,0,[],Coor_k,1,find(keep)); title('Selected components','fontweight','bold','fontsize',14);
%     ax2 = subplot(122); plot_contours(A(:,throw),Cn,options,0,[],Coor_t,1,find(throw));title('Rejected components','fontweight','bold','fontsize',14);
%     linkaxes([ax1,ax2],'xy')
%     
%     %% keep only the active components    
% A_keep = A(:,keep);
% C_keep = C(keep,:);




%%% Make goodNeuron and badNeuron structs