

a1 = imread('/media/Data/data/20171228/v10_thy1gc6f_im1.tif');
a2 = imread('/media/Data/data/20171228/v10_thy1gc6f_im2.tif');


%% Define bounding rectangle that contains only brain
figure, imagesc(a1)
 cc1 = getrect;
 hold on; rectangle('Position', cc1);

cc = round(cc1); 
crop1 = a1(cc(2):cc(2)+cc(4), cc(1):cc(1)+cc(3));

crop1 = double(crop1);

%%
figure, imagesc(a2)
 cc2 = getrect;
 hold on; rectangle('Position', cc2);

cc = round(cc2); 
crop2 = a2(cc(2):cc(2)+cc(4), cc(1):cc(1)+cc(3));

crop2 = double(crop2);
%% Take FFT across vertical dimension
m1 = crop1./repmat(mean(crop1, 1), size(crop1, 1), 1);
m2 = crop2./repmat(mean(crop2, 1), size(crop2, 1), 1);

figure, plot(smooth(max(abs(gradient(m1)), [], 1)))
hold on
plot(smooth(max(abs(gradient(m2)), [], 1), 'r'))



%% Registration
[p1, p2] = cpselect(a1, a2, 'Wait', true);

%%
t= fitgeotrans(p1,p2,'affine');
f_ref = imref2d(size(a1)); %relate intrinsic and world coordinates
m_reg = imwarp(a2,t,'OutputView',f_ref);
figure, imshowpair(m_reg,a1,'blend')

%% Select a dividing line
x1 = 165;
x2 = 545;

padding = 1

c1 = a1;
c1(:,x1+padding:x2-padding) = 0;
c2 = m_reg;
c2(:,1:x1) = 0;
c2(:,x2:end) = 0;

figure, imagesc(c1+c2)


%%%% To do:
%%%% Figure out where the dividing line should be, then just do tile
%%%% alignment across that border??



%%%% To do tomorrow:
%%%% Select ROIs
%%%% Run cnmf on the ROIs