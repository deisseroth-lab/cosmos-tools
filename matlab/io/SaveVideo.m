function  SaveVideo( v1, fps, savePath, scaleBar, cmap, doColorbar )

if nargin < 4
    scaleBar = [0 max(v1(:))];
end
if nargin < 5
    cmap = gray;
end
if nargin < 6
    doColorbar = false;
end

XX = 512;
YY = 512; 
if size(v1,4) > 1
    [X,Y,T] = size(v1)   
    disp('SaveVideo 4dimensional')
    v = VideoWriter(savePath);
    v.FrameRate = fps;
    v.open();
    f = figure(1);
    set(gcf, 'Position', [100 100 XX YY]);
%     imshow(v1(:,:,:,1));
    axis tight; axis off;    

    for i=1:size(v1,3)
        figure(1);
        if mod(i,10) == 0
            fprintf('%d, ', i);
        end
        if mod(i, 100) == 0
            fprintf('\n');
        end
        imshow(v1(:,:,:,1));
        axis tight; axis off;    
        set(gcf, 'Color', 'w');
    %     set(gcf, 'Position', [100 100 512 512]);
        pause(0.2);
        img = getframe(gcf);      
        [I,map] = frame2im(img);
        fprintf('Size = %d,%d,%d\n',size(I));
        v.writeVideo(img);    

    end
    v.close();
    close(f);
    
else


    [X,Y,T] = size(v1);
    disp('SaveVideo')
    v = VideoWriter(savePath);
    v.FrameRate = fps;
    v.open();
    f = figure(1);
    set(gcf, 'Position', [100 100 XX YY]);
    imagesc(v1(:,:,1)', scaleBar); colormap hot; 
    axis square; axis tight; axis off;    

    disp(savePath)
    for i=1:size(v1,3)
        figure(1);
        if mod(i,10) == 0
            fprintf('%d, ', i);
        end
        if mod(i, 100) == 0
            fprintf('\n');
        end
        imagesc(v1(:,:,i)', scaleBar); colormap(cmap); 
        if doColorbar,
            colorbar
        end
        axis square; axis tight; axis off;    
        set(gca,'LooseInset',get(gca,'TightInset'));
        set(gcf, 'Color', 'w');
    %     set(gcf, 'Position', [100 100 512 512]);
        pause(0.2);
        img = getframe(gcf);     
        if i == 1
            img1 = img;
        end
        [c1, c] = MatchSizes(img1.cdata, img.cdata);
        img.cdata = c;
        [I,map] = frame2im(img);
%         fprintf('Size = %d,%d,%d\n',size(I));
        v.writeVideo(img);    

    end
    v.close();
    close(f);
end
