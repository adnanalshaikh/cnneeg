function eeg = transform_to_images(aug_data, width, height, min, max, augwind)

    figure('visible','off');
    eeg = zeros(size(aug_data, 1), width * height + 2);
    
    for i=1:size(aug_data, 1) 
        plot(1:size(aug_data, 2)-2, aug_data(i, 1:end-2), 'k', 'linewidth', 2);
        axis([0 augwind min max])
        axis off;
    
        F = getframe(gca);
        [x,~] = frame2im(F);
        xgray = rgb2gray(x);
        xgray = imresize(xgray,[width height]); 
        xgray = xgray';
        
        y = [xgray(:); uint8(aug_data(i, end-1)); uint8(aug_data(i, end))];
        eeg(i, :) = y';
        
        if (mod(i, 10) == 0)
            fprintf ('Done transforming %d out of %d \n', i, size(aug_data, 1)  );
        end        
        pause(.001);
    end
    
    figure('visible','on');
end

