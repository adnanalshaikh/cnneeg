function [signals_imgs] = augment_eeg(raw_datapath, config)

win       = config.win ;
stride    = config.stride;
imheight  = config.imheight;  
imwidth   = config.imwidth;
balanced  = config.isbalanced;

[signals, gmin, gmax] = load_signals(raw_datapath);

%%%%%%%%%%%%% Augment the data %%%%%%%%%%%%%%%%%
fprintf('Augmenting signals .... \n');
if balanced
    
    signals_e = signals(signals(:, end-1) == 5, :);
    signals(signals(:, end-1) == 5, :) = [];
    
    aug_signals   = augment(signals, win, stride);
    aug_signals_e = augment(signals_e, win, stride/2);
    aug_signals = vertcat(aug_signals, aug_signals_e);
    
else
    aug_signals = augment(signals, win, stride);
end

%%%%%%%%%%% transform to images %%%%%%%%%%%%%%%%%%
signals_imgs = transform_to_images(aug_signals, imwidth, imheight, gmin, gmax, win);
fprintf('done\n ');


end
