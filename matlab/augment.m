
function [augeeg] = augment(eeg, win, stride)

    num_eegs = size(eeg, 1);
    num_samples = size(eeg, 2) - 2;
    num_win = floor((num_samples-win)/stride)+1;
    
    
    augeeg = zeros(num_win * num_eegs, win+2); % +1 for label

    j = 0;
    for i=1:num_eegs
        a = augment_signal(eeg(i, :), win, stride);
        augeeg(j+1 : j + size(a, 1), :) = a;
        j = j + size(a, 1);
    end
end

