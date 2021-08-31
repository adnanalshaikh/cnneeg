% given eeg (d) signals generate sub-signals using 
% wind: windows size in samples
% stride: stride of moving the windows in sample

% input: eeg (d) - matrix of size N x 1 where N is the is number of samples 
%                  in the signal + 1 for the label 

% return a matrix of sub-samples 

function augdata = augment_signal(d, wind, stride)
    maxrows = fix((size(d, 2)-1)/wind);
    dat = zeros(maxrows, wind+2);

    j = 0;
    class = d(1, end-1);
    file  = d(1, end);
    d(:, end) = [];
    
    for i=1:stride:length(d)-1
        wins = i;
        winf = i+wind-1;
        if (winf > length(d)-1) 
            break; 
        end
        dd = d(wins:winf);
        j = j+1;
        dat(j,:) = [dd class file];
        %dat(j, end) = class; 
        %fprintf('[%d : %d ]  ==>  %d\n', i, i+wind-1, length(dd));
    end
    augdata = dat(1:j, :);
end

