clearvars
% This code  
% 1) augments EEG signals
% 2) convert augmented subsignals into images for feature extraction 
%    and machine learning 

% Input data
raw_datapath     = 'D:\\research\\cnneeg\\data\\raw\\bonn_data_set\\';
out_datapath     = 'D:\\research\\cnneeg\\data\\';
header_filename  = 'data_64_gen.hdr';

train_conf.out_datapath = out_datapath;
train_conf.imwidth      = 64;              
train_conf.imheight     = train_conf.imwidth;
train_conf.win          = 512;           
train_conf.stride       = 512;             
train_conf.isbalanced     = 0;                % Wether to generate roughly balanced data
                                 % For only the three cases 

% A, B --> normal     C, D --> interictal           E    --> ictal

%%%%%%%%% Generate Training Data   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_conf.data             = augment_eeg(raw_datapath, train_conf);

% We can augment the signals using different strides so we can sample from 
% for testing and/or validation. Or we can sample from the training  

% get test data by sampling from train data 
test_conf.data           = [];
test_conf.out_datapath   = "";
test_conf.imwidth        = 0;              
test_conf.imheight       = train_conf.imwidth;
test_conf.win            = 0;           
test_conf.stride         = 0;             
test_conf.isbalanced     = 0; 


header_path = sprintf("%s%s", out_datapath, header_filename);
write_pydic(header_path, train_conf, test_conf)
