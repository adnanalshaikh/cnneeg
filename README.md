# cnneeg
Classification of EEG signals into three categories: normal, interictal, and ictal, using 2D convolution neural network 

To run the program:
1) Software 
	Python version  3.8.0
	Tensorflow      2.4.1
	sklearn         1.0.dev0 to use stratifiedGroupKfold otherwise last release 
	CUDA            11.0

2) Set the experiments path and the data path   
	"exppath"             : "D:/research/cnneeg/experiments/", 
  	"datapath"            : "D:/research/cnneeg/data/images/",

   either use linux slash (/) style or 
   windows escaped backslash (\\) 

3) Un compress the file in the data directory

4) In the source directory 
   $ python main.py

5) To generate other data sets, use the matlab provided code 



