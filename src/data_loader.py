import numpy as np
import os
import json
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical

	
class DataLoader(object):
    
	def __init__(self, config, train_test_f = None):
		self.config    = config		
		self.train = self.test = None
		
		self.load_numpy_files()
		
		print (self.train.shape)
		#print (self.test.shape)
		
	def load_numpy_files(self):	

		data_hdr = os.path.join(self.config["datapath"], self.config["data_file_name"])	
		with open(data_hdr, 'r') as f: 
			self.data_attributes = json.load(f)
			self.imdim = self.data_attributes['imdim']
			self.imsize = self.imdim * self.imdim + 2; # class and file id
	
		filename = os.path.splitext(self.config["data_file_name"])[0]	
		numpy_train_file = os.path.join(self.config["datapath"], filename + '_train.npy')
		numpy_test_file = os.path.join(self.config["datapath"], filename + '_test.npy')
		
		print (numpy_train_file)
		try:
			self.train = np.load(numpy_train_file, mmap_mode = 'r')			
		except:
			raise
	
	def load_data (self):
		data_hdr = os.path.join(self.config["datapath"], self.config["data_file_name"])	
		with open(data_hdr, 'r') as f: 
			self.data_attributes = json.load(f)
			self.imdim = self.data_attributes['imdim']
			self.imsize = self.imdim * self.imdim + 2; # class and file id
		
		print ('train file:', self.data_attributes['train_filename'])
		print ('test file: ', self.data_attributes['test_filename'])
		train_filename, ext = os.path.splitext(self.data_attributes['train_filename'])
		
		if ext == '.npz':
			self.load_data_numpy(self.data_attributes['train_filename'])
			
		elif  ext == '.dat':
			self.load_train_data(self.data_attributes['train_filename'])
		
		if self.test is None:
			test_filename, ext = os.path.splitext(self.data_attributes['test_filename'])
			if ext == '.dat':
				self.load_test_data(self.data_attributes['test_filename'])
					
								
	def load_train_data(self, filename):
		try:
			self.train = np.fromfile(filename, dtype='uint8')
			self.train = self.train.reshape(self.imsize , self.data_attributes['train_num_imgs'])
			self.train = self.train.transpose()

		except:
			raise 
	
	def load_test_data (self, filename):
		try:
			self.test = np.fromfile(filename, dtype='uint8')
			self.test = self.test.reshape(self.imsize , self.data_attributes['test_num_imgs'])
			self.test = self.test.transpose()
		except:
			raise 
			
	def write_data_numpy (self):
		filename = os.path.splitext(self.config["data_file_name"])[0]		
		numpy_file_name = os.path.join(self.config["datapath"], filename + '.npz')
		if self.test is not None:
			np.savez_compressed(numpy_file_name, train = self.train, test = self.test )
		else:
			np.savez_compressed(numpy_file_name, train = self.train )

	def load_data_numpy (self, filename):
		
		try:
			data = np.load(filename, mmap_mode = 'r')
			self.train = data['train']
			self.test =  data['test'] if 'test' in data else None
			
		except:
			raise
	

	def train_test_split_from_ids (self, ids = None, test_ratio = None):
		if not test_ratio:
			test_ratio = self.config.get("test_split", 0.10)
		
		if ids is None:
			ids = np.unique(self.train[:, -2]* 1000 + self.train[:, -1])
		
		x = ids % 1000
		y = ids // 1000
		
		if test_ratio >= 1:
			return np.array([]), ids
		strat =  StratifiedShuffleSplit(n_splits=2, test_size=test_ratio)
		for train_index, test_index in strat.split(x, y): 
			train_ids, test_ids = ids[train_index], ids[test_index]
			
		return train_ids, test_ids
			
	def samples_from_ids( self, ids ):
	
		all_ids = self.train[:, -2]* 1000 + self.train[:, -1]
		indices = [i for i, v in enumerate(all_ids) if v in ids]
		samples = self.train[indices]	
		np.random.shuffle(samples)
		return samples

    # sample and allow duplicates -- used in bagging 		
	def samples_from_ids_dup( self, ids ):
		all_ids = self.train[:, -2]* 1000 + self.train[:, -1]
		x = np.array([], dtype = np.int32)
		
		for i in ids:
			q = np.where(all_ids == i)
			x = np.append(x, q)
		print (x)
		
		samples = self.train[x, :]
		
		return samples
		
		
	def ml_prepare (self, data, onehot = True):
		
		num_classes = 3
		
		labels = data[:, -2] - 1 
		labels[labels == 1] = 0;
		labels[labels == 2] = 1;
		labels[labels == 3] = 1;
		labels[labels == 4] = 2;
									
		data = data[:, :-2] 
		data = data.reshape(len(data), self.imdim, self.imdim, 1)

		data = data.astype('float32')
		data = data / 255.0
		
		if onehot:
			labels = to_categorical(labels, num_classes=num_classes )
		
		return (data, labels)
	
	
if __name__ == "__main__" :
	with open('testing.json', 'r') as f: 
		config = json.load(f)
	
	dataloader = DataLoader(config)
	ids = np.array([1023, 2017])
	x = dataloader.samples_from_ids(ids)
	x, y = dataloader.ml_prepare(x)


	
	
