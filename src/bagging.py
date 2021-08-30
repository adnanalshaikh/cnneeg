from data_loader import DataLoader
import numpy as np
from saver import Saver 
from trainer import Trainer
try:
	from sklearn.model_selection import StratifiedGroupKFold
except: 
	strat_gkfole = False 
	pass

import pprint
from sklearn.utils import shuffle 
from sklearn.utils import resample
import os 
import time 
from collections import Counter
from tensorflow.keras.models import load_model
from measures import Measures 

from numpy import genfromtxt


class Bagging(object):
	def __init__(self, config):
		self.saver       = Saver(config)
		self.dataloader  = DataLoader(config) 
		self.trainer     = Trainer(config, self.saver)
		self.config      = config
		self.progress    = 0
		self.bg_exp      = None
    
	def generate_exp(self, nbags = None):
		# split and save data for evaluation
		
		nbags = self.config.get("num_bags", 10)
		
		bags = {}
		
		ids, test_ids  = self.dataloader.train_test_split_ids( )
		bags['test']  = test_ids
		bags['train'] = ids
		
		for i in range(1, nbags+1):
			train_ids = resample(ids, replace=True, n_samples=len(ids))
			oob_ids   = ids[np.logical_not(np.isin(ids, train_ids))]
			
			bags[i] = {'train' : train_ids, 'oob' : oob_ids}
			print ("len-train-ids: ", len(train_ids))
			print ("len-oob-ids: ", len(oob_ids))
			
			print ("ulen-train-ids: ", len(np.unique(train_ids)))
			print ("ulen-oob-ids: ", len(np.unique(oob_ids)))

			#print ("len-train-ids: ", train_ids)
			#print ("len-oob-ids: ", oob_ids)
		
		bags['completed'] = 0
		bags['max_bags']   = nbags
		
		self.bg_exp = bags
		self.saver.save_bg_exp(bags, overwrite = False )
		
		return bags
					
	def evaluate ( self ):
		start = self.bg_exp['completed'] + 1
		end = self.bg_exp['max_bags'] + 1
		
		if start >= end : 
			print ("Bagging already finished -- try another exp")
			return 
			
		for x in range(start, end):
			print (f"========== processing bag: {x} ")
			
			x_train, y_train, x_valid, y_valid = self.get_bg_data(x)
			x_train, y_train = shuffle(x_train, y_train, random_state = self.config["random_state"])
				
			model = self.trainer.train( x_train, y_train, valid_data = (x_valid, y_valid ), id = x )
			self.bg_exp['completed'] += 1
			self.saver.save_bg_exp(self.bg_exp, overwrite = True)
		
		self.evaluate_ensemble()

			
	def evaluate_ensemble(self):
		
		models_paths = self.saver.models_path
		model_names = [f for f in os.listdir(models_paths) if os.path.isfile(os.path.join(models_paths, f)) and f.endswith('.h5')]
		models      = [load_model(os.path.join(models_paths, model)) for model in model_names]

		# hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
		#weights = np.loadtxt(os.path.join(models_paths,'model_weights.txt'))
		#print (weights)
		#weights = weights / np.sum(weights)
		#print (weights)

		print ("==========================")
		pprint.pprint (model_names)

		x_test, y_test = self.get_bg_test_data()
		y_pred = np.zeros((x_test.shape[0], 3));
		m = 1
		
		for model in models:
			print("processing model: ", m)
			#y_pred = y_pred + weights[m-1] * model.predict(x_test)	
			my_pred = model.predict(x_test) 
			eval = Evaluation(y_test, my_pred)
			m_test_measure = eval.eeg_metrics()
			pprint.pprint(f"accuracy [{m}] = ", m_test_measure["accuracy "])
		
			y_pred = y_pred +  my_pred	
			m += 1
		
		y_pred = y_pred / len(models)
		eval = Evaluation(y_test, y_pred)
		final_test_measure = eval.eeg_metrics()
		pprint.pprint(final_test_measure)

		
	def bg_resume(self):
		self.bg_exp = self.saver.load_bg_exp()
		self.evaluate()
		
		
		#tic = time.perf_counter()
		#toc = time.perf_counter()
		#print ("Time = ", (toc - tic)*1000)
		
	def get_bg_data(self, bag):
		train_ids, valid_ids = self.bg_exp[bag]['train'], self.bg_exp[bag]['oob']		

		train_f              = self.dataloader.samples_from_ids_dup( train_ids )	
		valid_f              = self.dataloader.samples_from_ids( valid_ids  )
		x_train, y_train     = self.dataloader.ml_prepare(train_f)
		x_valid, y_valid     = self.dataloader.ml_prepare(valid_f)
		
		return x_train, y_train, x_valid, y_valid
		
	def get_bg_test_data(self):
		if not self.bg_exp:
			self.bg_exp = self.saver.load_bg_exp()
			
		test_ids  = self.bg_exp['test']	
		test_f              = self.dataloader.samples_from_ids_dup( test_ids )	
		x_test, y_test     = self.dataloader.ml_prepare(test_f)

		return x_test, y_test	