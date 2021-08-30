import pandas as pd
import os, datetime, warnings, pickle
import json
from tensorflow.keras.models import load_model

warnings.simplefilter('ignore')

class Saver(object):

	def __init__(self, config):
		self.config = config
		self.load_data_attrib()
		
		self.exppath         = os.path.join(config["exppath"], config["expname"])
		self.evaluation_path = os.path.join(self.exppath, "evaluations" )
		self.logs_path       = os.path.join(self.exppath, "logs" )
		self.models_path     = os.path.join(self.exppath, "models" )
		self.createDirs()
		self.save_exp_config(config)

	def createDirs(self):
		try:
			os.makedirs(self.exppath)
			os.makedirs(self.evaluation_path)
			os.makedirs(self.logs_path)
			os.makedirs(self.models_path)
		
		except FileExistsError:
			if self.config["expname"] == "default":
				pass
			else:
				pass
            
	def load_data_attrib (self):
		data_hdr = os.path.join(self.config["datapath"], self.config["data_file_name"])
		with open(data_hdr, 'r') as f: 
			self.data_attrib = json.load(f)
	
	def save_bg_exp(self, bags, overwrite = False ):
		exp_name = self.config["expname"]
		
		bagging_exp = os.path.join(self.bg_path, f'{exp_name}_bagg_exp.pickle')
		print (bagging_exp)
		
		if  os.path.isfile(bagging_exp) and not overwrite:
			raise FileExistsError (bagging_exp)
			
		with open(bagging_exp, 'wb') as f:
			pickle.dump(bags, f)
	
	def load_bg_exp (self):
		exp_name = self.config["expname"]
		bagging_exp = os.path.join(self.bg_path, f"{exp_name}_bagg_exp.pickle")
		
		with open(bagging_exp, 'rb') as f: 
			bags = pickle.load(f)
			
		return bags
			
	def save_exp_config (self, data):
		config_file_name = os.path.join(self.exppath, self.config["expname"] + '.json')
		with open(config_file_name, 'w') as fp:
			json.dump(data, fp, indent = 4)
			
	def logs_file (self, fold_no = 0 ):
		logs = os.path.join(self.logs_path, f'{self.config["expname"]}_fold_{fold_no}_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
		return logs 
	
	def checkpoints_file(self, fold_no = 0):	
		
		checkpoints_path = os.path.join(self.models_path, f"model_weights_{fold_no}.hdf5")
		print (f"checkpoints_file: {checkpoints_path}")
		try:
			os.remove(checkpoints_path)
		except:
			pass
			
		return checkpoints_path
		
	def save_hist(self, hist, fold_no = 0):
		hist_file = os.path.join(self.evaluation_path, f'training_fold_{fold_no}.hist')
		with open(hist_file, 'wb') as f:
			pickle.dump(hist.history, f)

	def save_cv_exp(self, cv_exp, overwrite = False):
		cv_file = os.path.join(self.evaluation_path, 'cv_exp.dict')
		if os.path.isfile(cv_file) and not overwrite:
			raise  FileExistsError(cv_file)
			
		with open(cv_file, 'wb') as f:
			pickle.dump(cv_exp, f)
	
	def load_cv_exp(self):
		cv_file = os.path.join(self.evaluation_path, 'cv_exp.dict')
		try:
			with open(cv_file, 'rb') as f:
				cv_exp = pickle.load(f)
		except:
			raise 
		else :
			return cv_exp
			
	def save_measures (self, final_test_measure, cv_fold = 0, valid = 0):		
		meterics_file_name = os.path.join(self.exppath, 'metrics.csv')
		config = self.config
  
		if "test_filename" in self.data_attrib:
			test_stride = self.data_attrib["test_stride"]
		else:
			test_stride = self.data_attrib["train_stride"]
			
		df = pd.DataFrame([[config["expname"], cv_fold, valid, self.data_attrib["train_num_samples"], self.data_attrib["imdim"], self.data_attrib["train_stride"], \
		test_stride, config["model"], config["learning_rate"], config["batch_size"], config["validation_ratio"], \
		config["dropout_rate"], final_test_measure['loss'], final_test_measure['accuracy'], final_test_measure['precision_macro'], \
		final_test_measure['recall_macro'], final_test_measure['f1Score_macro'], final_test_measure['precision_micro'], \
		final_test_measure['recall_micro'], final_test_measure['f1Score_micro'], final_test_measure['aucs_ovo_macro'], \
		final_test_measure['aucs_ovo_weighted'], final_test_measure['aucs_ovr_macro'], \
		final_test_measure['aucs_ovr_weighted'], final_test_measure['mcm'], final_test_measure['cm']]], \
		
		columns = ["exp_name", "cv_fold", "valid", "num_samples", "imdim", "train_stride", "test_stride", 'model', "learning_rate", "batch_size", \
		"validation_split", "dropout_rate", 'loss', 'accuracy', 'precision_macro', 'recall_macro', 'f1Score_macro', \
		'precision_micro', 'recall_micro','f1Score_micro', 'aucs_ovo_macro', 'aucs_ovo_weighted', 'aucs_ovr_macro',\
		'aucs_ovr_weighted', 'mcm', 'cm'])
	
		if os.path.exists(meterics_file_name ): 
			df.to_csv(meterics_file_name, mode='a', header=False)
		else:
			df.to_csv(meterics_file_name, mode='w', header=True)
	
	def save_model(self, model, id = 0, include_optimizer=False):
		exp_name = self.config["expname"]
		model_file_path = os.path.join(self.models_path, 'model_' + exp_name + f"_fold_{id}_" + '.h5')
		model.save(model_file_path, include_optimizer = include_optimizer)
				
	def load_model(self, id = 0):
		exp_name = self.config["expname"]
		model_file_path = os.path.join(self.models_path, 'model_' + exp_name + f"_fold_{id}" + '.h5')
		return load_model(model_file_path)
		
if __name__ == "__main__":
	from eeg_model import ConvEEGModel
	
	with open('testing.json', 'r') as f:
		config = json.load(f)
	
	cnn_model = ConvEEGModel(config)

	
