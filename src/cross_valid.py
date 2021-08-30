from data_loader import DataLoader
import numpy as np
from saver import Saver 
from trainer import Trainer

# StratifiedGroupKFold only available from the nightly repo 
try:
    from sklearn.model_selection import StratifiedGroupKFold
    strat_gkfold = True 
except: 
    strat_gkfold = False 
    pass

from sklearn.utils import shuffle 

import os 

from measures import Measures
import pprint

class CrossValidation(object):
    def __init__(self, config):
        
        self.saver       = Saver(config)
        self.dataloader  = DataLoader(config) 
        self.trainer     = Trainer(config, self.saver)
        self.config      = config
        self.progress    = 0
        self.cv_exp      = None
        
        resume_exp = config.get("resume_cv", 1)
        if resume_exp: self.cv_resume()
        else : self.generate_exp()
        
                
    def generate_exp(self, sklearn_strat = None ):
        sklearn_strat = sklearn_strat or bool(self.config.get("use_skstrat", 0 ))         
        self.skstrat_gcv_generator() if strat_gkfold and sklearn_strat else self.strat_gcv_generator()
        self.evaluate()
        
    def cv_resume(self):
        self.cv_exp = self.saver.load_cv_exp()
        self.evaluate()
        
    def write_cv_files (self):
        for i in range(self.cv_exp['progress']+1, self.cv_exp['num_folds']+1):
            x_train, y_train, x_valid, y_valid = self.get_cv_data(i)
            
            filename = os.path.join(self.saver.cv_path, f"fold_x_train_{i}.npy")
            np.save(filename, x_train)
            
            filename = os.path.join(self.saver.cv_path, f"fold_y_train_{i}.npy")
            np.save(filename, y_train)
            
            filename = os.path.join(self.saver.cv_path, f"fold_x_valid_{i}.npy")
            np.save(filename, x_valid)
            
            filename = os.path.join(self.saver.cv_path, f"fold_y_valid_{i}.npy")
            np.save(filename, y_valid)
        
        del self.dataloader.train 
        del self.dataloader.test
        self.dataloader.train = None
        self.dataloader.test  = None
        
    def skstrat_gcv_generator(self):
        
        no_folds = self.config.get("number_folds", 10)
        cv_exp = {}
        
        print ("Cross validation using: StratifiedGroupKFoldnhouse ")
        print (f"Number folds  =  {no_folds} ")
        
        
        x = self.dataloader.train
        groups = x[:, -2] * 1000 + x[:, -1]
        y = x[:, -2]

        cv = StratifiedGroupKFold(n_splits = no_folds, shuffle = True)
        k = 1
        
        for train_idxs, valid_idxs in cv.split(x, y, groups):    
            train_ids, valid_ids = np.unique(groups[train_idxs]), np.unique(groups[valid_idxs])
            
            np.random.shuffle(train_ids)
            np.random.shuffle(valid_ids)
            
            cv_exp[k] = {'train' : train_ids, 'valid': valid_ids }
            k = k + 1
        
        cv_exp['progress'] = self.progress 
        cv_exp['num_folds'] = no_folds
        
        self.cv_exp = cv_exp    
        self.saver.save_cv_exp(cv_exp, overwrite = False)
    
    
    def strat_gcv_generator(self):    
        
        np.random.seed(self.config["random_state"])
        no_folds = self.config.get("number_folds", 10)
        
        print ("Cross validation using: inhouse ")
        print (f"Number folds  =  {no_folds} ")
        
        x = self.dataloader.train
        classes = np.unique(x[:, -2])
        
        folds_train = {}
        folds_test  = {}
        cv_exp = {}
            
        for i in range(1, no_folds + 1):
            folds_train[i] = np.array([])
            folds_test[i]  = np.array([])

        for j in classes:
            ids = np.unique(x[x[:, -2] == j, -1])
            np.random.shuffle(ids)            
            splits = np.array_split(ids, no_folds)
            for i in range(1, no_folds+1):
                folds_test[i]   =     np.concatenate((folds_test[i], j * 1000 + splits[i-1]))    
                if i > 1:
                    folds_train[i] = np.concatenate ( (folds_train[i], 1000 * j + np.concatenate(splits[:i-1]) ))
                if i < no_folds:
                    folds_train[i] = np.concatenate ( (folds_train[i], 1000 * j + np.concatenate(splits[i:]) ))
                                    
                            
        for k in range (1, no_folds + 1):
            np.random.shuffle(folds_train[k])
            np.random.shuffle(folds_test[k])
            
            cv_exp[k]  = {'train' : folds_train[k], 'valid' : folds_test[k] }    
                
        cv_exp['progress'] = self.progress 
        cv_exp['num_folds'] = no_folds
        self.cv_exp = cv_exp
        
        self.saver.save_cv_exp(cv_exp, overwrite = False)
            
    def evaluate ( self ):
        
        start = self.cv_exp['progress'] + 1
        no_fold = self.cv_exp['num_folds']
        
        if start >= no_fold : 
            print ("cv already finished -- try another exp")
            return 
            
        for x in range(start, no_fold+1):
            print (f"========== processing fold: {x} =================")
            
            x_train, y_train, x_valid, y_valid = self.get_cv_data(x)
            x_train, y_train = shuffle(x_train, y_train, random_state = self.config["random_state"])
                
            model = self.trainer.train( x_train, y_train, valid_data = (x_valid, y_valid ), id = x )
            
            self.cv_exp['progress'] += 1
            self.saver.save_cv_exp(self.cv_exp, overwrite = True)
            self.calculate_measures(model, x_valid, y_valid, fold_no = id)
    
    def calculate_measures(self, model, x_test,  y_test, fold_no = 0, valid = 1):
        measures = Measures(model, x_test, y_test)
        m = measures.eval()
        pprint.pprint(m)
        self.saver.save_measures(m, cv_fold = fold_no, valid = valid)
        
                
    def get_cv_data(self, fold):
    
        train_ids, valid_ids = self.cv_exp[fold]['train'], self.cv_exp[fold]['valid']
        train_f              = self.dataloader.samples_from_ids( train_ids ) 
        valid_f              = self.dataloader.samples_from_ids( valid_ids  )
        x_train, y_train     = self.dataloader.ml_prepare(train_f)
        x_valid, y_valid     = self.dataloader.ml_prepare(valid_f)
        
        return x_train, y_train, x_valid, y_valid
        
