import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from cnn_model import CNNModel
from tensorflow.keras.callbacks import LearningRateScheduler
from lr_schedule import SGDRScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import shuffle 
from keras_hist_graph import plot_history

class Trainer(object):
    
    def __init__(self, config, saver):
        self.config = config
        self.saver  = saver 
  
    def callbacks(self, epoch_size, fold_no = 0):
        logdir                = self.saver.logs_file(fold_no)
        self.checkpoints_path = self.saver.checkpoints_file(fold_no)
        
        tensorboard = TensorBoard(log_dir = logdir, histogram_freq = 1, update_freq='epoch', profile_batch=0) 
        checkpoint = ModelCheckpoint(self.checkpoints_path , monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.config["es_patience"]) 

        spe = np.ceil(epoch_size/self.config["batch_size"])
        
        # different schedules for experimentation 
        #lr_sched = self.step_decay_schedule(initial_lr=0.01, decay_factor=0.75, step_size=20)
        schedule = SGDRScheduler(min_lr=1e-6, max_lr=0.005, steps_per_epoch=spe, lr_decay=0.9, cycle_length=5, mult_factor=1.5)
        #red_lr   = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto',  min_delta=0.0001, cooldown=0, min_lr=0.00001)
                                     
        return [checkpoint, tensorboard, es, schedule]


    def step_decay_schedule(self, initial_lr=1e-2, decay_factor=0.75, step_size=10):

        def schedule(epoch):
            new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
            print ("New learning rate: ", new_lr)
            return new_lr 
            
        return LearningRateScheduler(schedule)
        
    
    def train(self, x_train, y_train, valid_ratio = None, valid_data = None, id = 0 ):
        
        x_train, y_train  = shuffle(x_train, y_train)
        
        valid_ratio = valid_ratio or self.config.get("validation_ratio", 0.1)
        model = CNNModel(self.config, x_train[0].shape).model

        if valid_data:
            print ("Trainer: x_valid:", valid_data[0].shape) 
            
            hist = model.fit(x_train, y_train, validation_data = valid_data, \
                             epochs= self.config["num_epochs"], verbose=2,  \
                             batch_size=self.config["batch_size"], \
                             callbacks=self.callbacks(x_train.shape[0], id), shuffle = True )
            
            if self.config.get('plot_hist', 0)        : plot_history(hist, start_epoch = 1)
            
        else:
            hist = model.fit(x_train, y_train, validation_split = valid_ratio, 
                             epochs= self.config["num_epochs"], verbose=2, batch_size=self.config["batch_size"], 
                             callbacks=self.callbacks(x_train.shape[0], id), shuffle = True) 
                
        model.load_weights(self.checkpoints_path)
        if self.config.get('save_hist', 0)        : self.saver.save_hist(hist, id)
        if self.config.get('save_model', 0)       : self.saver.save_model(model, id)
        
        return model
                        
if __name__ == "__main__":
    pass