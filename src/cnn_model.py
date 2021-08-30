from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, ELU, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras import activations, regularizers
from tensorflow.keras.models import load_model, save_model
import os
import tensorflow
        
class CNNModel(object):
    
    def __init__(self, config, imshape):
        
        self.config = config

        # no effect if scheduled 
        lr = config.get("learning_rate", 0.001) 
        
        self.imdim     = imshape[0]
        self.dropout   = self.config.get("dropout_rate", 0.5)
        self.opt       = tensorflow.keras.optimizers.Adam(lr=lr)
        self.kinit     = 'he_normal' 
        
        self.model     = self.build_model(config["model"])
   
        
        
    def build_model(self, label):
        
        if label == 1:
            return self.build_model_m1()
        elif label == 2:
            return self.build_model_m2()
        elif label == 3:
            return self.build_model_m3()
        elif label == 4:
            return self.build_model_m4()
        elif label == 5:
            return self.build_model_m5()
            
    def build_model_m1(self):
        # 1 --> 1
        
        print ("Building model M1 ")
        imdim = self.imdim
        kinit = self.kinit 
        drop_rate = self.dropout
        
        model = Sequential()    
        model.add(BatchNormalization(input_shape=(imdim,imdim,1)))
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
    
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
            
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
                
        model.add(Flatten())
        model.add(BatchNormalization())
    
        model.add(Dense(2048))
        model.add(Dropout(drop_rate))
        model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer= self.opt, loss='categorical_crossentropy', metrics=["accuracy"])
        return model

    def build_model_m2(self):
        # 4 --> 2
        
        print ("Building model M2 ")
        imdim = self.imdim
        kinit = self.kinit
        drop_rate = self.dropout
        
        model = Sequential()
        model.add(BatchNormalization(input_shape=(imdim,imdim,1)))
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
    
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
            
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
                
        model.add(Flatten())
        model.add(BatchNormalization())
    
        model.add(Dense(1048))
        model.add(Dropout(drop_rate))
        model.add(BatchNormalization())    
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer = self.opt, loss='categorical_crossentropy', metrics=["accuracy"])
        return model
                

    def build_model_m3(self):
        # 14 --> 3
        print ("Building model M14 ")
        imdim = self.imdim
        kinit = 'he_normal'
        drop_rate = self.config["dropout_rate"]

        model = Sequential()        
        model.add(BatchNormalization(input_shape=(imdim,imdim,1)))
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
    
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
                    
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
            
        model.add(Conv2D(512, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(512, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(512, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Flatten())    
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer = self.opt, loss='categorical_crossentropy', metrics=["accuracy"])
        return model

    def build_model_m4(self):
        # 18 --> 4
        print ("Building model M4 ")
        imdim     = self.imdim
        kinit     = self.kinit
        drop_rate = self.dropout

        model = Sequential()        
        model.add(BatchNormalization(input_shape=(imdim,imdim,1)))
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
    
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
            
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
                
        model.add(Flatten())
        model.add(BatchNormalization())

        model.add(Dense(1048))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))        
        model.add(Dropout(drop_rate))

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))        
        
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer= self.opt, loss='categorical_crossentropy', metrics=["accuracy"])
        return model

    def build_model_m5(self):
        # 20 --> 5
        print ("Building model M5 ")
        imdim = self.imdim
        kinit = self.kinit
        drop_rate = self.dropout

        model = Sequential()        
        model.add(BatchNormalization(input_shape=(imdim,imdim,1)))
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
    
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
            
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(Dropout(rate=drop_rate))
        
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer=kinit))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))
        model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
        model.add(Dropout(rate=drop_rate))
                
        model.add(Flatten())
        model.add(BatchNormalization())

        model.add(Dense(1048))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))        
        model.add(Dropout(drop_rate))

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))        
        model.add(Dropout(drop_rate))
                
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation(activations.relu))    
        
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer = self.opt, loss='categorical_crossentropy', metrics=["accuracy"])
        return model
                
    def eeg_save_model(self, filename = "model.h5"):

        if self.model is None:
            raise Exception("You have to build the model first.")
                
        filename_path = self.checkpoint_path + '/' + filename
        self.model.save(filename_path)

    def eeg_load_model(self, filename = "model.h5"):
        
        filename_path = self.checkpoint_path + '/' + filename 
        self.model = load_model(filename_path)

