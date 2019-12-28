# Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

from tensorflow import keras
import numpy as np
import cv2
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, x_col, directory, batch_size, 
        dim=(224, 224), n_channels=3):
        'Initialization'
        self.dataframe = dataframe
        self.x_col = x_col
        self.directory = directory
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.dataframe[self.x_col][k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image_path = os.path.join(self.directory, ID)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.dim)
            X[i,] = image

        return X