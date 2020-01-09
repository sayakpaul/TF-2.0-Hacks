# Imports
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from wandb.keras import WandbCallback
import tensorflow as tf
import numpy as np
import wandb
import time

# Fix the random generator seeds for better reproducibility
tf.random.set_seed(67)
np.random.seed(67)

# Load the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Scale the pixel values of the images to 
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the pixel values so that they are compatible with
# the conv layers
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Specify the labels of FashionMNIST dataset, it would
# be needed later 😉
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# Prepare data tuples
(X_train, y_train) = train_images, train_labels
(X_test, y_test) = test_images, test_labels

# Default values for hyper-parameters we're going to sweep over
configs = {
    'layers': 128,
    'batch_size': 64,
    'epochs': 5
}

# Initilize a new wandb run
wandb.init(project='hyperparameter-sweeps-comparison', config=configs)

# Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config

# Add the config items to wandb (make sure you have logged wandb in)
if wandb.run:
    wandb.config.update({k: v for k, v in configs.items() if k not in dict(wandb.config.user_items())})
                       
# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3, 3), activation='relu'),
    GlobalAveragePooling2D(),
    Dense(config.layers, activation=tf.nn.relu),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, 
         epochs=config.epochs,
         batch_size=config.batch_size,
         validation_data=(X_test, y_test),
         callbacks=[WandbCallback(data_type="image", 
            validation_data=(X_test, y_test), labels=labels)])
