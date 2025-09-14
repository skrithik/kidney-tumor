import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path#Loads the updated VGG16 model (prepared in Stage 2)
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,#Rescales pixel values (0–255 → 0–1)
            validation_split=0.20#Splits dataset into training (80%) and validation (20%).
        )

        dataflow_kwargs = dict( #Defines how images are resized & batched
            target_size=self.config.params_image_size[:-1], ## (224,224)
            batch_size=self.config.params_batch_size,## 16
            interpolation="bilinear"
        )
#When resizing images from their original size to the target size (224,224), TensorFlow/Keras needs a method to estimate pixel values for the new size.
#That’s what interpolation does:It decides how new pixels are computed when scaling up or down.
#Bilinear: Uses weighted average of 4 nearest pixels → smoother results.This gives smooth transitions, avoiding jagged edges.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(#Image generator for validation data.
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        #Creates validation generator from folder structure
        #Does not shuffle (keeps labels aligned)

        if self.config.params_is_augmentation:#If augmentation enabled → Adds random transformations (rotation, flips, zoom).
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator#Otherwise, same as validation preprocessing.

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        #Creates training generator.
        #Data is shuffled each epoch.

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,#Trains the model for specified epochs
            steps_per_epoch=self.steps_per_epoch,#Uses mini-batch gradient descent
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )
#during an epoch:
#The model goes through all training batches (steps_per_epoch).
#After finishing → It immediately runs through the validation set:
#Calculates validation loss.
#Calculates validation accuracy .(this project can be optimized using techniques like early stopping as well)
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )


#process of training:
#1,mini batch creation:At each step, the generator picks 16 images from the dataset.
#2.On-the-Fly Augmentation (Training Only)-
#For each image in the batch, augmentation is applied randomly:
#model never sees the exact same batch twice.
# Rotate (up to 40°).

# Flip horizontally.

# Shift up/down/left/right.

# Zoom in/out.

# Shear distortions.
#Exact Flow

# ImageDataGenerator(...) → sets the rules for augmentation.

# flow_from_directory(...) → each time it pulls a batch:

# Randomly applies augmentations within the given ranges.
#3.Forward Pass
#4.Loss Calculation
#5.Backpropagation
#6.Repeat for All Batches
#7.Validation step(validation images are only rescaled not augmented for better generalization).
