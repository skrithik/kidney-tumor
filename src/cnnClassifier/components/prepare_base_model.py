import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(#Loads pretrained VGG16 model from TensorFlow.
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)#Saves the base model (without additional layers) to disk.

    

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all: #Freezes all layers or up to a certain layer depending on config.
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)#Adds Flatten + Dense softmax output layer.
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),#Compiles the model with SGD optimizer and categorical crossentropy loss.
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model#Returns the full model ready for training.
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(#Call _prepare_full_model() with config params.
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,#All convolutional layers of the VGG16 model are frozen.
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)#saveing the updated model

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)#Saves the TensorFlow model to the specified path.



# [Input Image]
#     (224, 224, 3)
#          ↓
# [Conv Block 1]
#     Conv2D → (224, 224, 64)
#     Conv2D → (224, 224, 64)
#     MaxPooling2D (2x2, stride=2)
#     ↓
#     (112, 112, 64)

# [Conv Block 2]
#     Conv2D → (112, 112, 128)
#     Conv2D → (112, 112, 128)
#     MaxPooling2D (2x2, stride=2)
#     ↓
#     (56, 56, 128)

# [Conv Block 3]
#     Conv2D → (56, 56, 256)
#     Conv2D → (56, 56, 256)
#     Conv2D → (56, 56, 256)
#     MaxPooling2D (2x2, stride=2)
#     ↓
#     (28, 28, 256)

# [Conv Block 4]
#     Conv2D → (28, 28, 512)
#     Conv2D → (28, 28, 512)
#     Conv2D → (28, 28, 512)
#     MaxPooling2D (2x2, stride=2)
#     ↓
#     (14, 14, 512)

# [Conv Block 5]
#     Conv2D → (14, 14, 512)
#     Conv2D → (14, 14, 512)
#     Conv2D → (14, 14, 512)
#     MaxPooling2D (2x2, stride=2)
#     ↓
#     (7, 7, 512)

# [Flatten Layer]
#     (7, 7, 512) → Flatten → (25088,)

# [Dense Output Layer]
#     (25088,) → Dense(units=2, activation='softmax') → (2,)
#       ↓
# Final Output: Tumor / Normal probabilities

#SGD
#Calculate the gradient of the loss function w.r.t model weights using a mini-batch of data.
#pdate the weights using this formula:weight = weight - learning_rate * gradient
#Repeat this for many iterations (epochs) over the dataset.
#we are using mini batch gradient descent
# Shuffle dataset.

# Split into batches of 16 → 100 batches total.
# (1600 / 16 = 100)

# For each batch:

# Forward pass (predict outputs).

# Compute loss.

# Backpropagate (compute gradients).

# Update weights with SGD rule:

# w = w - η * ∇L_batch


# (η = learning rate, ∇L_batch = gradient from this batch).

# After 100 batches → 1 epoch completed.
#categoricalcross entropy loss
#Used when there are 2 or more classes (output is softmax vector).


