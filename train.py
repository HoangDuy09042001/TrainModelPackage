import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import  Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import applications

def train(path:str, train_ds, val_ds, epochs, use_gpu=True):
    if use_gpu and tf.config.list_physical_devices('GPU'):
        print("GPU device is available")
        device = '/gpu:0'  
    else:
        print("GPU device is not available or not requested, falling back to CPU")
        device = '/cpu:0'  
    with tf.device(device):
        no_labels = sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))
        vgg_base = applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (115, 115, 3))
        vgg_base.trainable = False
        inputs = Input(shape=(115, 115, 3))

        x = vgg_base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(no_labels, activation = 'sigmoid')(x)
        vgg_model = Model(inputs, outputs)
        vgg_model.summary()
        vgg_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits = False),
            metrics= [tf.keras.metrics.CategoricalAccuracy()],
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        vgg_model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[early_stopping])
        return vgg_model

class PyDataset(tf.data.Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
