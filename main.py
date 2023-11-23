import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import activations
from keras import initializers
from keras import regularizers
from keras import optimizers
from keras import metrics
from keras import losses
from keras import callbacks
import scipy
from keras.preprocessing import image

model = models.Sequential()


model.add(layers.Conv2D(
    34, (5,5), 
    input_shape = (64, 64, 3), 
    activation = 'relu'
))

model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))

model.add(layers.Conv2D(
    16, (5,5), 
    activation = 'relu'
))

model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))

model.add(layers.Conv2D(
    8, (5,5), 
    activation = 'relu'
))

model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))

model.add(layers.Flatten())

model.add(layers.Dense(256, 
    kernel_initializer= initializers.RandomNormal(stddev=1), 
    bias_initializer=initializers.Zeros())
)
model.add(layers.Activation(activations.relu))

model.add(layers.Dropout(0.2))
model.add(layers.Activation(activations.relu))



model.add(layers.Dense(
    64, 
   kernel_regularizer= regularizers.L2(1e-4),
    kernel_initializer= initializers.RandomNormal(stddev = 1),
    bias_initializer= initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    64, 
   kernel_regularizer= regularizers.L2(1e-4),
    kernel_initializer= initializers.RandomNormal(stddev = 1),
    bias_initializer= initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    64, 
   kernel_regularizer= regularizers.L2(1e-4),
    kernel_initializer= initializers.RandomNormal(stddev = 1),
    bias_initializer= initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    2, 
    kernel_initializer= initializers.RandomNormal(stddev = 1),
    bias_initializer= initializers.Zeros()
))
model.add(layers.Activation(activations.softmax))

model.compile(
    optimizer=optimizers.Adam(),
    # Loss : indica o quão esta errado
    loss=losses.BinaryCrossentropy(),
    metrics=[metrics.CategoricalAccuracy(), metrics.Precision()]
)


dataGen = image.ImageDataGenerator(
    rescale= 1.0/255,
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2
)

X_train = dataGen.flow_from_directory(
    'Images',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

X_test = dataGen.flow_from_directory(
    'Images',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'

)

model.fit(
    X_train,
    steps_per_epoch=X_train.samples // X_train.batch_size,
    epochs=50,
    validation_steps = 78,
    callbacks= [
        callbacks.ModelCheckpoint(
            filepath = 'model.{epoch:02d}.h5'
        )
    ]
)

results = model.evaluate(X_test)

metric_names = model.metrics_names
for metric_name, result in zip(metric_names, results):
    print(f'{metric_name}: {result}')