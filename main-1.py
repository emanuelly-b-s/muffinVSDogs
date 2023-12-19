import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras import activations
from keras import callbacks
from keras import initializers
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image


model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), input_shape=(64, 64, 3), activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(Dense(4096, activation='relu'))
model.add(layers.Dropout(7 / 8))

model.add(Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(Dense(2, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=1), bias_initializer=initializers.Zeros()))

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=losses.BinaryCrossentropy(), metrics=[metrics.CategoricalAccuracy(), metrics.Precision()])


dataGen = image.ImageDataGenerator(
    rescale=1 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2,
    rotation_range=60,  # Novo parâmetro para rotação
    width_shift_range=0.2,  # Novo parâmetro para deslocamento horizontal
    fill_mode='nearest'  # Novo parâmetro para preenchimento de pixels após transformações
)

X_train = dataGen.flow_from_directory(
    'Images',
    target_size=(64, 64),
    batch_size=256,
    class_mode='categorical',
    subset='training'
)

X_test = dataGen.flow_from_directory(
    'Images',
    target_size=(64, 64),
    batch_size=256,
    class_mode='categorical',
    subset='validation'
)


# for images, labels in X_test.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.axis("off")


epochs = 1000

history = model.fit(
    X_train,
    batch_size = 256, #performace, dependendo processador e gpu, batch size nao pode ser maior q o tam do banco de dados 
    steps_per_epoch=X_train.samples // X_train.batch_size,
    epochs=epochs,
    validation_steps = 64, 
    callbacks= [
        # callbacks.LearningRateScheduler(lambda epoch: 2.5e-3 * 10 ** -(epoch / 75)),
        callbacks.LearningRateScheduler(lambda epoch: 2e-3 * 10 ** (-(epoch / 75) ** 0.5)),
        callbacks.EarlyStopping(monitor='loss', patience=20),
        callbacks.ModelCheckpoint(filepath='model.{epoch:03d}.keras')
    ]
)


results = model.evaluate(X_test)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
