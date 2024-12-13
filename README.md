#complete code for the model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import keras.backend as K
import tensorflow as tf

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    directory='../input/train',
    target_size=(48, 48),
    class_mode='categorical',
    subset='training',
    batch_size=64
)

valid_dataset = valid_datagen.flow_from_directory(
    directory='../input/train',
    target_size=(48, 48),
    class_mode='categorical',
    subset='validation',
    batch_size=64
)

test_dataset = test_datagen.flow_from_directory(
    directory='../input/test',
    target_size=(48, 48),
    class_mode='categorical',
    batch_size=64
)

# Base Model (ResNet50)
base_model = ResNet50(input_shape=(48, 48, 3), include_top=False, weights="imagenet")

# Freeze Layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Model Architecture
model = Sequential([
    base_model,
    Dropout(0.5),
    Flatten(),
    BatchNormalization(),
    Dense(32, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(32, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(32, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dense(7, activation='softmax')
])

# Model Summary
model.summary()

# Custom F1 Score
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# Metrics
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    f1_score
]

# Callbacks
lrd = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.5, min_lr=1e-10)
mcp = ModelCheckpoint('model.h5', save_best_only=True, verbose=1)
es = EarlyStopping(patience=20, verbose=1, restore_best_weights=True)

# Compile Model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=METRICS)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=60,
    callbacks=[lrd, mcp, es],
    verbose=1
)

# Plotting Function
def train_val_plot(history):
    metrics = ['accuracy', 'loss', 'auc', 'precision', 'f1_score']
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, metric in enumerate(metrics):
        axs[i].plot(history.history[metric], label=f'Train {metric}')
        axs[i].plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        axs[i].set_title(f'{metric.capitalize()} Over Epochs')
        axs[i].legend()
    plt.show()

train_val_plot(history)
