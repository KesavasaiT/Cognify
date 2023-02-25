# Borrowed from Lukas Mendes; Reference at https://www.kaggle.com/code/lukasmendes/brain-tumor-cnn-98

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score
import keras.backend as K


model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Conv2D(32,(3,3), input_shape=(64, 64, 1), activation='relu'))
model1.add(tf.keras.layers.BatchNormalization()) # Batch normalization prevents internal covariate shift
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.Conv2D(32,(3,3), activation='relu'))
model1.add(tf.keras.layers.BatchNormalization()) # Batch normalization prevents internal covariate shift
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(tf.keras.layers.Flatten()) 
model1.add(tf.keras.layers.Dense(units= 252, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.2)) # Dropout prevents overfitting
model1.add(tf.keras.layers.Dense(units=252, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.2)) # Dropout prevents overfitting
model1.add(tf.keras.layers.Dense(units=4, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model1.compile(optimizer=optimizer, loss='categorical_crossentropy',
                   metrics= ['categorical_accuracy', f1_metric])

generator_train = ImageDataGenerator(rescale=1./255,
                                    featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    rotation_range=0,
                                    zoom_range = 0,
                                    width_shift_range=0,
                                    height_shift_range=0,
                                    horizontal_flip=True,
                                    vertical_flip=False) 

generator_test = ImageDataGenerator(rescale=1./255,
                                    featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    rotation_range=0,
                                    zoom_range = 0,
                                    width_shift_range=0,
                                    height_shift_range=0,
                                    horizontal_flip=True,
                                    vertical_flip=False)



train = generator_train.flow_from_directory('/Users/Ramesh/Downloads/dataset/Training', target_size=(64,64),
                                              batch_size=32, class_mode= "categorical", color_mode='grayscale')

test = generator_test.flow_from_directory('/Users/Ramesh/Downloads/dataset/Testing', target_size=(64,64),
                                              batch_size=32, class_mode= "categorical", color_mode='grayscale')

model1_es = EarlyStopping(monitor = 'loss', min_delta = 1e-11, patience = 12, verbose = 1) # Stop training when a monitored metric has stopped improving.

model1_rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 6, verbose = 1) # Reduce learning rate when a metric has stopped improving. 

model1_mcp = ModelCheckpoint(filepath = 'model1_weights.h5', monitor = 'val_categorical_accuracy', 
                      save_best_only = True, verbose = 1)

history1 = model1.fit(train, steps_per_epoch=5712//32, epochs=5, validation_data=test, validation_steps= 1311//32,
                     callbacks=[model1_es, model1_rlr, model1_mcp])

import numpy as np
from keras.preprocessing import image

print(model1.summary())

