#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np
import cv2
import os
# os.listdir()
# os.chroot('/')
# os.listdir()
# os.chdir('/AK')

os.listdir('./AK/CF')
img = image.load_img('./AK/CF/Screen Shot 2020-09-01 at 23.18.12.png')
plt.imshow(img)
i = cv2.imread('./AK/CF/Screen Shot 2020-09-01 at 23.18.12.png')
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory('./AK', target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')

validation_dataset = validation.flow_from_directory('./Valida',
                                                   target_size=(200,200),
                                                    batch_size=3,
                                                   class_mode='binary')
# train_dataset.class_indices
# train_dataset.classes

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                          input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    #
    tf.keras.layers.Dense(512, activation='relu'),
    ##
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer= RMSprop(lr=0.001), metrics=['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs=30, validation_data=validation_dataset)

dir_path = './Test'
for i in os.listdir('./Test'):
    img = image.load_img(dir_path + '/' + i, target_size=(200, 200))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    val = model.predict(images)
    if val == 0:
        print('黑咖啡')
    else:
        print("附子理中丸")

os.listdir('./Test')
