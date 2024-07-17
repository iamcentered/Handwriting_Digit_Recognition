import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255
Y_train = tf.keras.utils.to_categorical(train_labels, 10)
X_test = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255

datagen = ImageDataGenerator(
    rotation_range=15,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)
datagen.fit(X_train)

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(datagen.flow(X_train, Y_train, batch_size=32),
          epochs=10,
          validation_data=(X_test, tf.keras.utils.to_categorical(test_labels, 10)))

# Save model
model.save('digit_classifier.h5')
