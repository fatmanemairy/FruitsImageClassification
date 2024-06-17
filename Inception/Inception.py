
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import keras, os
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models, losses
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, MaxPooling2D, GlobalAveragePooling2D,AveragePooling2D, Dropout, Activation, BatchNormalization

Data_path = "train"




train_DataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.,validation_split = 0.2)


train_d = train_DataGenerator.flow_from_directory(Data_path,
                                                    batch_size = 64,
                                                    subset="training",
                                                    target_size = (224, 224),
                                                  class_mode="categorical")

val_d = train_DataGenerator.flow_from_directory(Data_path,
                                                subset = "validation",
                                                 batch_size = 64,
                                                 target_size = (224, 224),
                                                class_mode="categorical")


def Convolutional_with_Batch_Normalisation(prev_layer, num_kernels, filter_Size, strides=(1, 1), padding='same'):
    x = Conv2D(filters=num_kernels, kernel_size=filter_Size, strides=strides, padding=padding)(prev_layer)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    return x


def Stem(prev_layer):
    x = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=32, filter_Size=(3, 3), strides=(2, 2))
    x = Convolutional_with_Batch_Normalisation(x, num_kernels=32, filter_Size=(3, 3))
    x = Convolutional_with_Batch_Normalisation(x, num_kernels=64, filter_Size=(3, 3))
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Convolutional_with_Batch_Normalisation(x, num_kernels=80, filter_Size=(1, 1))
    x = Convolutional_with_Batch_Normalisation(x, num_kernels=192, filter_Size=(3, 3))
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    return x


def Inception_A(prev_layer, num_kernels):
    branch1 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=64, filter_Size=(1, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=96, filter_Size=(3, 3))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=96, filter_Size=(3, 3))

    branch2 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=48, filter_Size=(1, 1))
    branch2 = Convolutional_with_Batch_Normalisation(branch2, num_kernels=64, filter_Size=(3, 3))

    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
    branch3 = Convolutional_with_Batch_Normalisation(branch3, num_kernels=num_kernels, filter_Size=(1, 1))

    branch4 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=64, filter_Size=(1, 1))

    output = concatenate([branch1, branch2, branch3, branch4], axis=3)

    return output


def Inception_B(prev_layer, num_kernels):
    branch1 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=num_kernels, filter_Size=(1, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=num_kernels, filter_Size=(7, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=num_kernels, filter_Size=(1, 7))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=num_kernels, filter_Size=(7, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=192, filter_Size=(1, 7))

    branch2 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=num_kernels, filter_Size=(1, 1))
    branch2 = Convolutional_with_Batch_Normalisation(branch2, num_kernels=num_kernels, filter_Size=(1, 7))
    branch2 = Convolutional_with_Batch_Normalisation(branch2, num_kernels=192, filter_Size=(7, 1))

    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
    branch3 = Convolutional_with_Batch_Normalisation(branch3, num_kernels=192, filter_Size=(1, 1))

    branch4 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=192, filter_Size=(1, 1))

    output = concatenate([branch1, branch2, branch3, branch4], axis=3)

    return output


def Inception_C(prev_layer):
    branch1 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=448, filter_Size=(1, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=384, filter_Size=(3, 3))
    branch1_1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=384, filter_Size=(1, 3))
    branch1_2 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=384, filter_Size=(3, 1))
    branch1 = concatenate([branch1_1, branch1_2], axis=3)

    branch2 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=384, filter_Size=(1, 1))
    branch2_1 = Convolutional_with_Batch_Normalisation(branch2, num_kernels=384, filter_Size=(1, 3))
    branch2_2 = Convolutional_with_Batch_Normalisation(branch2, num_kernels=384, filter_Size=(3, 1))
    branch2 = concatenate([branch2_1, branch2_2], axis=3)

    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
    branch3 = Convolutional_with_Batch_Normalisation(branch3, num_kernels=192, filter_Size=(1, 1))
    branch4 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=320, filter_Size=(1, 1))

    output = concatenate([branch1, branch2, branch3, branch4], axis=3)

    return output


def Reduction_A(prev_layer):
    branch1 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=64, filter_Size=(1, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=96, filter_Size=(3, 3))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=96, filter_Size=(3, 3), strides=(2, 2))

    branch2 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=384, filter_Size=(3, 3), strides=(2, 2))

    branch3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(prev_layer)

    output = concatenate([branch1, branch2, branch3], axis=3)

    return output


def Reduction_B(prev_layer):
    branch1 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=192, filter_Size=(1, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=192, filter_Size=(1, 7))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=192, filter_Size=(7, 1))
    branch1 = Convolutional_with_Batch_Normalisation(branch1, num_kernels=192, filter_Size=(3, 3), strides=(2, 2),
                                                                                             padding='valid')

    branch2 = Convolutional_with_Batch_Normalisation(prev_layer, num_kernels=192, filter_Size=(1, 1))
    branch2 = Convolutional_with_Batch_Normalisation(branch2, num_kernels=320, filter_Size=(3, 3), strides=(2, 2),
                                                     padding='valid')

    branch3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(prev_layer)

    output = concatenate([branch1, branch2, branch3], axis=3)

    return output
def auxiliary_classifier(prev_Layer):
  x = AveragePooling2D(pool_size=(5,5) , strides=(3,3)) (prev_Layer)
  x = Convolutional_with_Batch_Normalisation(x, num_kernels = 128, filter_Size = (1,1))
  x = Flatten()(x)
  x = Dense(units = 768, activation='relu') (x)
  x = Dropout(rate = 0.2) (x)
  x = Dense(units = 5, activation='softmax') (x)
  return x


def Inception():
    input_layer = Input(shape=(224, 224, 3))

    x = Stem(input_layer)

    x = Inception_A(prev_layer=x, num_kernels=32)
    x = Inception_A(prev_layer=x, num_kernels=64)
    x = Inception_A(prev_layer=x, num_kernels=64)

    x = Reduction_A(prev_layer=x)

    x = Inception_B(prev_layer=x, num_kernels=128)
    x = Inception_B(prev_layer=x, num_kernels=160)
    x = Inception_B(prev_layer=x, num_kernels=160)
    x = Inception_B(prev_layer=x, num_kernels=192)

    Aux = auxiliary_classifier(prev_Layer=x)

    x = Reduction_B(prev_layer=x)
    x = Inception_C(prev_layer=x)
    x = Inception_C(prev_layer=x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=5, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=[x, Aux])
    return model
from tensorflow.keras.optimizers import RMSprop
model_inception_v3 = InceptionV3()
model_inception_v3.compile( optimizer = RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(restore_best_weights=True, patience=3)

model = model_inception_v3.fit(train_d, validation_data=val_d, epochs=30, callbacks=[early_stopping])






import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import csv

test_path = "test"

# List all files in the folder
# Predict on test data
test_image_files = [
    f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))
]
predicted_classes = []

for test_image_file in test_image_files:
    test_image_path = os.path.join(test_path, test_image_file)
    test_img = image.load_img(test_image_path, target_size=(224, 224))
    test_img_array = image.img_to_array(test_img)
    test_img_array = np.expand_dims(test_img_array, axis=0)
    test_img_array /= 255.0

    predictions = model.predict(test_img_array)
    predicted_class = np.argmax(predictions, axis=-1)+1
    predicted_classes.append(predicted_class[0])

# Print or use the predicted class indices as needed
print("Predicted Classes:", predicted_classes)

# Create a DataFrame with filenames and predicted classes without file extension
result_df = pd.DataFrame(
    {
        "Filename": [os.path.splitext(f)[0] for f in test_image_files],
        "Predicted_Class": predicted_classes,
    }
)

# Save the DataFrame to a CSV file
result_df.to_csv("predicted_classes.csv", index=False)

# Display the DataFrame
print(result_df)

