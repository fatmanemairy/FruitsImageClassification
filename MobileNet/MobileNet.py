import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import pandas as pd

# Data paths
data_path = "train"
test_folder_path = "test"

# Data generators
batch_size = 64  # Increase batch size for faster training

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    validation_split=0.15,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0, validation_split=0.15
)

train_generator = train_datagen.flow_from_directory(
    data_path, batch_size=batch_size, subset="training", target_size=(224, 224)
)

val_generator = val_datagen.flow_from_directory(
    data_path, subset="validation", batch_size=batch_size, target_size=(224, 224)
)

# Pre-trained MobileNetV2 as a feature extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Model definition
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(5, activation="softmax")
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
early_stop = EarlyStopping(patience=20)
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy",
    patience=2,
    verbose=1,
    factor=0.3,
    min_lr=0.0001,
)
callbacks = [early_stop, learning_rate_reduction]

# Train the model
model.fit(
    train_generator,
    epochs=20,  # Increase the number of epochs as needed
    callbacks=callbacks,
    batch_size=batch_size,
    validation_data=val_generator
)




# Predict on test data
test_image_files = [
    f for f in os.listdir(test_folder_path) if os.path.isfile(os.path.join(test_folder_path, f))
]
predicted_classes = []

for test_image_file in test_image_files:
    test_image_path = os.path.join(test_folder_path, test_image_file)
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