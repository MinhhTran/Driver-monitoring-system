import tensorflow as tf, keras
import os

# Configuration
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 32
DATASET_DIR = 'dataset/train'

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="rgb", #grayscale
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="rgb",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print("Building MobileNetV2 architecture...")
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.1, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  tf.keras.layers.RandomZoom(0.1),
  tf.keras.layers.RandomTranslation(0.1, 0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    alpha=0.35, 
    include_top=False,
    weights='imagenet'
)

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    # 0 = normal, 1 = fatigue
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
    #tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    #tf.keras.layers.MaxPooling2D(2, 2),
    #tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2, 2),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(16, activation='relu'),
    #tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

print("Starting training...")
EPOCHS = 36

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

model.save('fatigue_model_base.h5')
print("Model saved as 'fatigue_model_base.h5'")