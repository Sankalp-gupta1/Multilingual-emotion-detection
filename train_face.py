import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# =========================
# PATHS
# =========================

TRAIN_DIR = r"D:\emotion-detection-whatsap\facial_emotion\dataset\train"

TEST_DIR = r"D:\emotion-detection-whatsap\facial_emotion\dataset\test"

SAVE_PATH = r"D:\emotion-detection-whatsap\facial_emotion\face_model"

MODEL_PATH = r"D:\emotion-detection-whatsap\facial_emotion\face_model\face_emotion_model.keras"

# =========================
# SETTINGS
# =========================

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 25

# =========================
# DATA GENERATOR
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("\nDetected Classes:")
print(train_data.class_indices)

# =========================
# LOAD PREVIOUS TRAINED MODEL
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nPREVIOUS MODEL LOADED SUCCESSFULLY")

# =========================
# COMPILE
# =========================

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2
)

# =========================
# CONTINUE TRAINING
# =========================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    initial_epoch=15,
    callbacks=[early_stop, reduce_lr]
)

# =========================
# SAVE UPDATED MODEL
# =========================

os.makedirs(SAVE_PATH, exist_ok=True)

model.save(os.path.join(SAVE_PATH, "face_emotion_model_v2.keras"))

print("\nUPDATED MODEL SAVED SUCCESSFULLY")

# =========================
# PLOTS
# =========================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

graph_path = os.path.join(SAVE_PATH, "training_graph_v2.png")

plt.savefig(graph_path)

plt.show()

print("\nTRAINING GRAPH SAVED")