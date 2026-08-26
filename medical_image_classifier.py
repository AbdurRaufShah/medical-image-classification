"""
Medical Image Classification - Transfer Learning (TensorFlow/Keras)
=====================================================================

Ye script kisi bhi medical image dataset (X-ray, MRI, skin lesion, etc.)
ko classify karne ke liye ready hai. Sirf apna dataset path set karein.

DATASET FOLDER STRUCTURE (required):
    dataset/
        train/
            class1/  (e.g. NORMAL)
                img1.jpg
                img2.jpg
            class2/  (e.g. PNEUMONIA)
                img1.jpg
        val/
            class1/
            class2/
        test/
            class1/
            class2/

INSTALL REQUIREMENTS:
    pip install tensorflow numpy matplotlib scikit-learn --break-system-packages

RECOMMENDED DATASET (beginner-friendly):
    Chest X-Ray Pneumonia:
    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. CONFIGURATION  -- apni requirement ke hisaab se change karein
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"          # apna dataset ka path yahan dein
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)            # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "best_medical_model.h5"

# ---------------------------------------------------------------------------
# 2. DATA LOADING + AUGMENTATION
# ---------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_generator.num_classes
class_labels = list(train_generator.class_indices.keys())
print(f"Classes found: {class_labels}")

# ---------------------------------------------------------------------------
# 3. MODEL - Transfer Learning with MobileNetV2 (fast, lightweight, accurate)
# ---------------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base layers initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------------------------
# 4. CALLBACKS
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
]

# ---------------------------------------------------------------------------
# 5. TRAINING - Phase 1 (frozen base)
# ---------------------------------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------------------------------------------------------------------
# 6. FINE-TUNING - Phase 2 (unfreeze top layers of base model)
# ---------------------------------------------------------------------------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # last 30 layers trainable

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# ---------------------------------------------------------------------------
# 7. EVALUATION ON TEST SET
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predictions for detailed metrics
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ---------------------------------------------------------------------------
# 8. TRAINING CURVES
# ---------------------------------------------------------------------------
acc = history.history["accuracy"] + fine_tune_history.history["accuracy"]
val_acc = history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
loss = history.history["loss"] + fine_tune_history.history["loss"]
val_loss = history.history["val_loss"] + fine_tune_history.history["val_loss"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

print("\nModel saved as:", MODEL_SAVE_PATH)
print("Done!")

# ---------------------------------------------------------------------------
# 9. SINGLE IMAGE PREDICTION (for later use / deployment)
# ---------------------------------------------------------------------------
def predict_single_image(img_path, model=model, class_labels=class_labels):
    """Ek single image ko predict karne ke liye function."""
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_class = class_labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
    return predicted_class, confidence

# Example usage:
# predict_single_image("dataset/test/PNEUMONIA/sample.jpg")
