import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import cv2

# Initialize parameters
lr = 1e-4
epochs = 25
batch_size = 32

# Paths
dataset_dir = 'C:/Securewave/data/images'
models_dir = 'C:/Securewave/models'
os.makedirs(models_dir, exist_ok=True)

# Load and preprocess data
data = []
labels = []

# Load images from 'man' and 'woman' folders
for category in ["man", "woman"]:
    path = os.path.join(dataset_dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))  # Increased to 224x224 for better feature extraction
        image = img_to_array(image)
        data.append(image)
        labels.append(category)

data = np.array(data, dtype="float") / 255.0
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Build model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Unfreeze layers in base model for fine-tuning
for layer in baseModel.layers[-20:]:
    layer.trainable = True

# Compile model
opt = Adam(learning_rate=lr, decay=lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True)

# Train the model
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# Save the model
model.save(os.path.join(models_dir, 'gender_detection_model_final.keras'))

# Plot training/validation loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig(os.path.join(models_dir, 'training_plot.acc'))
