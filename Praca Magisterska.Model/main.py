import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Ścieżka do folderu ze zdjęciami
data_dir = r"C:\Users\pr3vo\Desktop\PracaMagisterska_Model\Praca Magisterska.Model\Podział_Kategorie"

# Ustawienia
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 4
EPOCHS = 10

# Wczytanie zdjęć: 80% trening, 20% walidacja
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Klasy: {class_names}")


# Pre-processing pod ResNet50
def prepare(image, label):
    return preprocess_input(image), label


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(prepare, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(prepare, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# Augmentacja (działa tylko w trakcie treningu)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
])

# Model bazowy ResNet50 (zamrożony)
base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                      include_top=False,
                      weights="imagenet")
base_model.trainable = False

# Budowa modelu
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Trening
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Predykcje na zbiorze walidacyjnym
y_true = []
y_pred = []
for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Accuracy
print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}\n")

# Precision i recall dla każdej klasy
report = classification_report(
    y_true, y_pred,
    labels=range(num_classes),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

print(f"{'Klasa':<15} | {'Precision':<12} | {'Recall':<12}")
print("-" * 45)
for name in class_names:
    print(f"{name:<15} | {report[name]['precision']:<12.4f} | {report[name]['recall']:<12.4f}")

# Macierz pomyłek
print("\nMacierz pomyłek:")
print(confusion_matrix(y_true, y_pred, labels=range(num_classes)))

# Zapis modelu
model.save("model2.h5")
print("\nModel zapisany jako 'model2.h5'.")