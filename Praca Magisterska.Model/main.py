import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

#ścieżka
data_dir = "Podział_Kategorie"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

print("Wczytywanie zdjęć")

#ładowanie danych z folderów
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
print(f"Wykryte klasy ziół: {class_names}")

# Optymalizacja
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Pobieranie modelu bazowego ResNet50...")
base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                      include_top=False,
                      weights='imagenet')
base_model.trainable = False

# Budowa modelu
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Trening modelu")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Zapis modelu
model.save("model_ziola.h5")
print("Model został wytrenowany i zapisany.")