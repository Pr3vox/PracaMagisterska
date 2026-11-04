import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# --- ŚCIEŻKA DO GŁÓWNEGO FOLDERU ZE ZDJĘCIAMI ---
data_dir = r"C:\Users\pr3vo\Desktop\PracaMagisterska_Model\Praca Magisterska.Model\Podział_Kategorie"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32  # Jeśli masz bardzo mało zdjęć, możesz zmniejszyć np. do 4 lub 8
EPOCHS = 10

print("Wczytywanie zdjęć...")

# Ładowanie danych z folderów (80% trening, 20% walidacja)
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

# --- OPTYMALIZACJA WYDAJNOŚCI ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Pobieranie modelu bazowego ResNet50...")
base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                      include_top=False,
                      weights='imagenet')
base_model.trainable = False

# --- BUDOWA MODELU ---
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

# --- TRENING ---
print("Rozpoczynam trening modelu...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- RAPORT DLA PROMOTORA (Tylko Precision i Recall) ---
print("\n--- RAPORT KLASYFIKACJI (Precision i Recall) ---")
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# Wygenerowanie słownika z raportem (zabezpieczenie zero_division)
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
)

# Wyświetlenie czytelnej tabelki zawierającej wyłącznie Precision i Recall
print(f"{'Klasa':<15} | {'Precision':<12} | {'Recall':<12}")
print("-" * 45)

for class_name in class_names:
    prec = report_dict[class_name]["precision"]
    rec = report_dict[class_name]["recall"]
    print(f"{class_name:<15} | {prec:<12.4f} | {rec:<12.4f}")

# --- ZAPIS MODELU ---
model.save("model_ziola.h5")
print("\nModel został pomyślnie wytrenowany i zapisany jako 'model_ziola.h5'.")