import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


#Wczytanie zapisanego modelu
model = tf.keras.models.load_model('model_herbaty_ziola.h5')

img_path = r"C:\Users\pr3vo\Desktop\ZDJ\Organiczne\Pokrzywa-Organiczne.jpg"  # ścieżka

# 3. Przygotowanie obrazu do analizy przez model
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# 4. Wykonanie predykcji
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# 5. Definicja klas
class_names = ['Czyste', 'Krzyżowe', 'Organiczne']

print(
    f"To zdjęcie  należy do klasy: **{class_names[np.argmax(score)]}** "
    f"z pewnością wynoszącą {100 * np.max(score):.2f}%."
)