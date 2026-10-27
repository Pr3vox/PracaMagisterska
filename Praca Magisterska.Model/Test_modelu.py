import numpy as np
import tensorflow as tf

# Wczytanie modelu i definicja klas
model = tf.keras.models.load_model('model_ziola.h5')
class_names = ['Czyste', 'Krzyżowe', 'Organiczne']

# Ścieżka do zdjęcia
img_path = r'C:\Users\pr3vo\Desktop\PracaMagisterska_Model\Praca Magisterska.Model\Podział_Kategorie\Organiczne\Lipa-Ograniczne02.jpg'
# Wczytanie i przeskalowanie zdjęcia
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Predykcja
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Wynik
print(
    f'To zdjęcie należy do klasy: {class_names[np.argmax(score)]} z pewnością'
    f' wynoszącą {100 * np.max(score):.2f}%.'
)