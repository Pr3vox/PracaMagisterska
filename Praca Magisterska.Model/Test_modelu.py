import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Wczytanie modelu
model = tf.keras.models.load_model('model_lepszy.h5')
class_names = ['Czyste', 'Krzyżowe', 'Organiczne']

# Ścieżka do zdjęcia, którego model NIE widział w treningu
img_path = r'C:\Users\pr3vo\Desktop\PracaMagisterska_Model\Praca Magisterska.Model\Podział_Kategorie\Krzyżowe\Czystek-Pokrzywa02.jpg'

# Wczytanie i przeskalowanie zdjęcia
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Pre-processing dedykowany dla ResNet50
img_array = preprocess_input(img_array)

# Predykcja (model ma już softmax na końcu, więc nie liczymy go drugi raz)
predictions = model.predict(img_array)[0]

predicted_class = class_names[np.argmax(predictions)]
confidence = 100 * np.max(predictions)

print(f"To zdjęcie należy do klasy: {predicted_class} z pewnością wynoszącą {confidence:.2f}%.")