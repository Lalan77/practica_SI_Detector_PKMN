import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ruta base del dataset
base_path = "./PokemonData"


def cargar_imagenes_con_doble_etiqueta(base_path, dataset_type):
    pokemon_labels = []  # Etiqueta del Pokémon
    color_labels = []    # Etiqueta del tipo (Normal, Shiny, Modificado)
    imagenes = []        # Imágenes preprocesadas

    dataset_path = os.path.join(base_path, dataset_type)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)

                # Cargar y procesar la imagen
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))

                # Etiquetas: Pokémon y Tipo
                path_parts = root.split(os.sep)
                pokemon = path_parts[-2]  # Nombre del Pokémon
                tipo = path_parts[-1]     # Tipo (Normal, Shiny, Modificado)

                pokemon_labels.append(pokemon)
                color_labels.append(tipo)
                imagenes.append(img)

    return np.array(imagenes), np.array(pokemon_labels), np.array(color_labels)

# Cargar datos con dos etiquetas
X_train, y_train_pokemon, y_train_color = cargar_imagenes_con_doble_etiqueta(base_path, "Entrenamiento")
X_val, y_val_pokemon, y_val_color = cargar_imagenes_con_doble_etiqueta(base_path, "Validacion")
X_test, y_test_pokemon, y_test_color = cargar_imagenes_con_doble_etiqueta(base_path, "Pruebas")

# Normalización de las imágenes
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Codificar etiquetas
pokemon_encoder = LabelEncoder()
color_encoder = LabelEncoder()

y_train_pokemon = to_categorical(pokemon_encoder.fit_transform(y_train_pokemon))
y_val_pokemon = to_categorical(pokemon_encoder.transform(y_val_pokemon))
y_test_pokemon = to_categorical(pokemon_encoder.transform(y_test_pokemon))

y_train_color = to_categorical(color_encoder.fit_transform(y_train_color))
y_val_color = to_categorical(color_encoder.transform(y_val_color))
y_test_color = to_categorical(color_encoder.transform(y_test_color))
