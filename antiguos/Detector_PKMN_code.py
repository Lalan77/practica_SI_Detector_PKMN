import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model, Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
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
            if file.endswith((".jpg")):
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


#Definicion del Modelo
input_layer = Input(shape=(128, 128, 3))

# Capas compartidas (convolucionales)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Salida 1: Detección del Pokémon
pokemon_output = Dense(len(pokemon_encoder.classes_), activation='softmax', name='pokemon_output')(x)

# Salida 2: Detección del tipo de variante
color_output = Dense(len(color_encoder.classes_), activation='softmax', name='color_output')(x)

# Definir el modelo
model = Model(inputs=input_layer, outputs=[pokemon_output, color_output])

model.compile(
    optimizer='adam',
    loss={'pokemon_output': 'categorical_crossentropy', 'color_output': 'categorical_crossentropy'},
    metrics={'pokemon_output': 'accuracy', 'color_output': 'accuracy'}
)

#Entrenamiento del modelo

history = model.fit(
    X_train,
    {'pokemon_output': y_train_pokemon, 'color_output': y_train_color},
    validation_data=(X_val, {'pokemon_output': y_val_pokemon, 'color_output': y_val_color}),
    epochs=20,
    batch_size=32
)

#Evaluación del modelo
scores = model.evaluate(X_test, {'pokemon_output': y_test_pokemon, 'color_output': y_test_color})
print(f"Pérdida total: {scores[0]}")

# Guardar el modelo
model.save("pokemon_model.h5")

# Función para analizar una imagen
def analizar_imagen(ruta_imagen):
    try:
        # Cargar la imagen
        img = cv2.imread(ruta_imagen)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Realizar la predicción
        pokemon_pred, color_pred = model.predict(img)

        # Obtener las clases predichas
        pokemon_clase = np.argmax(pokemon_pred)
        color_clase = np.argmax(color_pred)

        pokemon = pokemon_encoder.inverse_transform([pokemon_clase])[0]
        color = color_encoder.inverse_transform([color_clase])[0]

        print(f"Predicción: Pokémon = {pokemon}, Variante = {color}")
    except Exception as e:
        print(f"Error al analizar la imagen: {e}")

# Prueba con una imagen
ruta_imagen = input("Ingresa la ruta de una imagen para analizar: ")
if ruta_imagen:
    analizar_imagen(ruta_imagen)