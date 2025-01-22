import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

def generador_pokemon_tipo(base_path, target_size=(128, 128), batch_size=32):
    """
    Genera datos para entrenamiento o validación desde la estructura ./Entrenamiento/pokemon/tipo.
    - pokemon_output: Basado en la carpeta del Pokémon.
    - tipo_output: Basado en la subcarpeta (Normal, Shiny).
    """
    pokemons = sorted(os.listdir(base_path))
    pokemon_indices = {name: idx for idx, name in enumerate(pokemons)}
    tipo_indices = {"Normal": 0, "Shiny": 1}

    # Crear listas de imágenes y etiquetas
    imagenes = []
    pokemon_labels = []
    tipo_labels = []

    for pokemon in pokemons:
        pokemon_path = os.path.join(base_path, pokemon)
        if not os.path.isdir(pokemon_path):
            continue

        for tipo in os.listdir(pokemon_path):
            tipo_path = os.path.join(pokemon_path, tipo)
            if not os.path.isdir(tipo_path):
                continue

            for img_name in os.listdir(tipo_path):
                img_path = os.path.join(tipo_path, img_name)
                try:
                    img = load_img(img_path, target_size=target_size)
                    img_array = img_to_array(img)
                    imagenes.append(img_array)
                    pokemon_labels.append(pokemon_indices[pokemon])
                    tipo_labels.append(tipo_indices[tipo])
                except Exception as e:
                    print(f"Error al procesar {img_path}: {e}")

    # Convertir a numpy arrays
    imagenes = np.array(imagenes)
    pokemon_labels = np.array(pokemon_labels)
    tipo_labels = np.array(tipo_labels)

    # Normalizar etiquetas
    pokemon_labels = np.eye(len(pokemon_indices))[pokemon_labels]
    tipo_labels = np.eye(len(tipo_indices))[tipo_labels]

    # Crear batches
    while True:
        for i in range(0, len(imagenes), batch_size):
            x_batch = imagenes[i:i + batch_size]
            y_pokemon_batch = pokemon_labels[i:i + batch_size]
            y_tipo_batch = tipo_labels[i:i + batch_size]

            yield x_batch, {"pokemon_output": y_pokemon_batch, "tipo_output": y_tipo_batch}
