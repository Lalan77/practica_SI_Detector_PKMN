import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from modelo.construir_modelo import construir_modelo
from modelo.generar_caracteristicas import generar_caracteristicas

import pandas as pd
import numpy as np
import os
import cv2
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from modelo.construir_modelo import construir_modelo
from modelo.generar_caracteristicas import generar_caracteristicas

def entrenar_modelo():
    # Configuración de rutas
    train_path = "./PokemonData/Entrenamiento"

    # Generar características
    print("Generando características...")
    caracteristicas = generar_caracteristicas(train_path)

    # Mapear nombres de Pokémon a índices numéricos
    unique_pokemon = sorted(caracteristicas["pokemon"].unique())
    pokemon_to_index = {name: idx for idx, name in enumerate(unique_pokemon)}
    caracteristicas["pokemon_idx"] = caracteristicas["pokemon"].map(pokemon_to_index)

    # Convertir etiquetas a one-hot encoding
    print("Convirtiendo etiquetas...")
    y_pokemon = to_categorical(caracteristicas["pokemon_idx"], num_classes=len(unique_pokemon))
    y_tipo = to_categorical(caracteristicas["tipo"].map({"Normal": 0, "Shiny": 1}), num_classes=2)

    # Separar características
    print("Preparando datos...")
    X_imagenes = np.array(caracteristicas["imagen"].tolist())
    X_forma = np.array(caracteristicas["contorno"].tolist())
    X_color = np.array(caracteristicas["color"].tolist())

    # Dividir datos en entrenamiento y validación
    from sklearn.model_selection import train_test_split
    X_train, X_val, X_forma_train, X_forma_val, X_color_train, X_color_val, y_pokemon_train, y_pokemon_val, y_tipo_train, y_tipo_val = train_test_split(
        X_imagenes, X_forma, X_color, y_pokemon, y_tipo, test_size=0.2, random_state=42
    )

    # Construir modelo
    print("Construyendo modelo...")
    model = construir_modelo()
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
    optimizer=optimizer,
    loss={"pokemon_output": "categorical_crossentropy", "tipo_output": "categorical_crossentropy"},
    metrics={"pokemon_output": "accuracy", "tipo_output": "accuracy"}
    )

    # Entrenar modelo
    print("Entrenando modelo...")
    model.fit(
        [X_train, X_forma_train, X_color_train],
        {"pokemon_output": y_pokemon_train, "tipo_output": y_tipo_train},
        validation_data=(
            [X_val, X_forma_val, X_color_val],
            {"pokemon_output": y_pokemon_val, "tipo_output": y_tipo_val},
        ),
        epochs=20,
        batch_size=32,
    )

    # Guardar modelo
    model.save("modelo_pokemon.h5")
    print("Modelo guardado como modelo_pokemon.h5")

