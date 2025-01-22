import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
from modelo.generar_caracteristicas import generar_caracteristicas
from keras.utils import to_categorical

def evaluar_modelo():
    # Cargar modelo
    print("Cargando modelo...")
    model = load_model("modelo_pokemon.h5")

    # Rutas
    test_path = "./PokemonData/Validacion"

    # Generar características de validación
    print("Generando características de validación...")
    caracteristicas = generar_caracteristicas(test_path)
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

    # Evaluar modelo
    print("Evaluando modelo...")
    resultados = model.evaluate(
        [X_imagenes, X_forma, X_color],
        {"pokemon_output": y_pokemon, "tipo_output": y_tipo},
        batch_size=32
    )

    print(f"Pérdida total: {resultados[0]}")
    print(f"Precisión en pruebas (Pokémon): {resultados[3] * 100:.2f}%")
    print(f"Precisión en pruebas (Tipo): {resultados[4] * 100:.2f}%")
