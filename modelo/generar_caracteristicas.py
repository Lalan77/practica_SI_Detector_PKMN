import cv2
import os
import numpy as np
import pandas as pd

def generar_caracteristicas(base_path):
    data = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                try:
                    print(f"Procesando {img_path}")
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, (128, 128))

                    # Contornos
                    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    contornos = cv2.Canny(img_gray, 100, 200)
                    contorno_hist = np.histogram(contornos.ravel(), bins=7, range=(0, 256))[0]

                    # Colores
                    color_hist = cv2.calcHist([img_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

                    # Etiquetas
                    pokemon = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                    tipo = os.path.basename(os.path.dirname(img_path))
                    if tipo not in ["Normal", "Shiny"]:
                        tipo = np.nan

                    data.append({
                        "imagen": img_resized / 255.0,
                        "contorno": contorno_hist / np.sum(contorno_hist),
                        "color": color_hist / np.sum(color_hist),
                        "pokemon": pokemon,
                        "tipo": tipo
                    })
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")

    caracteristicas = pd.DataFrame(data)

    caracteristicas.dropna(inplace=True)

    return caracteristicas
