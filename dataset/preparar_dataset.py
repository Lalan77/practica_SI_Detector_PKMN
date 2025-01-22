from PIL import Image, ImageEnhance
import re
import os
import cv2
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

base_path = "./PokemonData"

def obtener_ultimo_contador_y_archivos_correctos(path, pokemon_name, tipo):
    """
    Busca en una carpeta y determina:
    1. El último número del contador basado en el nombre estándar: <Pokemon>_<Tipo>_<Número>.jpg.
    2. Una lista de archivos que ya cumplen con el formato y el nombre correcto.
    """
    regex = re.compile(rf"^{pokemon_name}_{tipo}_(\d+)\.jpg$")
    max_contador = 0
    archivos_correctos = []

    for file in os.listdir(path):
        match = regex.match(file)
        if match:
            numero = int(match.group(1))
            max_contador = max(max_contador, numero)
            archivos_correctos.append(file)

    return max_contador, archivos_correctos

def procesar_imagenes():
    """
    Cambia los nombres y formatos de las imágenes en el dataset.
    Ignora archivos con el formato correcto y comienza la cuenta desde el último número encontrado.
    """
    for dataset_type in ["Entrenamiento", "Validacion", "Pruebas"]:
        dataset_path = os.path.join(base_path, dataset_type)

        for root, dirs, files in os.walk(dataset_path):
            # Extraer Pokémon y tipo del directorio
            path_parts = root.split(os.sep)
            if len(path_parts) < 3:
                continue  # Ignorar directorios no relacionados

            pokemon_name = path_parts[-2]  # Nombre del Pokémon
            tipo = path_parts[-1]         # Tipo (Normal, Shiny, Modificado)

            # Obtener último contador y archivos correctos
            counter, archivos_correctos = obtener_ultimo_contador_y_archivos_correctos(root, pokemon_name, tipo)
            counter += 1  # Comenzar desde el siguiente número

            for file in files:
                # Ignorar archivos que ya son correctos
                if file in archivos_correctos:
                    print(f"Ignorando archivo correcto: {file}")
                    continue

                file_path = os.path.join(root, file)

                try:
                    # Abrir la imagen usando el contexto para asegurar cierre
                    with Image.open(file_path) as img:
                        # Convertir a RGB
                        img = img.convert("RGB")

                        # Generar un nuevo nombre único
                        new_filename = f"{pokemon_name}_{tipo}_{counter}.jpg"
                        new_file_path = os.path.join(root, new_filename)

                        # Guardar la nueva imagen
                        img.save(new_file_path, "JPEG")
                        print(f"Guardada: {new_file_path}")

                    # Verificar si se guardó correctamente antes de eliminar
                    if os.path.exists(new_file_path):
                        os.remove(file_path)  # Eliminar la imagen original
                        print(f"Eliminada: {file_path}")
                        counter += 1
                    else:
                        print(f"Error al guardar: {new_file_path}. No se eliminará {file_path}.")
                except Exception as e:
                    print(f"Error procesando {file_path}: {e}")


def aumentar_datos(base_path="./PokemonData", num_variaciones=25, max_workers=4):
    """
    Aumenta los datos generando imágenes rotadas, volteadas y ajustadas.
    Las imágenes aumentadas tendrán el prefijo 'aumentada_'.
    Procesa las imágenes en paralelo para mayor velocidad.
    """
    for dataset_type in ["Entrenamiento", "Validacion"]:
        dataset_path = os.path.join(base_path, dataset_type)

        for root, dirs, files in os.walk(dataset_path):
            path_parts = root.split(os.sep)
            if len(path_parts) < 3:
                continue

            pokemon_name = path_parts[-2]
            tipo = path_parts[-1]

            # Lista de tareas para el ThreadPoolExecutor
            tareas = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for file in files:
                    if file.endswith((".jpg", ".png")):
                        file_path = os.path.join(root, file)
                        # Agregar tarea para procesamiento paralelo
                        tareas.append(
                            executor.submit(procesar_imagen, file_path, root, pokemon_name, tipo, num_variaciones)
                        )

                # Esperar a que todas las tareas terminen
                for tarea in tareas:
                    tarea.result()

def procesar_imagen(file_path, root, pokemon_name, tipo, num_variaciones):
    """
    Procesa una sola imagen y genera aumentaciones.
    """
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            for i in range(num_variaciones):
                nueva_imagen = aplicar_transformaciones(img)
                nueva_imagen.save(os.path.join(root, f"aumentada_{pokemon_name}_{tipo}_{i+1}.jpg"))
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")

def aplicar_transformaciones(imagen):
    """
    Aplica transformaciones aleatorias a una imagen.
    """
    angulo = random.randint(-30, 30)
    imagen = imagen.rotate(angulo)

    if random.random() > 0.5:
        imagen = imagen.transpose(Image.FLIP_LEFT_RIGHT)

    factor_brillo = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(imagen)
    imagen = enhancer.enhance(factor_brillo)

    return imagen

def procesar_forma(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.HuMoments(cv2.moments(largest_contour)).flatten()
    return [0] * 7  # Si no hay contornos

def procesar_color(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None  # Si la imagen no se puede cargar, devuelve None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calcula histogramas para los canales RGB
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
        
        # Normaliza los histogramas y los aplana
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        
        # Concatenar los histogramas en un solo vector
        hist_rgb = np.concatenate([hist_r, hist_g, hist_b])
        return hist_rgb
    except Exception as e:
        print(f"Error procesando color para {img_path}: {e}")
        return None

def extraer_caracteristicas(base_path, pca_model=None):
    data = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                forma = procesar_forma(img_path)
                color = procesar_color(img_path, pca_model)
                data.append({"path": img_path, "forma": forma, "color": color})
    return data


