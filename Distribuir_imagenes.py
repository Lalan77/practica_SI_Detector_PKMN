import os
import shutil
from sklearn.model_selection import train_test_split

# Ruta base del dataset
base_path = "./PokemonData"
train_path = os.path.join(base_path, "Entrenamiento")
val_path = os.path.join(base_path, "Validacion")
test_path = os.path.join(base_path, "Pruebas")

# Proporciones para Validación y Pruebas
val_size = 0.15  # 15% para validación
test_size = 0.15  # 15% para pruebas

def crear_directorios_si_no_existen(base_path, subdirectorios):
    """
    Crea los directorios especificados si no existen.
    """
    for subdir in subdirectorios:
        subdir_path = os.path.join(base_path, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

def distribuir_imagenes():
    """
    Distribuye las imágenes de Entrenamiento a Validación y Pruebas, creando
    las carpetas necesarias automáticamente.
    """
    # Recorrer las carpetas de entrenamiento
    for pokemon in os.listdir(train_path):
        pokemon_path = os.path.join(train_path, pokemon)

        if not os.path.isdir(pokemon_path):
            continue  # Ignorar archivos no relacionados

        for tipo in os.listdir(pokemon_path):
            tipo_path = os.path.join(pokemon_path, tipo)

            if not os.path.isdir(tipo_path):
                continue  # Ignorar archivos no relacionados

            # Leer todas las imágenes del directorio
            imagenes = [os.path.join(tipo_path, img) for img in os.listdir(tipo_path) if img.endswith((".jpg", ".png", ".jpeg"))]

            # Verificar si hay imágenes en el directorio
            if len(imagenes) == 0:
                print(f"Advertencia: No se encontraron imágenes en {tipo_path}.")
                continue

            # Dividir las imágenes
            train_imgs, temp_imgs = train_test_split(imagenes, test_size=(val_size + test_size), random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_size / (val_size + test_size), random_state=42)

            # Crear carpetas correspondientes en Validación y Pruebas
            val_dest = os.path.join(val_path, pokemon, tipo)
            test_dest = os.path.join(test_path, pokemon, tipo)
            os.makedirs(val_dest, exist_ok=True)
            os.makedirs(test_dest, exist_ok=True)

            # Mover las imágenes
            print(f"Distribuyendo imágenes para {pokemon}/{tipo}...")
            for img in val_imgs:
                shutil.move(img, os.path.join(val_dest, os.path.basename(img)))
            for img in test_imgs:
                shutil.move(img, os.path.join(test_dest, os.path.basename(img)))

            print(f"Imágenes distribuidas: {len(val_imgs)} en Validación, {len(test_imgs)} en Pruebas.")