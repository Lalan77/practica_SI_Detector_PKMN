import os
import shutil
from sklearn.model_selection import train_test_split

base_path = "./PokemonData"
train_path = os.path.join(base_path, "Entrenamiento")
val_path = os.path.join(base_path, "Validacion")
test_path = os.path.join(base_path, "Pruebas")

val_size = 0.15
test_size = 0.15

def distribuir_imagenes():
    for pokemon in os.listdir(train_path):
        pokemon_path = os.path.join(train_path, pokemon)
        if not os.path.isdir(pokemon_path):
            continue

        for tipo in os.listdir(pokemon_path):
            tipo_path = os.path.join(pokemon_path, tipo)
            if not os.path.isdir(tipo_path):
                continue

            imagenes = [os.path.join(tipo_path, img) for img in os.listdir(tipo_path) if img.endswith((".jpg", ".png", ".jpeg"))]

            if len(imagenes) == 0:
                print(f"Advertencia: No se encontraron imágenes en {tipo_path}.")
                continue

            train_imgs, temp_imgs = train_test_split(imagenes, test_size=(val_size + test_size), random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_size / (val_size + test_size), random_state=42)

            val_dest = os.path.join(val_path, pokemon, tipo)
            test_dest = os.path.join(test_path, pokemon, tipo)
            os.makedirs(val_dest, exist_ok=True)
            os.makedirs(test_dest, exist_ok=True)

            for img in val_imgs:
                shutil.move(img, os.path.join(val_dest, os.path.basename(img)))
            for img in test_imgs:
                shutil.move(img, os.path.join(test_dest, os.path.basename(img)))
