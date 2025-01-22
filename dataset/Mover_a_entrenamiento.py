import os
import shutil

base_path = "./PokemonData"
train_path = os.path.join(base_path, "Entrenamiento")
val_path = os.path.join(base_path, "Validacion")
test_path = os.path.join(base_path, "Pruebas")

def mover_a_entrenamiento():
    """
    Mueve todas las imágenes de Validación y Pruebas a Entrenamiento.
    Elimina las imágenes aumentadas (prefijo 'aumentada_').
    """
    for dataset_type, dataset_path in [("Validacion", val_path), ("Pruebas", test_path)]:
        for pokemon in os.listdir(dataset_path):
            pokemon_path = os.path.join(dataset_path, pokemon)
            if not os.path.isdir(pokemon_path):
                continue

            for tipo in os.listdir(pokemon_path):
                tipo_path = os.path.join(pokemon_path, tipo)
                if not os.path.isdir(tipo_path):
                    continue

                train_dest = os.path.join(train_path, pokemon, tipo)
                os.makedirs(train_dest, exist_ok=True)

                # Mover imágenes a Entrenamiento
                for img in os.listdir(tipo_path):
                    shutil.move(os.path.join(tipo_path, img), os.path.join(train_dest, img))

                os.rmdir(tipo_path)

            if not os.listdir(pokemon_path):
                os.rmdir(pokemon_path)

    # Eliminar imágenes aumentadas en Entrenamiento
    eliminar_imagenes_aumentadas()

def eliminar_imagenes_aumentadas():
    """
    Elimina todas las imágenes aumentadas (prefijo 'aumentada_') del dataset.
    """
    for root, dirs, files in os.walk(train_path):
        for file in files:
            if file.startswith("aumentada_"):
                os.remove(os.path.join(root, file))
                print(f"Imagen eliminada: {os.path.join(root, file)}")
