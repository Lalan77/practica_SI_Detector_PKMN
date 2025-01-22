import os
import shutil

# Rutas base
base_path = "./PokemonData"
train_path = os.path.join(base_path, "Entrenamiento")
val_path = os.path.join(base_path, "Validacion")
test_path = os.path.join(base_path, "Pruebas")

def mover_a_entrenamiento():
    """
    Mueve todas las imágenes desde las carpetas de Validación y Pruebas
    de vuelta a la carpeta de Entrenamiento, respetando la estructura original.
    """
    for dataset_type, dataset_path in [("Validacion", val_path), ("Pruebas", test_path)]:
        for pokemon in os.listdir(dataset_path):
            pokemon_path = os.path.join(dataset_path, pokemon)

            if not os.path.isdir(pokemon_path):
                continue  # Ignorar archivos no relacionados

            for tipo in os.listdir(pokemon_path):
                tipo_path = os.path.join(pokemon_path, tipo)

                if not os.path.isdir(tipo_path):
                    continue  # Ignorar archivos no relacionados

                # Ruta destino en Entrenamiento
                train_dest = os.path.join(train_path, pokemon, tipo)
                os.makedirs(train_dest, exist_ok=True)

                # Mover todas las imágenes al destino
                for img in os.listdir(tipo_path):
                    img_path = os.path.join(tipo_path, img)
                    shutil.move(img_path, os.path.join(train_dest, img))

                print(f"Imágenes de {dataset_type}/{pokemon}/{tipo} movidas a Entrenamiento.")

                # Eliminar carpeta vacía
                os.rmdir(tipo_path)

            # Eliminar carpeta de Pokémon si está vacía
            if not os.listdir(pokemon_path):
                os.rmdir(pokemon_path)

if __name__ == "__main__":
    mover_a_entrenamiento()