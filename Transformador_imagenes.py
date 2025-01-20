from PIL import Image
import os

# Ruta base del dataset
base_path = "./PokemonData"

# Recorre cada subdirectorio principal (Entrenamiento, Validacion, Pruebas)
for dataset_type in ["Entrenamiento", "Validacion", "Pruebas"]:
    dataset_path = os.path.join(base_path, dataset_type)

    # Recorre los directorios de Pokémon y tipos directamente
    for root, dirs, files in os.walk(dataset_path):
        if len(files) == 0:  # Salta directorios sin imágenes
            continue
        
        # Extrae el nombre del Pokémon y el tipo desde el nombre del directorio
        path_parts = root.split(os.sep)
        pokemon_name = path_parts[-2]  # Nombre del Pokémon
        tipo = path_parts[-1]  # Tipo (Normal, Shiny, etc.)
        counter = 1  # Contador para las imágenes en este directorio

        for file in files:
            # Ruta completa del archivo
            file_path = os.path.join(root, file)
            try:
                # Abrir la imagen
                with Image.open(file_path) as img:
                    # Convertir a RGB y cambiar formato a JPG
                    img = img.convert("RGB")
                    
                    # Nombre del nuevo archivo
                    new_filename = f"{pokemon_name}_{tipo}_{counter}.jpg"
                    new_file_path = os.path.join(root, new_filename)
                    
                    # Guardar la nueva imagen
                    img.save(new_file_path, "JPEG")
                    print(f"Guardada: {new_file_path}")
                    
                    # Eliminar la imagen original
                    os.remove(file_path)
                    print(f"Eliminada: {file_path}")
                    
                    # Incrementar el contador
                    counter += 1
            except Exception as e:
                print(f"Error procesando {file_path}: {e}")
