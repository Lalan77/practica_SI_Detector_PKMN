from PIL import Image
import os
import re

# Ruta base del dataset
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

if __name__ == "__main__":
    procesar_imagenes()

    
    