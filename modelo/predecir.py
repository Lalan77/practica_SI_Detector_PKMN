import cv2
import numpy as np
from keras.models import load_model
from dataset.preparar_dataset import procesar_forma, procesar_color

def predecir(imagen_path):
    print("Cargando modelo...")
    model = load_model("modelo_pokemon.h5")

    # Cargar imagen
    print(f"Procesando imagen: {imagen_path}")
    img = cv2.imread(imagen_path)
    img_resized = cv2.resize(img, (128, 128)) / 255.0

    # Mostrar la imagen
    cv2.imshow("Imagen a analizar", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extraer contornos
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contornos = cv2.Canny(img_gray, 100, 200)
    contorno_hist = np.histogram(contornos.ravel(), bins=7, range=(0, 256))[0]
    contorno_hist = contorno_hist / np.sum(contorno_hist)

    # Extraer colores
    color_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    color_hist = color_hist / np.sum(color_hist)

    # Ajustar dimensiones para el modelo
    img_resized = np.expand_dims(img_resized, axis=0)  # Convertir a forma (1, 128, 128, 3)
    contorno_hist = np.expand_dims(contorno_hist, axis=0)  # Convertir a forma (1, 7)
    color_hist = np.expand_dims(color_hist, axis=0)  # Convertir a forma (1, 512)

    # Realizar predicción
    print("Realizando predicción...")
    predicciones = model.predict([img_resized, contorno_hist, color_hist])

    # Interpretar resultados
    pokemon_pred = np.argmax(predicciones[0])
    tipo_pred = np.argmax(predicciones[1])

    # Convertir índices a etiquetas
    unique_pokemon = sorted(set(["Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
    "Squirtle", "Wartortle", "Blastoise", "Caterpie", "Metapod", "Butterfree",
    "Weedle", "Kakuna", "Beedrill", "Pidgey", "Pidgeotto", "Pidgeot",
    "Rattata", "Raticate", "Spearow", "Fearow", "Ekans", "Arbok",
    "Pikachu", "Raichu", "Sandshrew", "Sandslash", "Nidoran_hembra", "Nidorina",
    "Nidoqueen", "Nidoran_macho", "Nidorino", "Nidoking", "Clefairy", "Clefable",
    "Vulpix", "Ninetales", "Jigglypuff", "Wigglytuff", "Zubat", "Golbat",
    "Oddish", "Gloom", "Vileplume", "Paras", "Parasect", "Venonat",
    "Venomoth", "Diglett", "Dugtrio", "Meowth", "Persian", "Psyduck",
    "Golduck", "Mankey", "Primeape", "Growlithe", "Arcanine", "Poliwag",
    "Poliwhirl", "Poliwrath", "Abra", "Kadabra", "Alakazam", "Machop",
    "Machoke", "Machamp", "Bellsprout", "Weepinbell", "Victreebel", "Tentacool",
    "Tentacruel", "Geodude", "Graveler", "Golem", "Ponyta", "Rapidash",
    "Slowpoke", "Slowbro", "Magnemite", "Magneton", "Farfetchd", "Doduo",
    "Dodrio", "Seel", "Dewgong", "Grimer", "Muk", "Shellder",
    "Cloyster", "Gastly", "Haunter", "Gengar", "Onix", "Drowzee",
    "Hypno", "Krabby", "Kingler", "Voltorb", "Electrode", "Exeggcute",
    "Exeggutor", "Cubone", "Marowak", "Hitmonlee", "Hitmonchan", "Lickitung",
    "Koffing", "Weezing", "Rhyhorn", "Rhydon", "Chansey", "Tangela",
    "Kangaskhan", "Horsea", "Seadra", "Goldeen", "Seaking", "Staryu",
    "Starmie", "MrMime", "Scyther", "Jynx", "Electabuzz", "Magmar",
    "Pinsir", "Tauros", "Magikarp", "Gyarados", "Lapras", "Ditto",
    "Eevee", "Vaporeon", "Jolteon", "Flareon", "Porygon", "Omanyte",
    "Omastar", "Kabuto", "Kabutops", "Aerodactyl", "Snorlax", "Articuno",
    "Zapdos", "Moltres", "Dratini", "Dragonair", "Dragonite", "Mewtwo",
    "Mew"]))  # Añadir más Pokémon según el dataset
    unique_tipo = ["Normal", "Shiny"]

    # Validación de índices
    if pokemon_pred >= len(unique_pokemon):
        print("Error: Índice de predicción de Pokémon fuera de rango.")
        return
    if tipo_pred >= len(unique_tipo):
        print("Error: Índice de predicción de tipo fuera de rango.")
        return

    pokemon_nombre = unique_pokemon[pokemon_pred]
    tipo_nombre = unique_tipo[tipo_pred]

    print(f"Predicción: Pokémon - {pokemon_nombre}, Tipo - {tipo_nombre}")

    # Convertir índices a etiquetas
    

    