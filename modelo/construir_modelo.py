from keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from keras.models import Model
from keras.applications import MobileNetV2

def construir_modelo():
    # Entrada de imágenes
    imagen_input = Input(shape=(128, 128, 3), name="imagen_input")

    # MobileNetV2 para extracción de características
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=imagen_input)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)

    # Entrada de características de contorno y color
    contorno_input = Input(shape=(7,), name="contorno_input")
    color_input = Input(shape=(512,), name="color_input")

    # Combinar características
    combinadas = Concatenate()([x, contorno_input, color_input])

    # Capas densas finales
    pokemon_output = Dense(151, activation="softmax", name="pokemon_output")(combinadas)
    tipo_output = Dense(2, activation="softmax", name="tipo_output")(combinadas)

    # Modelo final
    modelo = Model(inputs=[imagen_input, contorno_input, color_input], outputs=[pokemon_output, tipo_output])

    return modelo
