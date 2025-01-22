from dataset.preparar_dataset import procesar_imagenes, aumentar_datos
from dataset.dividir_dataset import distribuir_imagenes
from dataset.Mover_a_entrenamiento import mover_a_entrenamiento
from modelo.entrenar_modelo import entrenar_modelo
from modelo.evaluar_modelo import evaluar_modelo
from modelo.predecir import predecir
import time

def main():
    

    print("Selecciona una acción:")
    print("1. Transformar y aumentar imágenes")
    print("2. Dividir dataset en Entrenamiento, Validación y Pruebas")
    print("3. Restaurar dataset (mover a Entrenamiento y eliminar aumentadas)")
    print("4. Entrenar el modelo")
    print("5. Evaluar el modelo")
    print("6. Predecir una imagen")

    opcion = input("Ingrese el número de la acción: ")

    if opcion == "1":
        print("Transformando y aumentando imágenes...")
        procesar_imagenes()
        start_time = time.time()
        aumentar_datos()
        end_time = time.time()
        print(f"Tiempo total de aumentación: {end_time - start_time:.2f} segundos")
    elif opcion == "2":
        print("Dividiendo imágenes en Entrenamiento, Validación y Pruebas...")
        distribuir_imagenes()
    elif opcion == "3":
        print("Restaurando dataset...")
        mover_a_entrenamiento()
    elif opcion == "4":
        print("Entrenando modelo...")
        entrenar_modelo()
    elif opcion == "5":
        print("Evaluando modelo...")
        evaluar_modelo()
    elif opcion == "6":
        ruta_imagen = input("Ingrese la ruta de la imagen para predecir: ")
        predecir(ruta_imagen)
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    while True:
        main()
        continuar = input("¿Desea realizar otra acción? (s/n): ")
        if continuar.lower() == "n":
            break