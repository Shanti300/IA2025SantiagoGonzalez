# Proyecto: Modelos de Lenguaje y Chatbots

## Descripción

Este repositorio contiene dos ejercicios orientados a la configuración de entornos de desarrollo, carga de modelos pre-entrenados y generación de respuestas utilizando bibliotecas populares como **Transformers** de Hugging Face y **PyTorch**.

Los ejercicios están enfocados en el procesamiento de entradas y la creación de un sistema que pueda interactuar con el usuario de forma coherente.

## Ejercicio 1: Configuración del Entorno y Carga de Modelo Base

### Objetivo

El objetivo de este ejercicio es establecer un entorno adecuado para trabajar con **modelos LLM** (Language Model) y cargar un modelo pre-entrenado usando las bibliotecas **Transformers** y **PyTorch**.

### Código Base

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configuración de las variables de entorno para la caché de modelos
# Establecer la carpeta donde se almacenarán los modelos descargados
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.
    
    Args:
        nombre_modelo (str): Identificador del modelo en Hugging Face Hub
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    # Cargar el modelo y el tokenizador
    modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    
    # Configurar el modelo para inferencia (evaluar y usar half-precision si es posible)
    if torch.cuda.is_available():
        modelo = modelo.half().to('cuda')
    else:
        modelo = modelo.to('cpu')
    
    return modelo, tokenizador

def verificar_dispositivo():
    """
    Verifica el dispositivo disponible (CPU/GPU) y muestra información relevante.
    
    Returns:
        torch.device: Dispositivo a utilizar
    """
    if torch.cuda.is_available():
        dispositivo = torch.device('cuda')
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = torch.device('cpu')
        print("Usando CPU")
    
    return dispositivo

# Función principal de prueba
def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")
    
    # Cargar el modelo pequeño adecuado para chatbots (ej. GPT2)
    modelo, tokenizador = cargar_modelo("gpt2")
    
    # Realizar una prueba simple de generación de texto
    input_text = "Hola, ¿cómo estás?"
    input_ids = tokenizador(input_text, return_tensors='pt').input_ids.to(dispositivo)
    output = modelo.generate(input_ids, max_length=50)
    print(tokenizador.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
