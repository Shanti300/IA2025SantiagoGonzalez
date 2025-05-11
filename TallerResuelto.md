# Implementación y Optimización de Chatbot con Modelos de Lenguaje de Gran Escala (LLM)
### Por Santiago González Olarte - Ing de sistemas Unilibre seccional Cali, 9 de mayo de 2025


## Tabla de Contenidos
- [Ejercicio 1: Configuración del Entorno y Carga de Modelo Base](#ejercicio-1-configuración-del-entorno-y-carga-de-modelo-base)
- [Ejercicio 2: Procesamiento de Entrada y Generación de Respuestas](#ejercicio-2-procesamiento-de-entrada-y-generación-de-respuestas)
- [Ejercicio 3: Manejo de Contexto Conversacional](#ejercicio-3-manejo-de-contexto-conversacional)
- [Ejercicio 4: Optimización del Modelo para Recursos Limitados](#ejercicio-4-optimización-del-modelo-para-recursos-limitados)
- [Ejercicio 5: Personalización del Chatbot y Despliegue](#ejercicio-5-personalización-del-chatbot-y-despliegue)
- [Preguntas teoricas](#Preguntas-teoricas)

## Ejercicio 1: Configuración del Entorno y Carga de Modelo Base

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configuración de la caché de modelos
os.environ['TRANSFORMERS_CACHE'] = './model_cache'


def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.
    
    Args:
        nombre_modelo (str): Identificador del modelo en Hugging Face Hub
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    print(f"Cargando el modelo '{nombre_modelo}'...")
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
    modelo.eval()
    if torch.cuda.is_available():
        modelo = modelo.half().to('cuda')
    else:
        modelo = modelo.to('cpu')
    print("Modelo cargado correctamente.")
    return modelo, tokenizador


def verificar_dispositivo():
    """
    Verifica el dispositivo disponible (CPU/GPU) y muestra información relevante.
    
    Returns:
        torch.device: Dispositivo a utilizar
    """
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dispositivo.type == 'cuda':
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Usando CPU")
    return dispositivo


def generar_texto(modelo, tokenizador, prompt, max_length=50):
    """
    Genera texto utilizando el modelo y el prompt proporcionado.
    
    Args:
        modelo: El modelo de lenguaje cargado.
        tokenizador: El tokenizador correspondiente.
        prompt (str): El texto de entrada para generar texto.
        max_length (int): Longitud máxima del texto generado.
    
    Returns:
        str: Texto generado.
    """
    inputs = tokenizador(prompt, return_tensors='pt').to(modelo.device)
    outputs = modelo.generate(**inputs, max_length=max_length)
    texto_generado = tokenizador.decode(outputs[0], skip_special_tokens=True)
    return texto_generado


def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")

    nombre_modelo = 'gpt2'  # Puedes cambiarlo por otro modelo si deseas
    modelo, tokenizador = cargar_modelo(nombre_modelo)

    prompt = "Hola, ¿cómo estás?"
    print("Generando texto...")
    texto = generar_texto(modelo, tokenizador, prompt)
    print("Texto generado:")
    print(texto)


if __name__ == "__main__":
    main()

```

## Ejercicio 2: Procesamiento de Entrada y Generación de Respuestas

```python
import torch

def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    """
    Preprocesa el texto de entrada para pasarlo al modelo.
    
    Args:
        texto (str): Texto de entrada del usuario
        tokenizador: Tokenizador del modelo
        longitud_maxima (int): Longitud máxima de la secuencia
    
    Returns:
        torch.Tensor: Tensor de entrada para el modelo
    """
    # Añadir tokens especiales si son necesarios
    # Algunos modelos requieren tokens especiales como [BOS], [SEP], etc.
    if hasattr(tokenizador, 'bos_token') and tokenizador.bos_token:
        texto = tokenizador.bos_token + texto
    
    # Tokenizar el texto
    tokens = tokenizador(
        texto,
        max_length=longitud_maxima,
        padding='max_length',
        truncation=True,
        return_tensors='pt'  # PyTorch tensors
    )
    
    # Pasar al dispositivo correspondiente (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in tokens:
        tokens[key] = tokens[key].to(device)
    
    return tokens

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera una respuesta basada en la entrada procesada.
    
    Args:
        modelo: Modelo de lenguaje
        entrada_procesada: Tokens de entrada procesados
        tokenizador: Tokenizador del modelo
        parametros_generacion (dict): Parámetros para controlar la generación
        
    Returns:
        str: Respuesta generada
    """
    # Configurar parámetros por defecto para la generación
    if parametros_generacion is None:
        # Usar max_new_tokens en lugar de max_length para evitar conflictos con la longitud de entrada
        parametros_generacion = {
            'max_new_tokens': 150,  # Genera hasta 150 nuevos tokens después de la entrada
            'min_length': entrada_procesada['input_ids'].shape[1] + 20,  # Asegura al menos 20 tokens nuevos
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'no_repeat_ngram_size': 2,
            'num_beams': 3,
            'do_sample': True,
            'early_stopping': True,
            'pad_token_id': tokenizador.pad_token_id if hasattr(tokenizador, 'pad_token_id') else tokenizador.eos_token_id,
            'eos_token_id': tokenizador.eos_token_id
        }
    
    # Generar texto usando el modelo
    with torch.no_grad():  # Desactivar cálculo de gradientes para inferencia
        salida_ids = modelo.generate(
            input_ids=entrada_procesada['input_ids'],
            attention_mask=entrada_procesada.get('attention_mask', None),
            **parametros_generacion
        )
    
    # Decodificar la salida y limpiar la respuesta
    respuesta_cruda = tokenizador.decode(salida_ids[0], skip_special_tokens=True)
    
    # Limpiar la respuesta - Quitar el texto de entrada si está contenido en la respuesta
    entrada_decodificada = tokenizador.decode(entrada_procesada['input_ids'][0], skip_special_tokens=True)
    if respuesta_cruda.startswith(entrada_decodificada):
        respuesta_limpia = respuesta_cruda[len(entrada_decodificada):].strip()
    else:
        respuesta_limpia = respuesta_cruda.strip()
    
    return respuesta_limpia

def crear_prompt_sistema(instrucciones):
    """
    Crea un prompt de sistema para dar instrucciones al modelo.
    
    Args:
        instrucciones (str): Instrucciones sobre cómo debe comportarse el chatbot
    
    Returns:
        str: Prompt formateado
    """
    # Formatear el prompt de sistema
    prompt_sistema = f"""
    Instrucciones del sistema:
    {instrucciones}
    
    A continuación, responde al usuario de manera coherente siguiendo las instrucciones anteriores.
    ---
    """
    
    return prompt_sistema.strip()

def cargar_modelo(modelo_nombre):
    """
    Carga el modelo y el tokenizador.
    
    Args:
        modelo_nombre (str): Nombre o ruta del modelo a cargar
        
    Returns:
        tuple: (modelo, tokenizador)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Cargar el tokenizador y el modelo
    tokenizador = AutoTokenizer.from_pretrained(modelo_nombre)
    modelo = AutoModelForCausalLM.from_pretrained(modelo_nombre)
    
    # Si el tokenizador no tiene token de padding, añadirlo
    if not hasattr(tokenizador, 'pad_token') or tokenizador.pad_token is None:
        if hasattr(tokenizador, 'eos_token'):
            tokenizador.pad_token = tokenizador.eos_token
        else:
            tokenizador.add_special_tokens({'pad_token': '[PAD]'})
            modelo.resize_token_embeddings(len(tokenizador))
    
    # Mover modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    
    return modelo, tokenizador

# Ejemplo de uso
def interaccion_simple():
    modelo, tokenizador = cargar_modelo("gpt2")  # Cambia por el modelo que uses
    
    # Crear un prompt de sistema para definir la personalidad del chatbot
    instrucciones = """
    Eres un asistente amable y servicial que proporciona información clara y precisa.
    Responde de manera respetuosa y concisa a las preguntas del usuario.
    Cuando no sepas la respuesta, indícalo honestamente en lugar de inventar información.
    """
    
    prompt_sistema = crear_prompt_sistema(instrucciones)
    
    # Ejemplo de entrada del usuario
    entrada_usuario = "¿Puedes explicarme cómo funciona la inteligencia artificial?"
    
    # Combinar el prompt del sistema con la entrada del usuario
    entrada_completa = f"{prompt_sistema}\nUsuario: {entrada_usuario}\nAsistente:"
    
    # Procesar la entrada
    entrada_procesada = preprocesar_entrada(entrada_completa, tokenizador, longitud_maxima=512)
    
    # Generar y mostrar la respuesta
    respuesta = generar_respuesta(modelo, entrada_procesada, tokenizador)
    print(f"Usuario: {entrada_usuario}")
    print(f"Asistente: {respuesta}")
    
    return respuesta

# Ejecutar el ejemplo si se ejecuta el script directamente
if __name__ == "__main__":
    interaccion_simple()

```

## Ejercicio 3: Manejo de Contexto Conversacional

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def cargar_modelo(modelo_id):
    """
    Carga el modelo y el tokenizador.
    
    Args:
        modelo_id (str): Identificador del modelo en Hugging Face
        
    Returns:
        tuple: (modelo, tokenizador)
    """
    # Cargar el tokenizador y el modelo
    tokenizador = AutoTokenizer.from_pretrained(modelo_id)
    modelo = AutoModelForCausalLM.from_pretrained(modelo_id)
    
    # Si el tokenizador no tiene token de padding, añadirlo
    if not hasattr(tokenizador, 'pad_token') or tokenizador.pad_token is None:
        if hasattr(tokenizador, 'eos_token'):
            tokenizador.pad_token = tokenizador.eos_token
        else:
            tokenizador.add_special_tokens({'pad_token': '[PAD]'})
            modelo.resize_token_embeddings(len(tokenizador))
    
    # Mover modelo a GPU si está disponible
    dispositivo = verificar_dispositivo()
    modelo.to(dispositivo)
    
    return modelo, tokenizador

def verificar_dispositivo():
    """
    Verifica y devuelve el dispositivo disponible (GPU o CPU).
    
    Returns:
        torch.device: Dispositivo a utilizar
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    """
    Preprocesa el texto de entrada para pasarlo al modelo.
    
    Args:
        texto (str): Texto de entrada
        tokenizador: Tokenizador del modelo
        longitud_maxima (int): Longitud máxima de la secuencia
    
    Returns:
        torch.Tensor: Tensor de entrada para el modelo
    """
    # Tokenizar el texto
    tokens = tokenizador(
        texto,
        padding=True,
        truncation=True,
        max_length=longitud_maxima,
        return_tensors='pt'
    )
    
    # Pasar al dispositivo correspondiente
    dispositivo = verificar_dispositivo()
    for key in tokens:
        tokens[key] = tokens[key].to(dispositivo)
    
    return tokens

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera una respuesta basada en la entrada procesada.
    
    Args:
        modelo: Modelo de lenguaje
        entrada_procesada: Tokens de entrada procesados
        tokenizador: Tokenizador del modelo
        parametros_generacion (dict): Parámetros para controlar la generación
        
    Returns:
        str: Respuesta generada
    """
    # Configurar parámetros por defecto para la generación
    if parametros_generacion is None:
        parametros_generacion = {
            'max_new_tokens': 150,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'no_repeat_ngram_size': 2,
            'do_sample': True,
            'early_stopping': True,
            'pad_token_id': tokenizador.pad_token_id if hasattr(tokenizador, 'pad_token_id') else tokenizador.eos_token_id,
            'eos_token_id': tokenizador.eos_token_id
        }
    
    # Generar texto usando el modelo
    with torch.no_grad():
        salida_ids = modelo.generate(
            input_ids=entrada_procesada['input_ids'],
            attention_mask=entrada_procesada.get('attention_mask', None),
            **parametros_generacion
        )
    
    # Decodificar la salida y limpiar la respuesta
    respuesta_cruda = tokenizador.decode(salida_ids[0], skip_special_tokens=True)
    
    # Limpiar la respuesta - Quitar el texto de entrada si está contenido en la respuesta
    entrada_decodificada = tokenizador.decode(entrada_procesada['input_ids'][0], skip_special_tokens=True)
    if respuesta_cruda.startswith(entrada_decodificada):
        respuesta_limpia = respuesta_cruda[len(entrada_decodificada):].strip()
    else:
        respuesta_limpia = respuesta_cruda.strip()
    
    return respuesta_limpia

class GestorContexto:
    """
    Clase para gestionar el contexto de una conversación con el chatbot.
    """
    
    def __init__(self, longitud_maxima=1024, formato_mensaje=None):
        """
        Inicializa el gestor de contexto.
        
        Args:
            longitud_maxima (int): Número máximo de tokens a mantener en el contexto
            formato_mensaje (callable): Función para formatear mensajes (por defecto, None)
        """
        self.historial = []
        self.longitud_maxima = longitud_maxima
        self.formato_mensaje = formato_mensaje or self._formato_predeterminado
        
    def _formato_predeterminado(self, rol, contenido):
        """
        Formato predeterminado para mensajes.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
            
        Returns:
            str: Mensaje formateado
        """
        prefijos = {
            'sistema': '### Instrucciones:\n',
            'usuario': '### Usuario:\n',
            'asistente': '### Asistente:\n'
        }
        
        # Asegurar que el rol existe en los prefijos
        if rol not in prefijos:
            raise ValueError(f"Rol '{rol}' no válido. Debe ser 'sistema', 'usuario' o 'asistente'")
        
        # Formatear el mensaje con el prefijo adecuado
        mensaje_formateado = f"{prefijos[rol]}{contenido}\n"
        return mensaje_formateado
    
    def agregar_mensaje(self, rol, contenido):
        """
        Agrega un mensaje al historial de conversación.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
        """
        # Formatear el mensaje con el formato establecido
        mensaje_formateado = self.formato_mensaje(rol, contenido)
        
        # Guardar tanto el mensaje formateado como los metadatos
        self.historial.append({
            'rol': rol,
            'contenido': contenido,
            'mensaje_formateado': mensaje_formateado
        })
    
    def construir_prompt_completo(self):
        """
        Construye un prompt completo basado en el historial.
        
        Returns:
            str: Prompt completo para el modelo
        """
        # Combinar todos los mensajes formateados en un solo prompt
        prompt_completo = ""
        
        for mensaje in self.historial:
            prompt_completo += mensaje['mensaje_formateado']
        
        # Añadir un separador para la próxima respuesta del asistente
        prompt_completo += "### Asistente:\n"
        
        return prompt_completo
    
    def truncar_historial(self, tokenizador):
        """
        Trunca el historial si excede la longitud máxima.
        
        Args:
            tokenizador: Tokenizador del modelo
        """
        # No hacer nada si no hay suficientes mensajes
        if len(self.historial) <= 2:  # Mantener al menos el mensaje del sistema y el último del usuario
            return
        
        # Construir el prompt actual
        prompt_actual = self.construir_prompt_completo()
        
        # Contar tokens
        tokens = tokenizador(prompt_actual, return_length=True, add_special_tokens=True)
        longitud_tokens = tokens['length'][0]
        
        # Si la longitud es menor que el máximo, no hacer nada
        if longitud_tokens <= self.longitud_maxima:
            return
        
        # Estrategia: Mantener siempre el mensaje del sistema (si existe) y los últimos 
        # mensajes hasta cumplir con la longitud máxima
        
        # Verificar si el primer mensaje es del sistema
        tiene_sistema = len(self.historial) > 0 and self.historial[0]['rol'] == 'sistema'
        
        # Comenzar eliminando mensajes antiguos (excepto el del sistema)
        indice_inicio = 1 if tiene_sistema else 0
        
        while longitud_tokens > self.longitud_maxima and indice_inicio < len(self.historial) - 1:
            # Eliminar el mensaje más antiguo (después del sistema y antes del último)
            mensaje_eliminado = self.historial.pop(indice_inicio)
            
            # Reconstruir el prompt y calcular nuevamente la longitud
            prompt_actual = self.construir_prompt_completo()
            tokens = tokenizador(prompt_actual, return_length=True, add_special_tokens=True)
            longitud_tokens = tokens['length'][0]

# Clase principal del chatbot
class Chatbot:
    """
    Implementación de chatbot con manejo de contexto.
    """
    
    def __init__(self, modelo_id, instrucciones_sistema=None):
        """
        Inicializa el chatbot.
        
        Args:
            modelo_id (str): Identificador del modelo en Hugging Face
            instrucciones_sistema (str): Instrucciones de comportamiento del sistema
        """
        self.modelo, self.tokenizador = cargar_modelo(modelo_id)
        self.dispositivo = verificar_dispositivo()
        self.gestor_contexto = GestorContexto()
        
        # Inicializar el contexto con instrucciones del sistema
        if instrucciones_sistema:
            self.gestor_contexto.agregar_mensaje('sistema', instrucciones_sistema)
    
    def responder(self, mensaje_usuario, parametros_generacion=None):
        """
        Genera una respuesta al mensaje del usuario.
        
        Args:
            mensaje_usuario (str): Mensaje del usuario
            parametros_generacion (dict): Parámetros para la generación
            
        Returns:
            str: Respuesta del chatbot
        """
        # 1. Agregar mensaje del usuario al contexto
        self.gestor_contexto.agregar_mensaje('usuario', mensaje_usuario)
        
        # 2. Truncar el historial si es necesario
        self.gestor_contexto.truncar_historial(self.tokenizador)
        
        # 3. Construir el prompt completo
        prompt_completo = self.gestor_contexto.construir_prompt_completo()
        
        # 4. Preprocesar la entrada
        entrada_procesada = preprocesar_entrada(prompt_completo, self.tokenizador)
        
        # 5. Generar la respuesta
        respuesta = generar_respuesta(self.modelo, entrada_procesada, self.tokenizador, parametros_generacion)
        
        # 6. Agregar respuesta al contexto
        self.gestor_contexto.agregar_mensaje('asistente', respuesta)
        
        # 7. Devolver la respuesta
        return respuesta

# Prueba del sistema
def prueba_conversacion():
    # Crear una instancia del chatbot
    instrucciones = """
    Eres un asistente IA amable y servicial. Tu objetivo es proporcionar respuestas útiles,
    informativas y éticas. Evita dar información falsa o engañosa. Si no sabes algo, admítelo
    en lugar de inventar. Mantén un tono amigable y profesional.
    """
    
    # Usar un modelo pequeño para pruebas, en producción debería usarse un modelo más potente
    chatbot = Chatbot("gpt2", instrucciones_sistema=instrucciones)
    
    # Simular una conversación de varios turnos
    conversacion = [
        "Hola, ¿cómo estás?",
        "¿Puedes explicarme qué es la inteligencia artificial?",
        "Dame un ejemplo de aplicación de IA en la vida cotidiana",
        "¿Estas aplicaciones tienen algún riesgo?",
        "Gracias por la información"
    ]
    
    # Ejecutar la conversación
    for mensaje in conversacion:
        print(f"\nUsuario: {mensaje}")
        respuesta = chatbot.responder(mensaje)
        print(f"Asistente: {respuesta}")
    
    return chatbot

# Ejecutar el ejemplo si se ejecuta el script directamente
if __name__ == "__main__":
    prueba_conversacion()
```




## Ejercicio 4: Optimización del Modelo para Recursos Limitados

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import time
import gc
import psutil
import os
from typing import Dict, Tuple, Optional, List, Any

def configurar_cuantizacion(bits=4):
    """
    Configura los parámetros para la cuantización del modelo.

    Args:
        bits (int): Bits para cuantización (4 u 8)

    Returns:
        BitsAndBytesConfig: Configuración de cuantización
    """
    # Implementación de la configuración de cuantización
    config_cuantizacion = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.float16 if bits == 4 else None,
        bnb_4bit_use_double_quant=True if bits == 4 else False,
        bnb_4bit_quant_type="nf4" if bits == 4 else None,  # "nf4" o "fp4"
    )

    return config_cuantizacion

def cargar_modelo_optimizado(nombre_modelo, optimizaciones=None):
    """
    Carga un modelo con optimizaciones aplicadas.

    Args:
        nombre_modelo (str): Identificador del modelo
        optimizaciones (dict): Diccionario con flags para las optimizaciones

    Returns:
        tuple: (modelo, tokenizador)
    """
    if optimizaciones is None:
        optimizaciones = {
            "cuantizacion": True,
            "bits": 4,
            "offload_cpu": False,
            "flash_attention": True
        }

    # Configurar opciones de carga de modelo
    kwargs = {}

    # Aplicar cuantización si está habilitada
    if optimizaciones.get("cuantizacion", False):
        config_cuantizacion = configurar_cuantizacion(optimizaciones.get("bits", 4))
        kwargs.update({"quantization_config": config_cuantizacion})

    # Configurar dispositivo y offload si se especifica
    device_map = "auto"
    if optimizaciones.get("offload_cpu", False):
        kwargs.update({
            "device_map": device_map,
            "offload_folder": "offload_folder",
        })
    else:
        kwargs.update({"device_map": device_map})

    # Aplicar Flash Attention 2 si está disponible y habilitado
    if optimizaciones.get("flash_attention", False) and torch.cuda.is_available():
        kwargs.update({"attn_implementation": "flash_attention_2"})

    # Cargar el modelo con las optimizaciones configuradas
    try:
        modelo = AutoModelForCausalLM.from_pretrained(
            nombre_modelo,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        )
    except Exception as e:
        print(f"Error al cargar el modelo con optimizaciones: {e}")
        print("Intentando cargar el modelo con configuración básica...")
        modelo = AutoModelForCausalLM.from_pretrained(
            nombre_modelo,
            trust_remote_code=True
        )

    # Cargar el tokenizador
    tokenizador = AutoTokenizer.from_pretrained(
        nombre_modelo,
        trust_remote_code=True,
        padding_side="left"
    )

    # Asegurar que el tokenizador tenga un token de padding
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token

    return modelo, tokenizador

def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante para procesar secuencias largas.

    Args:
        modelo: Modelo a configurar
        window_size (int): Tamaño de la ventana de atención
    """
    # Configurar sliding window attention en el modelo
    try:
        if hasattr(modelo.config, "sliding_window"):
            # Establecer el tamaño de la ventana deslizante
            modelo.config.sliding_window = window_size
            print(f"Sliding window configurado a {window_size} tokens")
        elif hasattr(modelo.config, "attention_window"):
            # Alternativa para modelos que usan attention_window
            modelo.config.attention_window = window_size
            print(f"Attention window configurado a {window_size} tokens")
        else:
            print("Este modelo no soporta directamente sliding window attention")
    except Exception as e:
        print(f"Error al configurar sliding window: {e}")

    return modelo

def obtener_memoria_utilizada():
    """
    Obtiene el uso actual de memoria en MB.

    Returns:
        float: Memoria utilizada en MB
    """
    try:
        if torch.cuda.is_available():
            # Obtener memoria GPU
            torch.cuda.synchronize()
            memoria_asignada = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return memoria_asignada
        else:
            # Obtener memoria RAM si no hay GPU
            proceso = psutil.Process(os.getpid())
            memoria_mb = proceso.memory_info().rss / (1024 * 1024)
            return memoria_mb
    except Exception as e:
        print(f"Error al obtener memoria: {e}")
        return 0.0

def evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo):
    """
    Evalúa el rendimiento del modelo en términos de velocidad y memoria.

    Args:
        modelo: Modelo a evaluar
        tokenizador: Tokenizador del modelo
        texto_prueba (str): Texto para pruebas de rendimiento
        dispositivo: Dispositivo donde se ejecutará

    Returns:
        dict: Métricas de rendimiento
    """
    # Limpiar caché de memoria antes de la evaluación
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Medir memoria inicial
    memoria_inicial = obtener_memoria_utilizada()

    # Tokenizar el texto de prueba
    try:
        inputs = tokenizador(texto_prueba, return_tensors="pt", truncation=True, max_length=512).to(dispositivo)
        input_tokens = len(inputs["input_ids"][0])

        # Medir tiempo de inferencia
        inicio = time.time()
        with torch.no_grad():
            # No queremos incluir la generación de tokens adicionales en esta evaluación
            salida = modelo(**inputs)
        fin = time.time()

        # Calcular métricas
        tiempo_inferencia = fin - inicio
        memoria_utilizada = obtener_memoria_utilizada() - memoria_inicial
        tokens_por_segundo = input_tokens / tiempo_inferencia if tiempo_inferencia > 0 else 0

        metricas = {
            "tiempo_inferencia_segundos": tiempo_inferencia,
            "memoria_adicional_mb": memoria_utilizada,
            "tokens_por_segundo": tokens_por_segundo,
            "tokens_procesados": input_tokens
        }
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        metricas = {
            "tiempo_inferencia_segundos": 0,
            "memoria_adicional_mb": 0,
            "tokens_por_segundo": 0,
            "tokens_procesados": 0,
            "error": str(e)
        }

    return metricas

def demo_optimizaciones(nombre_modelo="facebook/opt-125m"):
    """
    Demuestra y compara diferentes configuraciones de optimización.

    Args:
        nombre_modelo (str): Nombre del modelo a utilizar (predeterminado a un modelo más pequeño)

    Returns:
        Dict[str, Any]: Resultados comparativos de las optimizaciones
    """
    # Texto de prueba para evaluación de rendimiento (reducido para evitar problemas de memoria)
    texto_prueba = """
    En el ámbito de la inteligencia artificial y el procesamiento del lenguaje natural,
    los modelos de lenguaje han experimentado avances significativos en los últimos años.
    Estos modelos, basados en arquitecturas transformer, han revolucionado tareas como
    la traducción automática, la generación de texto, la clasificación de sentimientos
    y muchas otras aplicaciones de PLN.
    """ * 5  # Reducido a 5 repeticiones para evitar problemas de memoria

    # Definir dispositivo
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {dispositivo}")
    
    # Mostrar información sobre el dispositivo
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("No se detectó GPU. Usando CPU.")
        
    print(f"Modelo seleccionado: {nombre_modelo}")

    resultados = {}

    # Intentaremos cargar diferentes configuraciones dentro de bloques try-except para evitar errores fatales
    
    # 1. Modelo base sin optimizaciones
    print("\n1. Cargando modelo base sin optimizaciones...")
    try:
        modelo_base, tokenizador = cargar_modelo_optimizado(
            nombre_modelo,
            optimizaciones={
                "cuantizacion": False,
                "flash_attention": False,
                "offload_cpu": False
            }
        )
        print("Evaluando rendimiento del modelo base...")
        resultados["base"] = evaluar_rendimiento(modelo_base, tokenizador, texto_prueba, dispositivo)
        del modelo_base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error en evaluación del modelo base: {e}")
        resultados["base"] = {"error": str(e)}

    # 2. Modelo con cuantización de 4 bits
    print("\n2. Cargando modelo con cuantización de 4 bits...")
    try:
        modelo_cuantizado, tokenizador = cargar_modelo_optimizado(
            nombre_modelo,
            optimizaciones={
                "cuantizacion": True,
                "bits": 4,
                "flash_attention": False,
                "offload_cpu": False
            }
        )
        print("Evaluando rendimiento del modelo cuantizado...")
        resultados["cuantizado_4bits"] = evaluar_rendimiento(modelo_cuantizado, tokenizador, texto_prueba, dispositivo)
        del modelo_cuantizado
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error en evaluación del modelo cuantizado: {e}")
        resultados["cuantizado_4bits"] = {"error": str(e)}

    # 3. Modelo con sliding window attention
    print("\n3. Cargando modelo con sliding window attention...")
    try:
        modelo_sliding, tokenizador = cargar_modelo_optimizado(
            nombre_modelo,
            optimizaciones={
                "cuantizacion": False,
                "flash_attention": False,
                "offload_cpu": False
            }
        )
        aplicar_sliding_window(modelo_sliding, window_size=256)  # Ventana más pequeña para evitar problemas
        print("Evaluando rendimiento del modelo con sliding window...")
        resultados["sliding_window"] = evaluar_rendimiento(modelo_sliding, tokenizador, texto_prueba, dispositivo)
        del modelo_sliding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error en evaluación del modelo con sliding window: {e}")
        resultados["sliding_window"] = {"error": str(e)}

    # 4. Modelo con todas las optimizaciones
    print("\n4. Cargando modelo con todas las optimizaciones...")
    try:
        modelo_completo, tokenizador = cargar_modelo_optimizado(
            nombre_modelo,
            optimizaciones={
                "cuantizacion": True,
                "bits": 4,
                "flash_attention": True,
                "offload_cpu": dispositivo == "cuda"  # Solo activar offload si hay GPU
            }
        )
        aplicar_sliding_window(modelo_completo, window_size=256)
        print("Evaluando rendimiento del modelo con todas las optimizaciones...")
        resultados["optimizacion_completa"] = evaluar_rendimiento(modelo_completo, tokenizador, texto_prueba, dispositivo)
        del modelo_completo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error en evaluación del modelo con todas las optimizaciones: {e}")
        resultados["optimizacion_completa"] = {"error": str(e)}

    # Mostrar resultados disponibles
    print("\n=== COMPARACIÓN DE OPTIMIZACIONES ===")
    print(f"{'Configuración':<25} {'Tiempo (s)':<12} {'Memoria (MB)':<12} {'Tokens/s':<12} {'Estado':<12}")
    print("-" * 80)

    for config, metricas in resultados.items():
        if "error" in metricas:
            print(f"{config:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'Error':<12}")
        else:
            print(f"{config:<25} {metricas.get('tiempo_inferencia_segundos', 0):<12.4f} {metricas.get('memoria_adicional_mb', 0):<12.2f} {metricas.get('tokens_por_segundo', 0):<12.2f} {'OK':<12}")

    # Calcular porcentajes de mejora respecto al modelo base si es posible
    if "base" in resultados and "error" not in resultados["base"]:
        base_tiempo = resultados["base"].get("tiempo_inferencia_segundos", 0)
        base_memoria = resultados["base"].get("memoria_adicional_mb", 0)
        base_tokens_s = resultados["base"].get("tokens_por_segundo", 0)

        if base_tiempo > 0 and base_memoria > 0 and base_tokens_s > 0:
            print("\n=== MEJORAS RESPECTO AL MODELO BASE (%) ===")
            print(f"{'Configuración':<25} {'Tiempo':<12} {'Memoria':<12} {'Velocidad':<12}")
            print("-" * 70)

            for config, metricas in resultados.items():
                if config != "base" and "error" not in metricas:
                    tiempo = metricas.get("tiempo_inferencia_segundos", 0)
                    memoria = metricas.get("memoria_adicional_mb", 0)
                    tokens_s = metricas.get("tokens_por_segundo", 0)
                    
                    if tiempo > 0 and memoria > 0 and tokens_s > 0:
                        tiempo_mejora = ((base_tiempo - tiempo) / base_tiempo) * 100
                        memoria_mejora = ((base_memoria - memoria) / base_memoria) * 100
                        velocidad_mejora = ((tokens_s - base_tokens_s) / base_tokens_s) * 100

                        print(f"{config:<25} {tiempo_mejora:<12.2f}% {memoria_mejora:<12.2f}% {velocidad_mejora:<12.2f}%")

    return resultados

# Función principal para ejecutar la demostración
if __name__ == "__main__":
    print("Iniciando demostración de optimizaciones de modelo...")
    try:
        # Usar un modelo más pequeño para entornos con recursos limitados
        resultados = demo_optimizaciones(nombre_modelo="facebook/opt-125m")  # Modelo más pequeño que opt-350m
        print("\nDemostración completada exitosamente!")
    except Exception as e:
        print(f"Error en la demostración: {e}")
```
Esto fue lo que se mostró
![image](https://github.com/user-attachments/assets/493147f9-5faa-43a3-9079-26f0638fb43d)


## Ejercicio 5: Personalización del Chatbot y Despliegue

```python
!pip install transformers peft gradio bitsandbytes accelerate torch
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os

def configurar_peft(modelo, r=8, lora_alpha=32):
    """
    Configura el modelo para fine-tuning con PEFT/LoRA.
    
    Args:
        modelo: Modelo base
        r (int): Rango de adaptadores LoRA
        lora_alpha (int): Escala alpha para LoRA
    
    Returns:
        modelo: Modelo adaptado para fine-tuning
    """
    # Crear la configuración de LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Módulos comunes para modelos transformer
    )
    
    # Aplicar PEFT al modelo
    modelo_peft = get_peft_model(modelo, peft_config)
    
    # Imprimir información sobre el modelo y los parámetros entrenables
    modelo_peft.print_trainable_parameters()
    
    return modelo_peft

def guardar_modelo(modelo, tokenizador, ruta):
    """
    Guarda el modelo y tokenizador en una ruta específica.
    
    Args:
        modelo: Modelo a guardar
        tokenizador: Tokenizador del modelo
        ruta (str): Ruta donde guardar
    """
    # Crear el directorio si no existe
    os.makedirs(ruta, exist_ok=True)
    
    # Guardar el modelo
    modelo.save_pretrained(ruta)
    
    # Guardar el tokenizador
    tokenizador.save_pretrained(ruta)
    
    print(f"Modelo y tokenizador guardados en: {ruta}")

def cargar_modelo_personalizado(ruta, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Carga un modelo personalizado desde una ruta específica.
    
    Args:
        ruta (str): Ruta del modelo
        device (str): Dispositivo donde cargar el modelo
        
    Returns:
        tuple: (modelo, tokenizador)
    """
    # Configuración para cargar modelos grandes en memoria limitada
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None
    
    # Cargar el tokenizador
    tokenizador = AutoTokenizer.from_pretrained(ruta)
    
    # Cargar el modelo base
    if os.path.exists(os.path.join(ruta, "adapter_config.json")):
        # Es un modelo PEFT, necesitamos cargar el modelo base primero
        base_model_name = None
        with open(os.path.join(ruta, "adapter_config.json"), 'r') as f:
            import json
            config = json.load(f)
            if 'base_model_name_or_path' in config:
                base_model_name = config['base_model_name_or_path']
        
        if base_model_name:
            # Cargar el modelo base
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            
            # Cargar el modelo PEFT
            modelo = PeftModel.from_pretrained(base_model, ruta)
        else:
            raise ValueError("No se pudo determinar el modelo base del adaptador PEFT")
    else:
        # Es un modelo completo
        modelo = AutoModelForCausalLM.from_pretrained(
            ruta,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    
    # Configuración del tokenizador
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token
    
    print(f"Modelo cargado en: {device}")
    return modelo, tokenizador

class Chatbot:
    def __init__(self, modelo, tokenizador, max_length=512):
        self.modelo = modelo
        self.tokenizador = tokenizador
        self.max_length = max_length
        self.history = []
    
    def generar_respuesta(self, mensaje, temperatura=0.7, top_p=0.9, top_k=50):
        """
        Genera una respuesta a partir del mensaje del usuario.
        
        Args:
            mensaje (str): Mensaje del usuario
            temperatura (float): Temperatura para la generación
            top_p (float): Valor de top_p para la generación
            top_k (int): Valor de top_k para la generación
            
        Returns:
            str: Respuesta generada
        """
        # Preparar el contexto con el historial de conversación
        prompt = self._prepare_prompt(mensaje)
        
        # Tokenizar
        inputs = self.tokenizador(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.modelo.device)
        attention_mask = inputs["attention_mask"].to(self.modelo.device)
        
        # Generar respuesta
        with torch.no_grad():
            output = self.modelo.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=temperatura,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizador.pad_token_id,
            )
        
        # Decodificar la respuesta
        generated_text = self.tokenizador.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Actualizar historial
        self.history.append((mensaje, generated_text))
        if len(self.history) > 10:  # Mantener solo las últimas 10 interacciones
            self.history.pop(0)
        
        return generated_text
    
    def _prepare_prompt(self, mensaje):
        """
        Prepara el prompt completo con el historial de la conversación.
        
        Args:
            mensaje (str): Mensaje actual del usuario
            
        Returns:
            str: Prompt completo
        """
        # Template simple para la conversación
        prompt = ""
        
        # Añadir historial
        for user_msg, bot_msg in self.history:
            prompt += f"Usuario: {user_msg}\nAsistente: {bot_msg}\n\n"
        
        # Añadir mensaje actual
        prompt += f"Usuario: {mensaje}\nAsistente:"
        
        return prompt
    
    def reset_history(self):
        """Reinicia el historial de conversación"""
        self.history = []
        return "Historial de conversación borrado."

# Interfaz web simple con Gradio
def crear_interfaz_web(chatbot):
    """
    Crea una interfaz web simple para el chatbot usando Gradio.
    
    Args:
        chatbot: Instancia del chatbot
        
    Returns:
        gr.Interface: Interfaz de Gradio
    """
    # Función de callback para procesar entradas
    def responder(mensaje, historia=None, temperatura=0.7, top_p=0.9, top_k=50):
        if historia is None:
            historia = []
        
        # Generar respuesta
        respuesta = chatbot.generar_respuesta(mensaje, temperatura, top_p, top_k)
        
        # Actualizar la historia para la interfaz
        historia.append((mensaje, respuesta))
        
        return "", historia
    
    def reiniciar_chat():
        mensaje = chatbot.reset_history()
        return [], mensaje
    
    # Definir los componentes de la interfaz
    with gr.Blocks() as interfaz:
        gr.Markdown("# Chatbot personalizado")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot_component = gr.Chatbot(label="Conversación")
                mensaje_usuario = gr.Textbox(
                    placeholder="Escribe tu mensaje aquí...",
                    label="Mensaje",
                    lines=2
                )
                
                with gr.Row():
                    enviar_btn = gr.Button("Enviar", variant="primary")
                    reiniciar_btn = gr.Button("Reiniciar conversación")
            
            with gr.Column(scale=1):
                gr.Markdown("### Parámetros de generación")
                temperatura = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                    label="Temperatura", info="Valores más altos = respuestas más creativas"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                    label="Top-p", info="Filtrado por probabilidad acumulada"
                )
                top_k = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1,
                    label="Top-k", info="Filtrado por rank"
                )
        
        # Configurar eventos
        enviar_btn.click(
            responder,
            inputs=[mensaje_usuario, chatbot_component, temperatura, top_p, top_k],
            outputs=[mensaje_usuario, chatbot_component]
        )
        mensaje_usuario.submit(
            responder,
            inputs=[mensaje_usuario, chatbot_component, temperatura, top_p, top_k],
            outputs=[mensaje_usuario, chatbot_component]
        )
        reiniciar_btn.click(reiniciar_chat, outputs=[chatbot_component, gr.Textbox(label="Estado")])
        
    return interfaz

# Función para crear un modelo de ejemplo (usado solo para testing en caso de no tener uno fine-tuned)
def crear_modelo_ejemplo():
    """
    Crea un modelo pequeño para pruebas.
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    model_name = "facebook/opt-125m"  # Modelo pequeño para pruebas
    
    tokenizador = AutoTokenizer.from_pretrained(model_name)
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token
    
    modelo = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    return modelo, tokenizador

# Función principal para el despliegue
def main_despliegue(model_path=None):
    """
    Función principal para el despliegue del chatbot.
    
    Args:
        model_path (str, optional): Ruta al modelo personalizado.
    """
    # Cargar el modelo personalizado o crear uno de ejemplo
    if model_path and os.path.exists(model_path):
        print(f"Cargando modelo personalizado desde: {model_path}")
        modelo, tokenizador = cargar_modelo_personalizado(model_path)
    else:
        print("Modelo personalizado no encontrado. Usando modelo de ejemplo para demonstración.")
        modelo, tokenizador = crear_modelo_ejemplo()
    
    # Crear instancia del chatbot
    chatbot = Chatbot(modelo, tokenizador)
    
    # Crear y lanzar la interfaz web
    interfaz = crear_interfaz_web(chatbot)
    
    # Configurar parámetros para el despliegue
    interfaz.launch(
        share=True,  # Crear un enlace público (útil para Colab)
        debug=True,
        height=700,
    )

# Función para entrenar un modelo en datos personalizados (ejemplo simplificado)
def entrenar_modelo(modelo, tokenizador, dataset, output_dir, epochs=3, batch_size=4):
    """
    Entrena un modelo en un dataset personalizado.
    
    Args:
        modelo: Modelo base con PEFT aplicado
        tokenizador: Tokenizador del modelo
        dataset: Dataset de entrenamiento
        output_dir (str): Directorio para guardar el modelo
        epochs (int): Número de épocas de entrenamiento
        batch_size (int): Tamaño del batch
        
    Returns:
        modelo: Modelo entrenado
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=3e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Configurar el data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizador,
        mlm=False,  # No usamos masked language modeling para chatbots
    )
    
    # Configurar el trainer
    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Guardar el modelo
    modelo.save_pretrained(output_dir)
    tokenizador.save_pretrained(output_dir)
    
    return modelo

if __name__ == "__main__":
    # Ejemplo de cómo usar este script:
    # 1. Cargar/crear un modelo y aplicar PEFT
    # modelo_base, tokenizador = crear_modelo_ejemplo()
    # modelo_peft = configurar_peft(modelo_base)
    
    # 2. Entrenar el modelo (necesitarías preparar tu dataset)
    # entrenar_modelo(modelo_peft, tokenizador, dataset, "ruta_salida")
    
    # 3. Desplegar el modelo entrenado
    # main_despliegue("ruta_modelo_personalizado")
    
    # Para ejemplo rápido, simplemente desplegar un modelo de ejemplo
    main_despliegue()
```

Lo que se mostró al ejecutar en colab:
![image](https://github.com/user-attachments/assets/b5b1b0af-8aaf-4692-9530-a7cc58ae297a)


## Preguntas teoricas

Para ejecutar el proyecto completo, sigue estos pasos:

1. **¿Cuáles son las diferencias fundamentales entre los modelos encoder-only, decoder-only y encoder-decoder en el contexto de los chatbots conversacionales? Explique qué tipo de modelo sería más adecuado para cada caso de uso y por qué.**

Los modelos de lenguaje se pueden clasificar en tres arquitecturas principales según su estructura:
Los modelos encoder-only (como BERT) se especializan en comprender el contexto bidireccional del texto. Procesan todo el texto de entrada simultáneamente, permitiéndoles "ver" palabras tanto anteriores como posteriores. Esto los hace excelentes para tareas de comprensión del lenguaje como clasificación de texto, análisis de sentimiento, extracción de información y respuesta a preguntas sobre un texto específico.Sin embargo, no están diseñados para generar texto nuevo de forma autónoma.

Los modelos decoder-only (como GPT) generan texto de manera autoregresiva, prediciendo el siguiente token basándose únicamente en los tokens anteriores. Esta arquitectura unidireccional los hace ideales para tareas generativas como completado de texto, chatbots conversacionales y cualquier aplicación donde se requiera producir texto coherente y contextual. Su capacidad de generar secuencias de manera fluida los convierte en la opción preferida para asistentes de IA conversacionales.

Los modelos encoder-decoder (como T5 o BART) combinan ambas arquitecturas: un encoder procesa todo el texto de entrada bidireccional, y luego un decoder genera una salida basada en esa representación. Esta estructura los hace versátiles para tareas de transformación de texto como traducción, resumen, parafraseo y reformulación de consultas, donde hay una clara distinción entre entrada y salida.
Para chatbots conversacionales, los modelos decoder-only suelen ser más adecuados porque:

Mantienen mejor el estado conversacional a lo largo de múltiples turnos
Generan respuestas más naturales y coherentes
Pueden producir texto variado sin limitarse a estructuras predefinidas
Modelos como LLaMA, GPT-4 y Claude utilizan esta arquitectura por estas ventajas

2. **Explique el concepto de "temperatura" en la generación de texto con LLMs. ¿Cómo afecta al comportamiento del chatbot y qué consideraciones debemos tener al ajustar este parámetro para diferentes aplicaciones?**

La temperatura es un hiperparámetro que controla la aleatoriedad o determinismo en la generación de texto. Técnicamente, antes de seleccionar el siguiente token, el modelo aplica una distribución de probabilidad sobre todos los tokens posibles. La temperatura modifica esta distribución:

Temperatura baja (cerca de 0): Hace que el modelo elija consistentemente los tokens más probables, resultando en respuestas más deterministas, conservadoras y predecibles. El modelo tenderá a repetir patrones comunes y evitará respuestas creativas pero arriesgadas.
Temperatura alta (0.7-1.0): Distribuye la probabilidad más uniformemente entre diversos tokens, produciendo respuestas más variadas, creativas e impredecibles. El modelo experimenta más, pero también aumenta el riesgo de incoherencias.

La temperatura afecta directamente:

La coherencia vs creatividad de las respuestas
La diversidad del vocabulario utilizado
La probabilidad de que el modelo "tome riesgos" en sus respuestas

Consideraciones para ajustar la temperatura según la aplicación:

Para sistemas de asistencia técnica, documentación o aplicaciones médicas: usar temperatura baja (0.1-0.3) para maximizar precisión y consistencia
Para escritura creativa, brainstorming o entretenimiento: usar temperatura media-alta (0.7-1.0)
Para asistentes generales: un balance intermedio (0.5-0.7) suele funcionar bien
En sistemas críticos: considerar implementar varios pases con diferentes temperaturas y un mecanismo de verificación

3. **Describa las técnicas principales para reducir el problema de "alucinaciones" en chatbots basados en LLMs. ¿Qué estrategias podemos implementar a nivel de inferencia y a nivel de prompt engineering para mejorar la precisión factual de las respuestas?**

Las alucinaciones ocurren cuando los LLMs generan información incorrecta pero presentada con confianza. Para reducirlas, podemos implementar:
Estrategias a nivel de inferencia:

Retrieval-Augmented Generation (RAG): Complementar el conocimiento del modelo con información recuperada de fuentes verificables en tiempo real.
Verificación factual automatizada: Implementar sistemas que comprueben declaraciones contra bases de conocimiento verificadas.
Cadenas de pensamiento (Chain-of-Thought): Hacer que el modelo explique su razonamiento paso a paso antes de dar una respuesta final.
Técnicas de consenso: Ejecutar múltiples inferencias con diferentes parámetros y buscar consistencia entre respuestas.
Reducción de temperatura: Usar temperaturas más bajas cuando la precisión factual sea crítica.
Post-procesamiento con modelos especializados: Usar modelos secundarios para verificar la salida del modelo principal.

Estrategias a nivel de prompt engineering:

Instrucciones explícitas de precisión: Incluir directivas como "proporciona solo información verificable" o "indica claramente cuando no estés seguro".
Ejemplos few-shot: Demostrar en el prompt cómo manejar la incertidumbre adecuadamente.
Prompts estructurados: Guiar al modelo para que separe hechos de opiniones y especulaciones.
Técnica de reflexión: Pedir al modelo que evalúe su propia respuesta antes de finalizarla.
Contextual grounding: Proporcionar contexto suficiente y relevante en el prompt.
Instrucciones de citación: Pedir al modelo que mencione el origen de la información cuando sea posible.
