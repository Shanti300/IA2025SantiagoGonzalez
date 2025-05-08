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
        if rol == "sistema":
            return f"<s>[SYSTEM]\n{contenido}\n[/SYSTEM]\n\n"
        elif rol == "usuario":
            return f"[USER] {contenido} [/USER]\n"
        elif rol == "asistente":
            return f"[ASSISTANT] {contenido} [/ASSISTANT]\n"
        else:
            return f"[{rol.upper()}] {contenido} [{rol.upper()}]\n"
    
    def agregar_mensaje(self, rol, contenido):
        """
        Agrega un mensaje al historial de conversación.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
        """
        mensaje_formateado = self.formato_mensaje(rol, contenido)
        self.historial.append({"rol": rol, "contenido": contenido, "formateado": mensaje_formateado})
    
    def construir_prompt_completo(self):
        """
        Construye un prompt completo basado en el historial.
        
        Returns:
            str: Prompt completo para el modelo
        """
        return "".join([m["formateado"] for m in self.historial]) + "[ASSISTANT] "
    
    def truncar_historial(self, tokenizador):
        """
        Trunca el historial si excede la longitud máxima.
        
        Args:
            tokenizador: Tokenizador del modelo
        """
        # Preservar siempre el mensaje del sistema
        sistema_msgs = [msg for msg in self.historial if msg["rol"] == "sistema"]
        otros_msgs = [msg for msg in self.historial if msg["rol"] != "sistema"]
        
        # Verificar longitud actual
        prompt_actual = self.construir_prompt_completo()
        tokens_actuales = len(tokenizador.encode(prompt_actual))
        
        # Si la longitud es aceptable, no hacer nada
        if tokens_actuales <= self.longitud_maxima:
            return
        
        # Calcular cuántos tokens debemos eliminar
        tokens_a_eliminar = tokens_actuales - self.longitud_maxima + 100  # Margen de seguridad
        
        # Eliminar mensajes antiguos hasta cumplir con la longitud máxima
        while tokens_a_eliminar > 0 and otros_msgs:
            # Eliminar el mensaje más antiguo (excepto sistema)
            mensaje_eliminado = otros_msgs.pop(0)
            tokens_mensaje = len(tokenizador.encode(mensaje_eliminado["formateado"]))
            tokens_a_eliminar -= tokens_mensaje
        
        # Reconstruir el historial
        self.historial = sistema_msgs + otros_msgs

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
            self.gestor_contexto.agregar_mensaje("sistema", instrucciones_sistema)
    
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
        self.gestor_contexto.agregar_mensaje("usuario", mensaje_usuario)
        
        # 2. Construir el prompt completo
        prompt_completo = self.gestor_contexto.construir_prompt_completo()
        
        # 3. Preprocesar la entrada
        entrada_procesada = preprocesar_entrada(
            prompt_completo, 
            self.tokenizador, 
            longitud_maxima=2048,
            dispositivo=self.dispositivo
        )
        
        # 4. Generar la respuesta
        respuesta = generar_respuesta(
            self.modelo, 
            entrada_procesada, 
            self.tokenizador, 
            parametros_generacion
        )
        
        # 5. Agregar respuesta al contexto
        self.gestor_contexto.agregar_mensaje("asistente", respuesta)
        
        # 6. Truncar el historial si es necesario
        self.gestor_contexto.truncar_historial(self.tokenizador)
        
        # 7. Devolver la respuesta
        return respuesta

# Prueba del sistema
def prueba_conversacion():
    # Crear una instancia del chatbot
    chatbot = Chatbot(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Eres un asistente amable, servicial y conciso. Respondes con información precisa y útil."
    )
    
    # Simular una conversación de varios turnos
    preguntas = [
        "¿Qué es la inteligencia artificial?",
        "¿Cuáles son sus aplicaciones principales?",
        "¿Y qué riesgos existen?",
        "Gracias por la información"
    ]
    
    # Realizar la conversación
    for pregunta in preguntas:
        print(f"\nUsuario: {pregunta}")
        respuesta = chatbot.responder(pregunta)
        print(f"Asistente: {respuesta}")
```

## Ejercicio 4: Optimización del Modelo para Recursos Limitados

```python
from transformers import BitsAndBytesConfig
import torch.nn as nn
import time
import psutil
import gc

def configurar_cuantizacion(bits=4):
    """
    Configura los parámetros para la cuantización del modelo.
    
    Args:
        bits (int): Bits para cuantización (4 u 8)
    
    Returns:
        BitsAndBytesConfig: Configuración de cuantización
    """
    if bits not in [4, 8]:
        raise ValueError("La cuantización solo admite 4 u 8 bits")
    
    config_cuantizacion = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
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
    
    # Cargar tokenizador
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    
    # Configurar opciones de carga del modelo
    model_kwargs = {}
    
    # Aplicar cuantización si está habilitada
    if optimizaciones.get("cuantizacion", False):
        bits = optimizaciones.get("bits", 4)
        config_cuant = configurar_cuantizacion(bits)
        model_kwargs["quantization_config"] = config_cuant
    
    # Configurar dtype
    model_kwargs["torch_dtype"] = torch.float16
    
    # Configurar offload a CPU si está habilitado
    if optimizaciones.get("offload_cpu", False):
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_folder"] = "offload_folder"
    else:
        model_kwargs["device_map"] = "auto"
    
    # Activar Flash Attention 2 si está disponible y habilitado
    if optimizaciones.get("flash_attention", False):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Cargar el modelo con las optimizaciones configuradas
    print(f"Cargando modelo optimizado: {nombre_modelo}")
    print(f"Optimizaciones aplicadas: {optimizaciones}")
    
    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        **model_kwargs
    )
    
    # Establecer en modo evaluación
    modelo.eval()
    
    return modelo, tokenizador

def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante para procesar secuencias largas.
    
    Args:
        modelo: Modelo a configurar
        window_size (int): Tamaño de la ventana de atención
    """
    # Buscar las configuraciones de atención en el modelo
    for nombre, modulo in modelo.named_modules():
        # Buscar módulos de atención en el modelo
        if "attention" in nombre.lower() and hasattr(modulo, "window_size"):
            print(f"Configurando sliding window en {nombre}")
            modulo.window_size = window_size
    
    # Alternativamente, configurar a nivel global si el modelo lo soporta
    if hasattr(modelo.config, "sliding_window"):
        modelo.config.sliding_window = window_size
        print(f"Configurado sliding window global con tamaño {window_size}")

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
    # Recoger basura y liberar memoria antes de la prueba
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Preparar la entrada
    tokens = tokenizador(texto_prueba, return_tensors="pt").to(dispositivo)
    num_tokens_entrada = tokens.input_ids.shape[1]
    
    # Medir uso de memoria antes
    if torch.cuda.is_available():
        memoria_antes = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        memoria_antes = psutil.Process().memory_info().rss / 1024**2  # MB
    
    # Medir tiempo de inferencia
    inicio = time.time()
    with torch.no_grad():
        salida = modelo.generate(
            **tokens,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    fin = time.time()
    
    # Decodificar y contar tokens generados
    texto_generado = tokenizador.decode(salida[0], skip_special_tokens=True)
    num_tokens_generados = salida.shape[1] - num_tokens_entrada
    tiempo_inferencia = fin - inicio
    
    # Medir memoria después
    if torch.cuda.is_available():
        memoria_despues = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        memoria_despues = psutil.Process().memory_info().rss / 1024**2  # MB
    
    # Calcular métricas
    metricas = {
        "tiempo_inferencia_segundos": tiempo_inferencia,
        "tokens_generados": num_tokens_generados,
        "tokens_por_segundo": num_tokens_generados / tiempo_inferencia,
        "memoria_usada_mb": memoria_despues - memoria_antes,
        "memoria_total_mb": memoria_despues
    }
    
    return metricas

# Función de demostración
def demo_optimizaciones():
    # Texto para pruebas
    texto_prueba = """
    Los modelos de lenguaje de gran escala (LLMs) están transformando la inteligencia artificial.
    Estos modelos ofrecen capacidades impresionantes en generación de texto, traducción,
    resumen y muchas otras tareas. Sin embargo, también presentan desafíos importantes
    en términos de recursos computacionales y energéticos.
    
    ¿Cuáles son las principales ventajas y desventajas de estos modelos?
    """
    
    dispositivo = verificar_dispositivo()
    modelo_base = "mistralai/Mistral-7B-Instruct-v0.2"  # Modelo para pruebas
    
    # Configuraciones a probar
    configuraciones = {
        "base": {"cuantizacion": False, "flash_attention": False},
        "cuant4": {"cuantizacion": True, "bits": 4, "flash_attention": False},
        "sliding": {"cuantizacion": False, "flash_attention": False, "sliding_window": True},
        "completo": {"cuantizacion": True, "bits": 4, "flash_attention": True, "sliding_window": True}
    }
    
    resultados = {}
    
    # Probar cada configuración
    for nombre, config in configuraciones.items():
        print(f"\n{'='*50}")
        print(f"Evaluando configuración: {nombre}")
        print(f"{'='*50}")
        
        # Cargar modelo con la configuración específica
        modelo, tokenizador = cargar_modelo_optimizado(modelo_base, config)
        
        # Aplicar sliding window si está habilitado
        if config.get("sliding_window", False):
            aplicar_sliding_window(modelo, window_size=512)
        
        # Evaluar rendimiento
        metricas = evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo)
        resultados[nombre] = metricas
        
        # Mostrar resultados
        print(f"Tiempo de inferencia: {metricas['tiempo_inferencia_segundos']:.2f} segundos")
        print(f"Tokens generados: {metricas['tokens_generados']}")
        print(f"Velocidad: {metricas['tokens_por_segundo']:.2f} tokens/segundo")
        print(f"Memoria utilizada: {metricas['memoria_usada_mb']:.2f} MB")
        
        # Liberar memoria
        del modelo, tokenizador
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Comparar resultados
    print("\n\nComparación de configuraciones:")
    print(f"{'Configuración':<10} | {'Tiempo (s)':<10} | {'Tokens/s':<10} | {'Memoria (MB)':<10}")
    print(f"{'-'*50}")
    
    for nombre, metricas in resultados.items():
        print(f"{nombre:<10} | {metricas['tiempo_inferencia_segundos']:<10.2f} | {metricas['tokens_por_segundo']:<10.2f} | {metricas['memoria_usada_mb']:<10.2f}")
```

## Ejercicio 5: Personalización del Chatbot y Despliegue

```python
import gradio as gr
from peft import LoraConfig, get_peft_model, TaskType

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
    # Definir módulos a optimizar (principalmente layers de atención y MLP)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Crear configuración LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Aplicar PEFT al modelo
    modelo_peft = get_peft_model(modelo, lora_config)
    print(f"Modelo PEFT configurado con rango {r} y alpha {lora_alpha}")
    print(f"Parámetros entrenables: {modelo_peft.print_trainable_parameters()}")
    
    return modelo_peft

def guardar_modelo(modelo, tokenizador, ruta):
    """
    Guarda el modelo y tokenizador en una ruta específica.
    
    Args:
        modelo: Modelo a guardar
        tokenizador: Tokenizador del modelo
        ruta (str): Ruta donde guardar
    """
    # Crear directorio si no existe
    os.makedirs(ruta, exist_ok=True)
    
    # Guardar modelo
    modelo.save_pretrained(ruta)
    
    # Guardar tokenizador
    tokenizador.save_pretrained(ruta)
    
    print(f"Modelo y tokenizador guardados en: {ruta}")

def cargar_modelo_personalizado(ruta):
    """
    Carga un modelo personalizado desde una ruta específica.
    
    Args:
        ruta (str): Ruta del modelo
        
    Returns:
        tuple: (modelo, tokenizador)
    """
    # Cargar tokenizador
    tokenizador = AutoTokenizer.from_pretrained(ruta)
    
    # Configurar opciones de carga
    device_map = "auto"
    if torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Cargar modelo
    modelo = AutoModelForCausalLM.from_pretrained(
        ruta,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    
    # Establecer en modo evaluación
    modelo.eval()
    
    print(f"Modelo personalizado cargado desde: {ruta}")
    return modelo, tokenizador

# Interfaz web simple con Gradio
def crear_interfaz_web(chatbot):
    """
    Crea una interfaz web simple para el chatbot usando Gradio.
    
    Args:
        chatbot: Instancia del chatbot
        
    Returns:
        gr.Interface: Interfaz de Gradio
    """
    # Historial de conversación para la interfaz
    historial_chat = []
    
    # Función de callback para procesar entradas
    def responder(mensaje, history):
        history.append((mensaje, ""))
        respuesta = chatbot.responder(mensaje)
        history[-1] = (mensaje, respuesta)
        return "", history
    
    # Crear la interfaz con Gradio
    interfaz = gr.ChatInterface(
        fn=responder,
        title="Chatbot con LLM",
        description="Un chatbot inteligente basado en modelos de lenguaje de gran escala.",
        examples=[
            "¿Qué es la inteligencia artificial?",
            "Explícame cómo funciona un transformador",
            "¿Cuáles son las aplicaciones de los LLMs en la educación?"
        ],
        theme=gr.themes.Soft()
    )
    
    return interfaz

# Función principal para el despliegue
def main_despliegue():
    """
    Función principal para desplegar el chatbot en una interfaz web.
    """
    # Determinar ruta del modelo a cargar
    modelo_path = "./modelo_personalizado"
    modelo_base = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Verificar si existe un modelo personalizado guardado
    if os.path.exists(modelo_path):
        print("Cargando modelo personalizado...")
        modelo, tokenizador = cargar_modelo_personalizado(modelo_path)
    else:
        print("No se encontró modelo personalizado. Cargando modelo base optimizado...")
        # Cargar modelo base con optimizaciones
        modelo, tokenizador = cargar_modelo_optimizado(modelo_base, {
            "cuantizacion": True,
            "bits": 4,
            "flash_attention": True
        })
    
    # Crear instancia del chatbot
    chatbot = Chatbot(
        modelo_id=None,  # Ya tenemos el modelo cargado
        instrucciones_sistema="Eres un asistente IA amable y servicial. Proporcionas respuestas precisas, informativas y útiles. Mantienes tus respuestas concisas cuando es posible, pero detalladas cuando sea necesario."
    )
    
    # Asignar modelo y tokenizador ya cargados
    chatbot.modelo = modelo
    chatbot.tokenizador = tokenizador
    
    # Crear y lanzar la interfaz web
    interfaz = crear_interfaz_web(chatbot)
    
    # Configurar parámetros para el despliegue
    interfaz.launch(
        server_name="0.0.0.0",  # Disponible en la red
        server_port=7860,       # Puerto estándar de Gradio
        share=True              # Crear enlace público temporal
    )

if __name__ == "__main__":
    main_despliegue()
```

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
