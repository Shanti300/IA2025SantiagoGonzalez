# Implementación y Optimización de Chatbot con Modelos de Lenguaje de Gran Escala (LLM)
### Por Santiago González Olarte

> Este repositorio contiene la implementación completa de un sistema de chatbot basado en Modelos de Lenguaje de Gran Escala (LLM), desarrollado como solución a los ejercicios propuestos en el curso avanzado de NLP.

## Tabla de Contenidos
- [Ejercicio 1: Configuración del Entorno y Carga de Modelo Base](#ejercicio-1-configuración-del-entorno-y-carga-de-modelo-base)
- [Ejercicio 2: Procesamiento de Entrada y Generación de Respuestas](#ejercicio-2-procesamiento-de-entrada-y-generación-de-respuestas)
- [Ejercicio 3: Manejo de Contexto Conversacional](#ejercicio-3-manejo-de-contexto-conversacional)
- [Ejercicio 4: Optimización del Modelo para Recursos Limitados](#ejercicio-4-optimización-del-modelo-para-recursos-limitados)
- [Ejercicio 5: Personalización del Chatbot y Despliegue](#ejercicio-5-personalización-del-chatbot-y-despliegue)
- [Ejecución del Proyecto](#ejecución-del-proyecto)
- [Requisitos y Dependencias](#requisitos-y-dependencias)

## Ejercicio 1: Configuración del Entorno y Carga de Modelo Base

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configurar las variables de entorno para la caché de modelos
os.environ['TRANSFORMERS_CACHE'] = './models_cache'
os.environ['HF_HOME'] = './huggingface'

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.
    
    Args:
        nombre_modelo (str): Identificador del modelo en Hugging Face Hub
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    print(f"Cargando modelo: {nombre_modelo}")
    
    # Cargar tokenizador
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    
    # Cargar modelo
    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        torch_dtype=torch.float16,  # Usar half-precision para optimizar memoria
        device_map="auto"  # Gestionar automáticamente la distribución en dispositivos
    )
    
    # Configurar el modelo para inferencia
    modelo.eval()  # Establecer en modo evaluación
    
    return modelo, tokenizador

def verificar_dispositivo():
    """
    Verifica el dispositivo disponible (CPU/GPU) y muestra información relevante.
    
    Returns:
        torch.device: Dispositivo a utilizar
    """
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Memoria GPU disponible: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        dispositivo = torch.device("mps")
        print("Apple Silicon (MPS) disponible")
    else:
        dispositivo = torch.device("cpu")
        print("Utilizando CPU")
    
    return dispositivo

# Función principal de prueba
def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")
    
    # Cargar un modelo pequeño adecuado para chatbots
    modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Alternativas: "gpt2", "facebook/opt-350m"
    modelo, tokenizador = cargar_modelo(modelo_id)
    
    # Realizar una prueba simple de generación de texto
    prompt = "Explica brevemente qué es un modelo de lenguaje:"
    inputs = tokenizador(prompt, return_tensors="pt").to(dispositivo)
    
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )
    
    respuesta = tokenizador.decode(outputs[0], skip_special_tokens=True)
    print("\nPrueba de generación:")
    print(f"Prompt: {prompt}")
    print(f"Respuesta: {respuesta}")

if __name__ == "__main__":
    main()
```

## Ejercicio 2: Procesamiento de Entrada y Generación de Respuestas

```python
def preprocesar_entrada(texto, tokenizador, longitud_maxima=512, dispositivo=None):
    """
    Preprocesa el texto de entrada para pasarlo al modelo.
    
    Args:
        texto (str): Texto de entrada del usuario
        tokenizador: Tokenizador del modelo
        longitud_maxima (int): Longitud máxima de la secuencia
        dispositivo: Dispositivo donde se ejecutará el modelo
    
    Returns:
        torch.Tensor: Tensor de entrada para el modelo
    """
    # Tokenizar la entrada
    tokens = tokenizador(
        texto,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=longitud_maxima
    )
    
    # Pasar al dispositivo correspondiente si se proporciona
    if dispositivo:
        tokens = {k: v.to(dispositivo) for k, v in tokens.items()}
    
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
    # Valores por defecto para parámetros de generación
    if parametros_generacion is None:
        parametros_generacion = {
            "max_new_tokens": 256,       # Máximo de tokens nuevos a generar
            "temperature": 0.7,          # Control de aleatoriedad (0.0-1.0)
            "top_p": 0.9,                # Muestreo de núcleo (nucleus sampling)
            "do_sample": True,           # Usar muestreo estocástico
            "repetition_penalty": 1.2,   # Penalización por repetición
            "num_beams": 1,              # Usar búsqueda codiciosa (greedy)
            "early_stopping": True       # Detener generación temprana
        }
    
    # Generar respuesta con los parámetros especificados
    with torch.no_grad():
        salida = modelo.generate(
            **entrada_procesada,
            **parametros_generacion
        )
    
    # Decodificar la salida
    respuesta_completa = tokenizador.decode(salida[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta (eliminar el prompt original)
    prompt_original = tokenizador.decode(entrada_procesada['input_ids'][0], skip_special_tokens=True)
    respuesta = respuesta_completa[len(prompt_original):].strip()
    
    return respuesta

def crear_prompt_sistema(instrucciones):
    """
    Crea un prompt de sistema para dar instrucciones al modelo.
    
    Args:
        instrucciones (str): Instrucciones sobre cómo debe comportarse el chatbot
    
    Returns:
        str: Prompt formateado
    """
    # Crear un formato común para modelos de instrucción
    prompt_sistema = f"""<s>[SYSTEM]
{instrucciones}
[/SYSTEM]

"""
    return prompt_sistema

# Ejemplo de uso
def interaccion_simple():
    dispositivo = verificar_dispositivo()
    modelo, tokenizador = cargar_modelo("mistralai/Mistral-7B-Instruct-v0.2")
    
    # Crear un prompt de sistema para definir la personalidad del chatbot
    instrucciones = "Eres un asistente amable y servicial que proporciona respuestas concisas y correctas."
    prompt_sistema = crear_prompt_sistema(instrucciones)
    
    # Entrada del usuario de ejemplo
    entrada_usuario = "¿Cuáles son los principales beneficios de la inteligencia artificial?"
    
    # Combinar el prompt del sistema con la entrada del usuario
    prompt_completo = f"{prompt_sistema}[USER] {entrada_usuario} [/USER]\n[ASSISTANT]"
    
    # Procesar la entrada
    entrada_procesada = preprocesar_entrada(prompt_completo, tokenizador, dispositivo=dispositivo)
    
    # Generar la respuesta
    respuesta = generar_respuesta(
        modelo, 
        entrada_procesada, 
        tokenizador,
        {"max_new_tokens": 300, "temperature": 0.7}
    )
    
    print(f"Prompt: {entrada_usuario}")
    print(f"Respuesta: {respuesta}")

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

## Ejecución del Proyecto

Para ejecutar el proyecto completo, sigue estos pasos:

1. **Configuración del entorno**: Primero asegúrate de tener todas las dependencias instaladas (ver sección de Requisitos).

2. **Ejecución del chatbot básico**:
   ```bash
   # Ejecutar la versión básica del chatbot
   python chatbot_basic.py
   ```

3. **Ejecución con optimizaciones**:
   ```bash
   # Ejecutar el demo de optimizaciones para comparar rendimiento
   python chatbot_optimized.py --demo
   ```

4. **Despliegue de la interfaz web**:
   ```bash
   # Lanzar la interfaz web con Gradio
   python chatbot_web.py
   ```

## Requisitos y Dependencias

Para ejecutar este proyecto necesitarás las siguientes dependencias:

```bash
# Instalar dependencias principales
pip install torch transformers accelerate bitsandbytes

# Para optimizaciones y PEFT
pip install peft optimum

# Para la interfaz web
pip install gradio
```

### Requisitos de Hardware Recomendados

- **GPU**: NVIDIA con al menos 8GB de VRAM para modelos de 7B parámetros con optimizaciones
- **RAM**: Mínimo 16GB
- **Almacenamiento**: Al menos 20GB libres para modelos y caché

### Configuración Opcional

Si deseas utilizar modelos más grandes o mejorar el rendimiento:

```bash
# Para habilitar Flash Attention 2
pip install flash-attn --no-build-isolation

# Para métricas avanzadas de rendimiento
pip install psutil gputil
```

### Notas Adicionales

- Los modelos se descargarán automáticamente la primera vez que ejecutes el código
- Puedes modificar los paths de caché en el código según tus necesidades
- Para entrenar adaptadores LoRA personalizados, necesitarás datos específicos para tu caso de uso
