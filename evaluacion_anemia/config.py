"""
config.py - Configuracion central del evaluador
================================================
Ajusta estos valores segun tu entorno antes de ejecutar main.py.
"""

import os

# ===================== CLAVE API GEMINI =====================
GEMINI_API_KEY = "AIzaSyBNA6CEtjBXos7WkbL3vwUEA63Hw-K296M"

# ===================== RUTAS DE ARCHIVOS =====================
DOCUMENTS_DIR = r"C:\Users\I5\Documents\Proyectos Web\Chatbot-Nutricional\documents"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_documentos")

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset_anemia_100.csv")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")

# ===================== PARAMETROS DE CHUNKING =====================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

# ===================== MODELOS GEMINI =====================
EMBEDDING_MODEL = "gemini-embedding-2"          # 1,000 RPD / 100 RPM
GENERATION_MODEL = "gemini-3.1-flash-lite"        # 500 RPD / 15 RPM
EVALUATION_MODEL = "gemma-4-31b-it"               # 1,500 RPD / 15 RPM

# ===================== PARAMETROS DE GENERACION =====================
TEMPERATURE = 0.3
MAX_TOKENS = 1024
RETRY_DELAY = 3
MAX_RETRIES = 10
MAX_RETRY_BACKOFF = 60

# ===================== RATE LIMITING POR MODELO =====================
# Gemini 3.1 Flash Lite (generacion) — 15 RPM, 500 RPD
RPM_GENERACION = 14           # Margen sobre 15
RPD_GENERACION = 480          # Margen sobre 500

# Gemma 4 31B (evaluacion) — 15 RPM, 1,500 RPD
RPM_EVALUACION = 14           # Margen sobre 15
RPD_EVALUACION = 1480         # Margen sobre 1500

# Embeddings — 100 RPM, 1,000 RPD (nunca es bottleneck)
DELAY_ENTRE_EMBEDDINGS = 3.0

# Ventana de rate limiting (siempre 60s)
RATE_LIMIT_WINDOW_SECONDS = 60

# ===================== DELAYS ENTRE LLAMADAS =====================
# Se calculan automaticamente desde RPM: 60s / RPM
DELAY_ENTRE_GENERACION = round(60.0 / RPM_GENERACION, 1)    # ~4.3s
DELAY_ENTRE_EVALUACION = round(60.0 / RPM_EVALUACION, 1)    # ~4.3s
DELAY_ENTRE_GROUPS = 8.0
GROUP_SIZE = 8

# ===================== ARCHIVOS DE ESTADO =====================
RPD_COUNTER_PATH = os.path.join(os.path.dirname(__file__), ".rpd_counter.json")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "dataset_completo.xlsx")
