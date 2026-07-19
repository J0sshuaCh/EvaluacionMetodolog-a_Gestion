"""
generador_respuestas.py - Genera respuestas Gemini sin/con RAG
===============================================================
Para cada pregunta del dataset:
  - SIN RAG: Llama a Gemini 3.1 Flash Lite directamente
  - CON RAG: Recupera contexto -> llama a Gemini 3.1 Flash Lite con contexto

Usa rate limiter propio (generacion) con RPD counter de 480/dia.
"""

import os
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from google import genai

from config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    RETRY_DELAY,
    MAX_RETRIES,
    MAX_RETRY_BACKOFF,
    OUTPUT_DIR,
    DATASET_PATH,
    CHECKPOINT_PATH,
)
from retrieval_local import RecuperadorLocal
from rate_limiter import (
    get_handler_generacion,
    RPDAgotadoError,
    rpd_status,
)


# ── Cliente Gemini ──────────────────────────────────────────────
_cliente = None


def get_cliente():
    global _cliente
    if _cliente is None:
        _cliente = genai.Client(api_key=GEMINI_API_KEY)
    return _cliente


def _llamada_gemini(prompt: str, descripcion: str = "") -> Optional[str]:
    """Llama a Gemini con rate limiting y RPD check."""
    client = get_cliente()
    handler = get_handler_generacion()

    for intento in range(1, MAX_RETRIES + 1):
        try:
            handler.limiter.wait()
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_TOKENS,
                    "top_p": 0.95,
                },
            )
            if response.text:
                return response.text.strip()
            else:
                print(f"  [AVISO] Respuesta vacia ({descripcion})")
                return ""
        except RPDAgotadoError:
            raise  # Propagar para que main.py maneje
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                espera = min(RETRY_DELAY * (2 ** intento), MAX_RETRY_BACKOFF)
                if intento == MAX_RETRIES:
                    print(f"  [ERROR] Cuota agotada tras {MAX_RETRIES} intentos ({descripcion}): {e}")
                    return None
                print(f"  [RATE] Cuota excedida (intento {intento}/{MAX_RETRIES}, espera {espera}s): {descripcion[:50]}")
                time.sleep(espera)
            else:
                if intento == MAX_RETRIES:
                    print(f"  [ERROR] Fallo tras {MAX_RETRIES} intentos ({descripcion}): {e}")
                    return None
                espera = RETRY_DELAY * intento
                print(f"  [RETRY] Intento {intento}/{MAX_RETRIES} ({descripcion[:50]}), espera {espera}s: {e}")
                time.sleep(espera)
    return None


# ── Prompts del sistema ─────────────────────────────────────────
SISTEMA_SIN_RAG = """Eres un asistente de salud peruano llamado ANMI (Asistente Nutricional Materno Infantil).
Responde EN ESPANOL de forma clara y util.
NO tienes acceso a documentos de referencia, asi que responde basandote en tu conocimiento general.
Si no estas seguro de algun dato medico especifico, indicarlo honestamente.
Manten la respuesta breve (maximo 3 parrafos) y en lenguaje comprensible para padres de familia."""

SISTEMA_CON_RAG = """Eres un asistente de salud peruano llamado ANMI (Asistente Nutricional Materno Infantil).
Responde EN ESPANOL de forma clara y util.

INSTRUCCIONES ESTRICTAS:
1. PRIORIZA la informacion de los DOCUMENTOS DE REFERENCIA que se te proporcionan.
2. SOLO puedes dar informacion que este respaldada por los documentos.
3. Si los documentos NO mencionan algo especifico, NO lo inventes.
4. CITA la fuente del documento cuando uses informacion de el.
5. Manten la respuesta breve (maximo 3 parrafos) y en lenguaje comprensible para padres de familia.
6. Si la pregunta no puede responderse con los documentos proporcionados, indicarlo claramente."""


def generar_sin_rag(pregunta: str) -> Optional[str]:
    """Genera respuesta SIN contexto RAG."""
    prompt = f"{SISTEMA_SIN_RAG}\n\nPREGUNTA DEL USUARIO:\n{pregunta}"
    return _llamada_gemini(prompt, descripcion=f"SinRAG: {pregunta[:50]}")


def generar_con_rag(pregunta: str, contexto: str) -> Optional[str]:
    """Genera respuesta CON contexto RAG."""
    if not contexto:
        prompt = (
            f"{SISTEMA_CON_RAG}\n\n"
            f"NOTA: No se encontraron documentos de referencia relevantes para esta consulta.\n"
            f"Indica al usuario que no hay informacion disponible en las guias oficiales.\n\n"
            f"PREGUNTA DEL USUARIO:\n{pregunta}"
        )
    else:
        prompt = (
            f"{SISTEMA_CON_RAG}\n\n"
            f"DOCUMENTOS DE REFERENCIA:\n{contexto}\n\n"
            f"PREGUNTA DEL USUARIO:\n{pregunta}"
        )
    return _llamada_gemini(prompt, descripcion=f"ConRAG: {pregunta[:50]}")


# ── Pipeline de generacion ──────────────────────────────────────
def generar_todas_respuestas(
    df: pd.DataFrame,
    recuperador: RecuperadorLocal,
    archivo_salida: str = None,
    solo_faltantes: bool = True,
) -> pd.DataFrame:
    """Genera respuestas Sin RAG y Con RAG para todas las preguntas.

    Si se agota el RPD, guarda checkpoint y propaga la excepcion.
    """
    print("=" * 60)
    print("GENERACION DE RESPUESTAS (Gemini 3.1 Flash Lite)")
    print("=" * 60)

    for col in ["Respuesta_Sin_RAG", "Respuesta_Con_RAG",
                "Contexto_Recuperado", "Fuentes_Recuperadas"]:
        if col not in df.columns:
            df[col] = ""

    # Asegurar dtype object para columnas de texto (evita LossySetitemError con NaN→float64)
    for col in ["Respuesta_Sin_RAG", "Respuesta_Con_RAG", "Contexto_Recuperado", "Fuentes_Recuperadas"]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    total = len(df)
    sin_rag_pendientes = sum(
        1 for r in df["Respuesta_Sin_RAG"]
        if r == "" or not r or (solo_faltantes and pd.isna(r))
    )
    con_rag_pendientes = sum(
        1 for r in df["Respuesta_Con_RAG"]
        if r == "" or not r or (solo_faltantes and pd.isna(r))
    )
    total_llamadas = sin_rag_pendientes + con_rag_pendientes

    print(f"  Preguntas: {total} | SinRAG pendientes: {sin_rag_pendientes} | "
          f"ConRAG pendientes: {con_rag_pendientes}")

    status = rpd_status()
    gen_status = status["generacion"]
    print(f"  RPD disponibles: {gen_status['restantes']}/{gen_status['limite']} "
          f"({gen_status['modelo']})")

    if total_llamadas == 0:
        print("  OK Todas las respuestas ya estan generadas.")
        return df

    pbar = tqdm(total=total_llamadas, desc="Generando")
    exitos_sin = 0
    exitos_con = 0
    errores = 0

    try:
        for idx, row in df.iterrows():
            pregunta = row["Pregunta"]
            pregunta_id = row["ID"]

            # ── Sin RAG ──
            r_sin = df.at[idx, "Respuesta_Sin_RAG"]
            if r_sin == "" or not r_sin or (solo_faltantes and pd.isna(r_sin)):
                pbar.set_description(f"SinRAG {pregunta_id}")
                respuesta = generar_sin_rag(pregunta)
                if respuesta is not None:
                    df.at[idx, "Respuesta_Sin_RAG"] = respuesta
                    exitos_sin += 1
                else:
                    errores += 1
                pbar.update(1)

                if archivo_salida and (exitos_sin + exitos_con) % 10 == 0:
                    _guardado_parcial(df, archivo_salida)

            # ── Con RAG ──
            r_con = df.at[idx, "Respuesta_Con_RAG"]
            if r_con == "" or not r_con or (solo_faltantes and pd.isna(r_con)):
                pbar.set_description(f"ConRAG {pregunta_id}")

                contexto, resultados = recuperador.recuperar_y_formatear(pregunta)
                df.at[idx, "Contexto_Recuperado"] = contexto
                df.at[idx, "Fuentes_Recuperadas"] = (
                    "; ".join([r["fuente"] for r in resultados])
                    if resultados else ""
                )

                respuesta = generar_con_rag(pregunta, contexto)
                if respuesta is not None:
                    df.at[idx, "Respuesta_Con_RAG"] = respuesta
                    exitos_con += 1
                else:
                    errores += 1
                pbar.update(1)

                if archivo_salida and (exitos_sin + exitos_con) % 10 == 0:
                    _guardado_parcial(df, archivo_salida)

    except RPDAgotadoError:
        pbar.close()
        print(f"\n  [RPD] Cuota diaria de generacion agotada.")
        print(f"  Generadas: {exitos_sin} SinRAG + {exitos_con} ConRAG")
        if archivo_salida:
            _guardado_parcial(df, archivo_salida)
            print(f"  Checkpoint guardado en: {archivo_salida}")
        print("  Reanuda manana con: python main.py --saltar-generacion")
        raise  # Propagar a main.py

    pbar.close()

    print(f"\n  OK Generacion completada:")
    print(f"    Sin RAG: {exitos_sin} | Con RAG: {exitos_con}")
    if errores:
        print(f"    Errores: {errores}")

    if archivo_salida:
        _guardado_parcial(df, archivo_salida)
        print(f"  Guardado en: {archivo_salida}")

    return df


def _guardado_parcial(df: pd.DataFrame, ruta: str):
    """Guarda el DataFrame parcial."""
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    df.to_excel(ruta, index=False, engine="openpyxl")


# ── Entry point ─────────────────────────────────────────────────
def main():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AQUI_TU_API_KEY":
        print("[ERROR] Configura GEMINI_API_KEY en config.py")
        return

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] No se encontro el dataset en: {DATASET_PATH}")
        print("  Ejecuta primero 'generar_dataset.py'")
        return

    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    print(f"Dataset cargado: {len(df)} preguntas")

    recuperador = RecuperadorLocal()
    if not recuperador.cargar():
        print("  [AVISO] Sin cache de documentos. Solo Sin RAG.")

    archivo_salida = CHECKPOINT_PATH
    try:
        df_resultado = generar_todas_respuestas(df, recuperador, archivo_salida=archivo_salida)
        print("\nListo para la fase de evaluacion.")
    except RPDAgotadoError:
        print("\nGeneracion pausada por limite RPD. Reanuda manana.")
        return 1


if __name__ == "__main__":
    main()
