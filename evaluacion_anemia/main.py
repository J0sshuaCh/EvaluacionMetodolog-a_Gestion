"""
main.py - ORQUESTADOR PRINCIPAL
=================================
Pipeline completo de evaluacion:

  1. Verificar/generar dataset
  2. Procesar documentos PDF -> chunks -> embeddings
  3. Generar respuestas (Sin RAG y Con RAG) con Gemini 3.1 Flash Lite
  4. Evaluar metricas (Groundedness + Concordancia) con Gemma 4 31B
  5. Evaluar Opacidad Epistemica (NLP)
  6. Evaluar Potencial de Engano (PoD) con Gemma 4 31B + NLP
  7. Exportar Excel final + graficos

Uso:
  python main.py                          # Ejecucion completa
  python main.py --status                 # Ver RPD disponible
  python main.py --solo-exportar          # Solo exportar Excel
  python main.py --saltar-generacion      # Reanudar desde evaluacion
  python main.py --saltar-evaluacion      # Solo exportar desde parcial
  python main.py --forzar-documentos      # Reprocesar PDFs
  python main.py --forzar-dataset         # Regenerar CSV desde notebook
"""

import os
import sys
import io
import time
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    EVALUATION_MODEL,
    EMBEDDING_MODEL,
    DATASET_PATH,
    OUTPUT_DIR,
    CHECKPOINT_PATH,
)
from procesar_documentos import procesar_documentos, cache_existe
from retrieval_local import RecuperadorLocal
from generador_respuestas import generar_todas_respuestas, RPDAgotadoError as GenRPDError
from evaluador_metricas import evaluar_todo as evaluar_metricas
from evaluador_opacidad import evaluar_opacidad
from evaluador_pod import evaluar_pod
from consolidacion import exportar_excel
from rate_limiter import rpd_status


BANNER = """
======================================================================
     EVALUACION ANEMIA - Pipeline RAG
     Modelos:
       Generacion : Gemini 3.1 Flash Lite (480 RPD)
       Evaluacion : Gemma 4 31B it       (1480 RPD)
       Embeddings : Gemini Embedding 1   (990 RPD)
     Metricas: Groundedness, Concordancia,
               Opacidad Epistemica, Potencial de Engano
======================================================================
"""


# ── Utilidades ───────────────────────────────────────────────────

def _print_rpd_status():
    """Muestra estado de RPD de todos los modelos."""
    status = rpd_status()
    print("\n  Estado de RPD (hoy):")
    for key, s in status.items():
        pct = ((s["limite"] - s["restantes"]) / s["limite"]) * 100
        barra = "=" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"    {key:12s} [{barra}] {s['usadas']:4d}/{s['limite']}  ({s['restantes']} restantes)")
    print()


def verificar_api_key():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AQUI_TU_API_KEY":
        print("[ERROR] GEMINI_API_KEY no configurada en config.py")
        print("  Obten una en: https://aistudio.google.com/app/apikey")
        return False
    return True


def verificar_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"\n  Dataset no encontrado. Generando desde notebook...")
        from generar_dataset import main as gen_dataset
        gen_dataset()
        if not os.path.exists(DATASET_PATH):
            print("  [ERROR] No se pudo generar el dataset.")
            return False
    return True


# ── Pipelines por paso ───────────────────────────────────────────

def paso_dataset(forzar: bool = False) -> bool:
    print("\n[PASO 0] Verificar/Generar Dataset")
    print("-" * 50)
    if forzar or not os.path.exists(DATASET_PATH):
        from generar_dataset import main as gen_dataset
        gen_dataset()
    if verificar_dataset():
        df = pd.read_csv(DATASET_PATH, encoding="utf-8")
        cats_sospechosas = [
            c for c in df["Categoria"].unique()
            if len(str(c)) > 60
        ]
        if cats_sospechosas:
            print(f"  [AVISO] Categorias sospechosas: {cats_sospechosas}")
            print("  Revisa construccion_del_dataset.ipynb")
        print(f"  OK Dataset: {len(df)} preguntas")
        print(f"  Categorias: {dict(df['Categoria'].value_counts())}")
        return True
    return False


def paso_documentos(forzar: bool = False) -> bool:
    print("\n[PASO 1] Procesar Documentos PDF")
    print("-" * 50)
    _print_rpd_status()
    chunks, embeddings, metadata = procesar_documentos(forzar_reprocesar=forzar)
    if chunks is None:
        print("  [AVISO] Sin documentos. Con RAG se generara sin contexto.")
        return False
    return True


def paso_generar_respuestas_desde_csv(recuperador=None) -> pd.DataFrame:
    print("\n[PASO 2] Generar Respuestas")
    print("-" * 50)
    print(f"  Modelo: {GENERATION_MODEL}")
    _print_rpd_status()

    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    print(f"  Dataset: {len(df)} preguntas")

    if recuperador is None:
        recuperador = RecuperadorLocal()
        hay_cache = cache_existe()
        if hay_cache:
            recuperador.cargar()
        else:
            print("  [AVISO] Sin cache de documentos. Solo respuestas Sin RAG.")

    df = generar_todas_respuestas(
        df, recuperador,
        archivo_salida=CHECKPOINT_PATH,
        solo_faltantes=True,
    )
    return df


def paso_evaluar_metricas(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PASO 3] Evaluar Metricas (Groundedness + Concordancia)")
    print("-" * 50)
    print(f"  Modelo: {EVALUATION_MODEL}")
    _print_rpd_status()
    df = evaluar_metricas(df, archivo_salida=CHECKPOINT_PATH)
    return df


def paso_evaluar_opacidad(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PASO 4] Evaluar Opacidad Epistemica")
    print("-" * 50)
    print("  Metodo: NLP puro (sin llamadas API)")
    df = evaluar_opacidad(df)
    return df


def paso_evaluar_pod(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PASO 5] Evaluar Potencial de Engano (PoD)")
    print("-" * 50)
    _print_rpd_status()
    df = evaluar_pod(df)
    return df


def paso_exportar(df: pd.DataFrame):
    print("\n[PASO 6] Exportar Resultados y Graficos")
    print("-" * 50)
    ruta = exportar_excel(df)
    print(f"  OK Excel: {ruta}")

    # Generar graficos si es posible
    try:
        from generador_graficos import generar_graficos
        ruta_graficos = generar_graficos(df)
        print(f"  OK Graficos: {ruta_graficos}")
    except Exception as e:
        print(f"  [AVISO] No se generaron graficos: {e}")
    return ruta


# ── Pipeline completo ────────────────────────────────────────────

def ejecutar_pipeline_completo(
    forzar_docs: bool = False,
    saltar_generacion: bool = False,
    saltar_evaluacion: bool = False,
    forzar_dataset: bool = False,
):
    """Ejecuta el pipeline con manejo de RPD.

    Si un paso agota el RPD, guarda checkpoint y avisa.
    Al reanudar, los pasos completados se saltan automaticamente.
    """
    print(BANNER)
    inicio = time.time()

    # ── Paso 0: Dataset ──
    if not paso_dataset(forzar=forzar_dataset):
        return

    # ── Paso 1: Documentos ──
    paso_documentos(forzar=forzar_docs)

    # Preparar recuperador (cache de documentos para ConRAG)
    recuperador = RecuperadorLocal()
    hay_cache = cache_existe()
    if hay_cache:
        recuperador.cargar()

    # Cargar checkpoint o generar desde CSV
    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_excel(CHECKPOINT_PATH, engine="openpyxl")
        print(f"\n[PASO 2] Cargado desde checkpoint ({len(df)} filas)")

        col_con = "Respuesta_Con_RAG"
        pendientes = sum(1 for r in df[col_con] if pd.isna(r) or r == "")
        if pendientes > 0:
            print(f"  {pendientes} ConRAG pendientes. Regenerando...")
            try:
                df = generar_todas_respuestas(
                    df, recuperador,
                    archivo_salida=CHECKPOINT_PATH,
                    solo_faltantes=True,
                )
            except GenRPDError:
                print("\n  [RPD] Generacion pausada por limite diario.")
                print("  Manana ejecuta: python main.py --saltar-generacion")
                return
    else:
        try:
            df = paso_generar_respuestas_desde_csv(recuperador)
        except GenRPDError:
            print("\n  [RPD] Generacion pausada por limite diario.")
            print("  Manana ejecuta: python main.py --saltar-generacion")
            return

    # ── Paso 3: Metricas ──
    ya_evaluado = (
        "Groundedness_ConRAG" in df.columns
        and df["Groundedness_ConRAG"].notna().sum() > 0
    )
    if saltar_evaluacion or ya_evaluado:
        if ya_evaluado:
            print("\n[PASO 3] Metricas ya calculadas. Saltando.")
        else:
            print("\n[PASO 3] Saltado (--saltar-evaluacion)")
    else:
        try:
            df = paso_evaluar_metricas(df)
        except GenRPDError:
            print("\n  [RPD] Evaluacion pausada por limite diario.")
            print("  Manana ejecuta: python main.py --saltar-generacion --saltar-evaluacion")
            return

    # ── Paso 4: Opacidad (NLP, no gasta RPD) ──
    opacidad_hecho = "Opacidad_Score_SinRAG" in df.columns and df["Opacidad_Score_SinRAG"].notna().sum() > 0
    if opacidad_hecho:
        print("\n[PASO 4] Opacidad ya calculada. Saltando.")
    else:
        df = paso_evaluar_opacidad(df)

    # ── Paso 5: PoD ──
    pod_hecho = "PoD_Score_SinRAG" in df.columns and df["PoD_Score_SinRAG"].notna().sum() > 0
    if saltar_evaluacion or pod_hecho:
        if pod_hecho:
            print("\n[PASO 5] PoD ya calculado. Saltando.")
        else:
            print("\n[PASO 5] Saltado (--saltar-evaluacion)")
    else:
        try:
            df = paso_evaluar_pod(df)
        except GenRPDError:
            print("\n  [RPD] PoD pausado por limite diario.")
            print("  Manana ejecuta: python main.py --saltar-generacion")
            return

    # ── Paso 6: Exportar ──
    paso_exportar(df)

    duracion = time.time() - inicio
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"  Duracion: {duracion/60:.1f} min")
    print(f"  Preguntas: {len(df)}")
    _print_rpd_status()

    print("\n  Proximo paso:")
    print("    python main.py --solo-exportar   (regenerar Excel)")


def solo_exportar():
    """Solo genera Excel final desde datos existentes."""
    print(BANNER)
    print("\nModo: Solo exportar Excel y graficos")

    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_excel(CHECKPOINT_PATH, engine="openpyxl")
        print(f"  Cargado: {CHECKPOINT_PATH} ({len(df)} filas)")
    elif os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, encoding="utf-8")
        print(f"  Cargado dataset original: {len(df)} preguntas (sin metricas)")
    else:
        print("[ERROR] No hay datos para exportar.")
        return

    paso_exportar(df)


def mostrar_status():
    """Muestra estado de RPD y progreso."""
    print(BANNER)
    _print_rpd_status()

    # Mostrar progreso si existe checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_excel(CHECKPOINT_PATH, engine="openpyxl")
        print(f"Checkpoint: {CHECKPOINT_PATH}")
        print(f"  Filas: {len(df)}")
        cols_con_datos = [c for c in df.columns if c not in ["ID", "Categoria", "Pregunta"]]
        for col in cols_con_datos:
            no_nulos = df[col].dropna().count()
            total = len(df)
            pct = (no_nulos / total) * 100
            if no_nulos > 0:
                print(f"  {col}: {no_nulos}/{total} ({pct:.0f}%)")
    else:
        print("No hay checkpoint de progreso.")


# ── Entry point ──────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not verificar_api_key():
        return

    if "--status" in args:
        mostrar_status()
        return

    if "--solo-exportar" in args:
        solo_exportar()
        return

    ejecutar_pipeline_completo(
        forzar_docs="--forzar-documentos" in args,
        saltar_generacion="--saltar-generacion" in args,
        saltar_evaluacion="--saltar-evaluacion" in args,
        forzar_dataset="--forzar-dataset" in args,
    )


if __name__ == "__main__":
    main()
