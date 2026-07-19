"""
evaluador_ragas.py - Metricas complementarias con RAGAS
=========================================================
Lee dataset_completo.xlsx (generado por tu pipeline),
calcula metrics RAGAS y guarda dataset_con_ragas.xlsx.

NO modifica el archivo original.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    GEMINI_API_KEY,
    CHECKPOINT_PATH,
    OUTPUT_DIR,
)

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


RUTA_SALIDA = os.path.join(OUTPUT_DIR, "dataset_con_ragas.xlsx")


def cargar_datos() -> pd.DataFrame:
    """Carga el Excel generado por tu pipeline."""
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  [ERROR] No se encontro: {CHECKPOINT_PATH}")
        sys.exit(1)
    df = pd.read_excel(CHECKPOINT_PATH, engine="openpyxl")
    print(f"  Cargadas {len(df)} filas desde {CHECKPOINT_PATH}")
    return df


def preparar_para_ragas(df: pd.DataFrame) -> Dataset:
    """Convierte tu DataFrame al formato Dataset de RAGAS."""
    def parse_contexts(texto):
        if not texto or pd.isna(texto):
            return [""]
        partes = str(texto).split("---")
        return [p.strip() for p in partes if p.strip()] or [""]

    records = []
    for _, row in df.iterrows():
        records.append({
            "question": str(row.get("Pregunta", "")),
            "answer": str(row.get("Respuesta_Con_RAG", "")),
            "contexts": parse_contexts(row.get("Contexto_Recuperado", "")),
            "ground_truth": str(row.get("Respuesta_Referencia_Ground_Truth", "")),
        })

    dataset = Dataset.from_list(records)
    print(f"  Dataset RAGAS creado: {len(dataset)} muestras")
    return dataset


def configurar_modelos():
    """Configura LLM y embeddings de Gemini para RAGAS."""
    print(f"  LLM: gemini-3.1-flash-lite")
    print(f"  Embeddings: gemini-embedding-2")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=GEMINI_API_KEY,
    )
    return llm, embeddings


def ejecutar_ragas(dataset: Dataset, llm, embeddings) -> pd.DataFrame:
    """Ejecuta las metricas de RAGAS y devuelve un DataFrame con los scores."""
    print("\n  Ejecutando RAGAS...")
    print(f"    - faithfulness")
    print(f"    - answer_relevancy")
    print(f"    - context_precision")
    print(f"    - context_recall")

    resultado = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    df_result = resultado.to_pandas()
    print(f"\n  RAGAS completado. {len(df_result)} filas evaluadas.")
    print(f"  Columnas: {list(df_result.columns)}")
    return df_result


def mostrar_resumen(df_ragas: pd.DataFrame, df_original: pd.DataFrame):
    """Muestra medias de las metricas RAGAS y comparacion con metricas propias."""
    print("\n" + "=" * 55)
    print("  RESUMEN METRICAS RAGAS (ConRAG)")
    print("=" * 55)
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in metric_cols:
        if col in df_ragas.columns:
            media = df_ragas[col].mean()
            std = df_ragas[col].std()
            print(f"    {col:35s}  {media:.4f} +/- {std:.4f}")

    # Comparacion faithfulness RAGAS vs Groundedness propia
    if "Groundedness_ConRAG" in df_original.columns:
        print("\n  VALIDACION CRUZADA - Faithfulness:")
        f_ragas = df_ragas["faithfulness"].mean()
        f_propia = df_original["Groundedness_ConRAG"].mean()
        diff = abs(f_ragas - f_propia)
        print(f"    RAGAS faithfulness:      {f_ragas:.4f}")
        print(f"    Tu Groundedness (Gemma): {f_propia:.4f}")
        print(f"    Diferencia absoluta:     {diff:.4f}")
        if diff < 0.1:
            print(f"    >> CORRELACION ALTA: tus metricas son consistentes")
        elif diff < 0.2:
            print(f"    >> CORRELACION MODERADA: revisar diferencias metodologicas")
        else:
            print(f"    >> DIFERENCIA SIGNIFICATIVA: revisar prompts de evaluacion")


def integrar_resultados(df_original: pd.DataFrame, df_ragas: pd.DataFrame) -> pd.DataFrame:
    """Agrega las columnas de RAGAS al DataFrame original."""
    mapeo = {
        "faithfulness": "Faithfulness_RAGAS_ConRAG",
        "answer_relevancy": "Answer_Relevancy_ConRAG",
        "context_precision": "Context_Precision_ConRAG",
        "context_recall": "Context_Recall_ConRAG",
    }
    df_ragas_renamed = df_ragas.rename(columns=mapeo)

    df_completo = df_original.copy()
    for col_ragas in df_ragas_renamed.columns:
        df_completo[col_ragas] = df_ragas_renamed[col_ragas].values

    return df_completo


def analisis_categorias(df: pd.DataFrame):
    """Muestra metricas RAGAS por categoria."""
    if "Categoria" not in df.columns:
        return

    cols_ragas = [c for c in df.columns if "RAGAS" in c]
    if not cols_ragas:
        return

    print("\n" + "=" * 55)
    print("  METRICAS RAGAS POR CATEGORIA")
    print("=" * 55)

    categorias = df["Categoria"].dropna().unique()
    for cat in sorted(categorias):
        mask = df["Categoria"] == cat
        print(f"\n  [{cat}]")
        for col in cols_ragas:
            media = df.loc[mask, col].mean()
            print(f"    {col:35s}  {media:.4f}")


def main():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AQUI_TU_API_KEY":
        print("[ERROR] GEMINI_API_KEY no configurada.")
        return

    print("=" * 60)
    print("  RAGAS - Metricas complementarias")
    print("=" * 60)

    print("\n[1/5] Cargando datos del pipeline...")
    df_original = cargar_datos()

    print("\n[2/5] Preparando dataset para RAGAS...")
    dataset = preparar_para_ragas(df_original)

    print("\n[3/5] Configurando modelos Gemini...")
    llm, embeddings = configurar_modelos()

    print("\n[4/5] Ejecutando evaluacion...")
    df_ragas = ejecutar_ragas(dataset, llm, embeddings)

    print("\n[5/5] Integrando resultados...")
    df_final = integrar_resultados(df_original, df_ragas)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_excel(RUTA_SALIDA, index=False, engine="openpyxl")
    print(f"\n  Archivo guardado: {RUTA_SALIDA}")
    print(f"  Columnas originales: {len(df_original.columns)}")
    print(f"  Columnas RAGAS anadidas: {len(df_ragas.columns)}")
    print(f"  Total columnas: {len(df_final.columns)}")

    mostrar_resumen(df_ragas, df_original)
    analisis_categorias(df_final)

    print("\n" + "=" * 60)
    print("  COMPLETADO")
    print("=" * 60)
    print(f"  Tu archivo original NO fue modificado.")
    print(f"  Resultados RAGAS en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
