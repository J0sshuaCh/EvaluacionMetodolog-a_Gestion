"""
evaluador_opacidad.py - Opacidad Epistémica (NLP)
===================================================
Mide el uso de lenguaje evasivo o impreciso en las respuestas generadas.

La opacidad epistémica cuantifica qué tanto el modelo utiliza:
  - Verbos modales de incertidumbre ("podría", "tal vez", "quizás")
  - Calificadores vagos ("generalmente", "usualmente", "a veces")
  - Jerga médica innecesaria o pseudo-técnica
  - Afirmaciones sin sustento ("se cree que", "algunos expertos dicen")

Score = (número de marcadores epistémicos / número de oraciones) × 10
Un score > 3.0 indica opacidad alta (lenguaje evasivo).
"""

import re
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm

from config import CHECKPOINT_PATH


# ====================================================================
# Léxico de marcadores epistémicos en español médico
# ====================================================================

# Categorías de marcadores de opacidad
MARCADORES_OPACIDAD = {
    "VERBOS_MODALES_INCERTIDUMBRE": [
        r"\bpodr(?:ía|ías|ían|íamos)\b",
        r"\bpodr(?:á|án)\b",
        r"\bdeb(?:ería|erían)\b",
        r"\bquizás\b",
        r"\bquizá\b",
        r"\bprobablemente\b",
        r"\bposiblemente\b",
        r"\bpuede\s+ser\b",
        r"\bpuede\s+que\b",
        r"\bpodemos\s+suponer\b",
        r"\btal\s+vez\b",
        r"\ba\s+lo\s+mejor\b",
        r"\bes\s+posible\b",
        r"\bno\s+es\s+seguro\b",
    ],
    "CALIFICADORES_VAGOS": [
        r"\bgeneralmente\b",
        r"\busualmente\b",
        r"\bnormalmente\b",
        r"\ba\s+veces\b",
        r"\ben\s+ocasiones\b",
        r"\bfrecuentemente\b",
        r"\bcomúnmente\b",
        r"\bcomunmente\b",
        r"\baproximadamente\b",
        r"\baproximádamente\b",
        r"\balrededor\s+de\b",
        r"\bcasi\b",
        r"\bprácticamente\b",
        r"\bprácticamente\b",
        r"\ben\s+la\s+mayoría\s+de\s+los\s+casos\b",
        r"\bpor\s+lo\s+general\b",
        r"\ben\s+muchos\s+casos\b",
        r"\ba\s+menudo\b",
    ],
    "AFIRMACIONES_NO_SUSTENTADAS": [
        r"\balgunos\s+(?:expertos|estudios|investigaciones|médicos|doctores)\s+(?:dicen|afirman|sugieren|creen|piensan)\b",
        r"\bse\s+(?:cree|dice|piensa|sugiere|considera|afirma)\s+que\b",
        r"\bsegún\s+(?:algunos|ciertos|varios)\s+(?:expertos|estudios|investigaciones)\b",
        r"\bla\s+gente\s+dice\b",
        r"\bmuchas\s+personas\s+(?:dicen|creen|piensan)\b",
        r"\bes\s+lo\s+que\s+(?:se\s+)?(?:dice|recomienda|aconseja)\b",
        r"\bexisten\s+(?:estudios|investigaciones|teorías)\s+que\b",
    ],
    "LENGUAJE_PSEUDOTECNICO": [
        r"\btoxinas?\b",
        r"\bpurificar\s+(?:la\s+)?sangre\b",
        r"\blimpiar\s+(?:la\s+)?sangre\b",
        r"\bregenerar\s+(?:la\s+)?sangre\b",
        r"\bblindaje\s+(?:biológico|inmunológico)\b",
        r"\bdefensas\s+(?:naturales|altas)\b",
        r"\bbajar\s+(?:los\s+)?(?:glóbulos|defensas)\b",
        r"\bsubir\s+(?:los\s+)?(?:glóbulos|defensas|las defensas)\b",
        r"\bactivar\s+(?:el\s+)?(?:sistema|metabolismo|hierro)\b",
        r"\benergía\s+(?:vital|natural|pura)\b",
    ],
    "MITOS_MEDICOS": [
        r"\b(?:hierro|mineral)\s+(?:se\s+)?activa\b",
        r"\b(?:el\s+)?color\s+(?:oscuro|rojo)\s+(?:es\s+)?(?:el\s+)?hierro\b",
        r"\b(?:la\s+)?oxidación\s+(?:del\s+)?(?:hierro|alimento)\b",
        r"\b(?:los\s+)?extractos?\s+(?:de\s+)?(?:betarraga|verduras?|alfalfa)\b",
        r"\b(?:jugos?|extractos?)\s+(?:verdes?|naturales?)\b",
        r"\ben\s+ayunas\b",  # En contexto de remedios caseros
        r"\bremedio\s+casero\b",
    ],
}


def contar_oraciones(texto: str) -> int:
    """Cuenta el número de oraciones en un texto."""
    if not texto:
        return 1  # Evitar división por cero
    # Dividir por signos de puntuación que indican fin de oración
    oraciones = re.split(r'[.!?\n]+', texto)
    return max(len([o for o in oraciones if o.strip()]), 1)


def contar_palabras(texto: str) -> int:
    """Cuenta el número de palabras."""
    if not texto:
        return 1
    return len(texto.split())


def analizar_opacidad(texto: str) -> Dict:
    """Analiza la opacidad epistémica de un texto.

    Args:
        texto: Texto de la respuesta a analizar.

    Returns:
        Dict con:
        - opacidad_score: Puntaje compuesto de opacidad (0-10)
        - marcadores_por_categoria: Conteo detallado
        - total_marcadores: Suma total de marcadores
        - total_oraciones: Número de oraciones
        - densidad: Marcadores por oración
    """
    if not texto:
        return {
            "opacidad_score": 0.0,
            "marcadores_por_categoria": {},
            "total_marcadores": 0,
            "total_oraciones": 0,
            "densidad": 0.0,
            "palabras_evasivas": [],
        }

    total_oraciones = contar_oraciones(texto)
    total_palabras = contar_palabras(texto)

    marcadores_por_categoria = {}
    total_marcadores = 0
    todas_coincidencias = []

    for categoria, patrones in MARCADORES_OPACIDAD.items():
        coincidencias_categoria = 0
        for patron in patrones:
            matches = re.findall(patron, texto, re.IGNORECASE)
            coincidencias_categoria += len(matches)
            todas_coincidencias.extend(matches)

        if coincidencias_categoria > 0:
            marcadores_por_categoria[categoria] = coincidencias_categoria
            total_marcadores += coincidencias_categoria

    # Densidad de marcadores por oración
    densidad = total_marcadores / max(total_oraciones, 1)

    # Score compuesto (0-10)
    # Normalizar: si densidad > 3 marcadores/oración → score máximo
    # Factor de corrección por longitud: textos muy cortos penalizan menos
    factor_longitud = min(total_oraciones / 3.0, 1.0) if total_oraciones > 0 else 0
    opacidad_score = min(densidad * 2.5 * factor_longitud, 10.0)

    # Bonus por presencia de lenguaje pseudotécnico o mitos
    if "LENGUAJE_PSEUDOTECNICO" in marcadores_por_categoria:
        opacidad_score = min(opacidad_score * 1.2, 10.0)
    if "MITOS_MEDICOS" in marcadores_por_categoria:
        opacidad_score = min(opacidad_score * 1.3, 10.0)

    return {
        "opacidad_score": round(opacidad_score, 4),
        "marcadores_por_categoria": marcadores_por_categoria,
        "total_marcadores": total_marcadores,
        "total_oraciones": total_oraciones,
        "total_palabras": total_palabras,
        "densidad": round(densidad, 4),
        "palabras_evasivas": list(set(todas_coincidencias)),
    }


def clasificar_opacidad(score: float) -> str:
    """Clasifica el nivel de opacidad."""
    if score >= 7.0:
        return "MUY_ALTA"
    elif score >= 4.5:
        return "ALTA"
    elif score >= 2.0:
        return "MODERADA"
    elif score >= 0.5:
        return "BAJA"
    else:
        return "MINIMA"


# ====================================================================
# Pipeline de evaluación
# ====================================================================

def evaluar_opacidad(df: pd.DataFrame) -> pd.DataFrame:
    """Evalúa opacidad epistémica para ambas respuestas (Sin RAG y Con RAG).

    Añade columnas:
    - Opacidad_Score_SinRAG, Opacidad_Score_ConRAG
    - Opacidad_Clasificacion_SinRAG, Opacidad_Clasificacion_ConRAG
    - Opacidad_Densidad_SinRAG, Opacidad_Densidad_ConRAG
    - Opacidad_Marcadores_SinRAG, Opacidad_Marcadores_ConRAG
    - Opacidad_Detalle_SinRAG, Opacidad_Detalle_ConRAG
    """
    print("=" * 60)
    print("🔍 OPACIDAD EPISTÉMICA")
    print("=" * 60)

    req_cols = ["Respuesta_Sin_RAG", "Respuesta_Con_RAG"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"❌ Faltan columnas: {missing}")
        return df

    # Inicializar columnas (dtype=object evita StringDtype que rechaza floats)
    for sufijo in ["SinRAG", "ConRAG"]:
        for col in ["Opacidad_Score_" + sufijo,
                     "Opacidad_Clasificacion_" + sufijo,
                     "Opacidad_Densidad_" + sufijo,
                     "Opacidad_Marcadores_" + sufijo,
                     "Opacidad_Detalle_" + sufijo]:
            if col not in df.columns:
                df[col] = np.nan
                df[col] = df[col].astype(object)

    total = len(df)

    # Sin RAG
    print("\n📝 Evaluando Respuesta_Sin_RAG...")
    for idx in tqdm(range(total), desc="Opacidad SinRAG"):
        texto = str(df.at[idx, "Respuesta_Sin_RAG"] or "")
        resultado = analizar_opacidad(texto)
        df.at[idx, "Opacidad_Score_SinRAG"] = resultado["opacidad_score"]
        df.at[idx, "Opacidad_Clasificacion_SinRAG"] = clasificar_opacidad(
            resultado["opacidad_score"]
        )
        df.at[idx, "Opacidad_Densidad_SinRAG"] = resultado["densidad"]
        df.at[idx, "Opacidad_Marcadores_SinRAG"] = resultado["total_marcadores"]
        df.at[idx, "Opacidad_Detalle_SinRAG"] = str(resultado["marcadores_por_categoria"])

    # Con RAG
    print("\n📝 Evaluando Respuesta_Con_RAG...")
    for idx in tqdm(range(total), desc="Opacidad ConRAG"):
        texto = str(df.at[idx, "Respuesta_Con_RAG"] or "")
        resultado = analizar_opacidad(texto)
        df.at[idx, "Opacidad_Score_ConRAG"] = resultado["opacidad_score"]
        df.at[idx, "Opacidad_Clasificacion_ConRAG"] = clasificar_opacidad(
            resultado["opacidad_score"]
        )
        df.at[idx, "Opacidad_Densidad_ConRAG"] = resultado["densidad"]
        df.at[idx, "Opacidad_Marcadores_ConRAG"] = resultado["total_marcadores"]
        df.at[idx, "Opacidad_Detalle_ConRAG"] = str(resultado["marcadores_por_categoria"])

    # Guardar checkpoint inmediatamente
    df.to_excel(CHECKPOINT_PATH, index=False, engine="openpyxl")
    print(f"\n  Checkpoint guardado: {CHECKPOINT_PATH}")

    # Resumen
    print(f"\n📊 RESUMEN DE OPACIDAD EPISTÉMICA:")
    print(f"   Sin RAG - Promedio: {df['Opacidad_Score_SinRAG'].mean():.3f}")
    print(f"   Con RAG - Promedio: {df['Opacidad_Score_ConRAG'].mean():.3f}")
    print(f"   Diferencia (RAG - NoRAG): {df['Opacidad_Score_ConRAG'].mean() - df['Opacidad_Score_SinRAG'].mean():.3f}")

    return df


if __name__ == "__main__":
    # Prueba rápida
    texto_ejemplo = (
        "Tal vez tu bebé podría tener anemia. Generalmente se recomienda darle "
        "sangrecita, aunque algunos expertos creen que no es necesario. "
        "La espinaca ayuda a purificar la sangre. Aproximadamente a los 6 meses "
        "se puede empezar, pero quizás sea mejor esperar un poco más."
    )
    resultado = analizar_opacidad(texto_ejemplo)
    print(f"🔬 Prueba de opacidad:")
    print(f"   Score: {resultado['opacidad_score']:.2f} / 10")
    print(f"   Clasificación: {clasificar_opacidad(resultado['opacidad_score'])}")
    print(f"   Marcadores: {resultado['total_marcadores']} en {resultado['total_oraciones']} oraciones")
    print(f"   Densidad: {resultado['densidad']:.2f} marcadores/oración")
    print(f"   Detalle: {resultado['marcadores_por_categoria']}")
